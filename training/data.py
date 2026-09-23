import os
import hashlib
import wfdb

from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .evaluation.metadata import zero_shot_class
from .distributed import is_master

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class ECGTextDataset(Dataset):
    def __init__(
        self,
        path,
        texts,
        transforms=None,
        tokenizer=None,
        is_train=True,
        use_ccar=False,
        concept_weighter=None,
        return_weight=False,
        sample_weights=None,
    ):

        super().__init__()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.path = path
        self.y = texts
        self.is_train = is_train
        self.use_ccar = bool(use_ccar)
        self.concept_weighter = concept_weighter
        self.return_weight = bool(return_weight)
        self.sample_weights = None

        if self.use_ccar:
            if not self.is_train:
                raise ValueError("CCAR weights should only be enabled for training dataset.")
            if self.y is None:
                raise ValueError("CCAR requires text captions.")

            if sample_weights is None:
                raise RuntimeError(
                    "use_ccar=True but sample_weights is None. "
                    "Please build/load CCAR cache before constructing ECGTextDataset."
                )

            self.sample_weights = np.asarray(sample_weights, dtype=np.float32)

            if len(self.sample_weights) != len(self.y):
                raise RuntimeError(
                    f"CCAR weight length mismatch: "
                    f"weights={len(self.sample_weights)} texts={len(self.y)}"
                )

            ws = self.sample_weights
            print(
                f"[CCAR][INIT] samples={len(ws)} | "
                f"mean={ws.mean():.3f} std={ws.std():.3f} "
                f"min={ws.min():.3f} max={ws.max():.3f}",
                flush=True,
            )

    def tokenize(self, text):
        if text is None:
            raise RuntimeError("tokenize() called but text is None")
        text = text.lower()
        encoded = self.tokenizer(text)
        return encoded[0]

    def load_data(self, idx):
        data = wfdb.rdsamp(self.path[idx])[0]
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        data = torch.Tensor(np.transpose(data, (1, 0)).astype(np.float32))
        data = torch.unsqueeze(data, 0)
        if self.transforms is not None:
            data = self.transforms(data)
        data = torch.squeeze(data, 0)
        return data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.load_data(idx)
        y = self.y[idx]
        tokens = self.tokenize(y)

        if self.return_weight:
            w = self.sample_weights[idx] if self.sample_weights is not None else 1.0
            return x, tokens, torch.tensor(w, dtype=torch.float32)

        return x, tokens

class ECGValDataset(ECGTextDataset):
    def __init__(self, dir, path, diagnostics, transforms=None, tokenizer=None):
        abs_path = [os.path.join(dir, p) for p in path]
        super().__init__(
            abs_path,
            None,
            transforms=transforms,
            tokenizer=tokenizer,
            is_train=False,
            use_ccar=False,
            concept_weighter=None,
            return_weight=False,
        )
        self.diagnostics = diagnostics

    def __len__(self):
        return self.diagnostics.shape[0]

    def __getitem__(self, idx):
        x = self.load_data(idx)
        diagnostic = self.diagnostics[idx, :]
        return x, diagnostic


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for batch in dataloader:
        images, texts = batch[0], batch[1]
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def load_mimic_iv_ecg(path, train_stage):
    # --------------------------------------------------
    # 1. Load CSVs (stage controls text SOURCE ONLY)
    # --------------------------------------------------
    if train_stage == "stage1":
        record_list = pd.read_csv(os.path.join(path, "mimic_iv_ecg_clip.csv"))
    elif train_stage == "stage2":
        record_list = pd.read_csv(os.path.join(path, "mimic_iv_ecg_caption_new.csv"))
    else:
        raise ValueError(f"Unknown train_stage: {train_stage}")

    # --------------------------------------------------
    # 2. Fixed split (UNCHANGED)
    # --------------------------------------------------
    train_list = record_list.loc[
        np.where(record_list["file_name"].values % 10 > 0)
    ].set_index("study_id")

    test_list = record_list.loc[
        np.where(record_list["file_name"].values % 10 == 0)
    ].set_index("study_id")

    train_indexes = train_list.index.values
    val_indexes = test_list.index.values[-10000:-5000]
    test_indexes = test_list.index.values[-5000:]
    train_indexes = np.append(train_indexes, test_indexes[:-10000])

    print(
        f"[MIMIC][Split] raw counts | "
        f"train={len(train_indexes)} val={len(val_indexes)} test={len(test_indexes)}"
    )

    # --------------------------------------------------
    # 3. Bad report patterns (UNCHANGED)
    # --------------------------------------------------
    bad_reports = [
        "--- Warning: Data quality may affect interpretation ---",
        "--- Recording unsuitable for analysis - please repeat ---",
        "Analysis error",
        "conduction defect",
        "*** report made without knowing patient's sex ***",
        "--- Suspect arm lead reversal",
        "--- Possible measurement error ---",
        "--- Pediatric criteria used ---",
        "--- Suspect limb lead reversal",
        "-------------------- Pediatric ECG interpretation --------------------",
        "Lead(s) unsuitable for analysis:",
        "LEAD(S) UNSUITABLE FOR ANALYSIS:",
        "PACER DETECTION SUSPENDED DUE TO EXTERNAL NOISE-REVIEW ADVISED",
        "Pacer detection suspended due to external noise-REVIEW ADVISED"
    ]

    # --------------------------------------------------
    # 4. Text cleaning (UNCHANGED, shared by all stages)
    # --------------------------------------------------
    def clean_text(txt):
        if not isinstance(txt, str):
            return ""

        for bad in bad_reports:
            if bad in txt:
                return ""

        txt = txt.lower()
        txt = (txt.replace("---", "")
                   .replace("***", "")
                   .replace(" - age undetermined", ""))
        txt = (txt.replace('rbbb', 'right bundle branch block')
                .replace('lbbb', 'left bundle branch block')
                .replace('lvh', 'left ventricular hypertrophy')
                .replace("mi", "myocardial infarction")
                .replace("lafb", "left anterior fascicular block")
                .replace("pvc(s)", "ventricular premature complex")
                .replace("pvcs", "ventricular premature complex")
                .replace("pac(s)", "atrial premature complex")
                .replace("pacs", "atrial premature complex"))
        return txt.strip()

    # --------------------------------------------------
    # 5. Build paired (X, text) with strong debug
    # --------------------------------------------------
    def build_pairs(index_list, split_name):
        X, texts = [], []
        dropped_empty = 0
        dropped_non_str = 0

        for sid in index_list:
            row = record_list.loc[record_list["study_id"] == sid].iloc[0]

            # ---- choose raw text (SOURCE ONLY differs by stage)
            if train_stage == "stage1":
                raw_txt = row.get("clip_text", None)
            elif train_stage == "stage2":
                raw_txt = row.get("caption_text", None)
            else:
                raise ValueError(f"Unknown train_stage: {train_stage}")

            if not isinstance(raw_txt, str):
                dropped_non_str += 1
                continue

            clean_txt = clean_text(raw_txt)
            if clean_txt == "":
                dropped_empty += 1
                continue

            X.append(os.path.join(path, row["path"]))
            texts.append(clean_txt)

        print(
            f"[MIMIC][{train_stage}][{split_name}] "
            f"kept={len(X)} "
            f"dropped_non_str={dropped_non_str} "
            f"dropped_empty={dropped_empty}"
        )

        if len(X) == 0:
            print(f"❌ [MIMIC][{train_stage}][{split_name}] NO VALID SAMPLES!")

        return X, texts

    # --------------------------------------------------
    # 6. Build splits
    # --------------------------------------------------
    train_x, train_y = build_pairs(train_indexes, "train")
    val_x,   val_y   = build_pairs(val_indexes,   "val")
    test_x,  test_y  = build_pairs(test_indexes,  "test")

    # --------------------------------------------------
    # 7. Final sanity check
    # --------------------------------------------------
    assert len(train_x) == len(train_y)
    assert len(val_x)   == len(val_y)
    assert len(test_x)  == len(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_ptbxl(path, sampling_rate=500, train_stage="stage1"):


    # --------------------------------------------------
    # 1. Load CSVs (text SOURCE differs by stage only)
    # --------------------------------------------------
    if train_stage == "stage1":
        df = pd.read_csv(os.path.join(path, "ptbxl_database_clip.csv"), index_col="ecg_id")
    elif train_stage == "stage2":
        df = pd.read_csv(os.path.join(path, "ptbxl_database_caption_new.csv"), index_col="ecg_id")
    else:
        raise ValueError(f"Unknown train_stage: {train_stage}")
    # --------------------------------------------------
    # 2. Fixed official split (UNCHANGED)
    # --------------------------------------------------
    train_df = df[df.strat_fold.isin([1, 2, 3, 4, 5, 6, 7, 8])]
    val_df   = df[df.strat_fold == 9]
    test_df  = df[df.strat_fold == 10]

    print(
        f"[PTBXL][Split raw] "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    # --------------------------------------------------
    # 3. Shared text cleaning (ALL stages)
    # --------------------------------------------------
    def clean_text(t):
        if not isinstance(t, str):
            return ""

        t = t.lower()
        t = (
            t.replace("ekg", "ecg")
             .replace("normales ", "")
             .replace("4.46 ", "")
             .replace("unconfirmed report", "")
             .replace("unconfirmed", "")
             .replace(" age undetermined", "")
        )
        return t.strip()

    # --------------------------------------------------
    # 4. Build paired samples with debug
    # --------------------------------------------------
    def build_pairs(sub_df, split_name):
        X, texts = [], []
        dropped_non_str = 0
        dropped_empty = 0

        for _, row in sub_df.iterrows():
            # ---- waveform path
            if sampling_rate == 500:
                wave_rel = row.filename_hr
            else:
                wave_rel = row.filename_lr

            # ---- select raw text (SOURCE only)
            if train_stage == "stage1":
                raw_txt = row.get("clip_text", None)
            elif train_stage == "stage2":
                raw_txt = row.get("caption_text", None)
            else:
                raise ValueError(f"Unknown train_stage: {train_stage}")

            if not isinstance(raw_txt, str):
                dropped_non_str += 1
                continue

            clean_txt = clean_text(raw_txt)
            if clean_txt == "":
                dropped_empty += 1
                continue

            X.append(os.path.join(path, wave_rel))
            texts.append(clean_txt)

        print(
            f"[PTBXL][{train_stage}][{split_name}] "
            f"kept={len(X)} "
            f"dropped_non_str={dropped_non_str} "
            f"dropped_empty={dropped_empty}"
        )

        if len(X) == 0:
            print(f"❌ [PTBXL][{train_stage}][{split_name}] NO VALID SAMPLES!")

        return X, texts

    # --------------------------------------------------
    # 5. Build splits (PAIR-WISE)
    # --------------------------------------------------
    train_x, train_y = build_pairs(train_df, "train")
    val_x,   val_y   = build_pairs(val_df,   "val")
    test_x,  test_y  = build_pairs(test_df,  "test")

    # --------------------------------------------------
    # 6. Final sanity check
    # --------------------------------------------------
    assert len(train_x) == len(train_y)
    assert len(val_x)   == len(val_y)
    assert len(test_x)  == len(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y



def load_ptbxl_diagnostics(path, is_train, sampling_rate=500):
    dataset_names = ["super_class", "sub_class", "form", "rhythm"]
    data = {}
    for dataset in dataset_names:
        name2index = {}
        for i, name in enumerate(zero_shot_class["ptbxl_" + dataset]):
            name2index[name] = i
        Y = pd.read_csv(os.path.join(path, f"ptbxl_database_{dataset}.csv"))

        test_fold = 10
        if is_train:
            Y = Y[Y.strat_fold != test_fold]
        else:
            Y = Y[Y.strat_fold == test_fold]

        if sampling_rate == 500:
            X_rel = Y.filename_hr.values
        else:
            X_rel = Y.filename_lr.values
        X = [os.path.join(path, x) for x in X_rel]
        y = Y.labels.values

        labels = [label.split(';') for label in y]

        targets = np.zeros((len(X), len(zero_shot_class["ptbxl_" + dataset])))
        for i in range(len(X)):
            for lbl in labels[i]:
                targets[i][name2index[lbl]] = 1

        data[dataset] = (X, targets)
    return data


def load_sph(path, is_train, train_stage="stage1"):
    # --------------------------------------------------
    # 1. Load CSV(s)
    # --------------------------------------------------
    if train_stage == "stage1":
        df = pd.read_csv(os.path.join(path, "sph_clip.csv"))
    elif train_stage == "stage2":
        df = pd.read_csv(os.path.join(path, "sph_caption.csv"))
    else:
        raise ValueError(f"Unknown train_stage: {train_stage}")
    # --------------------------------------------------
    # 2. Patient-level split (UNCHANGED)
    # --------------------------------------------------
    duplicated_patient = df.Patient_ID.duplicated(keep=False)
    N = int(len(df) * 0.2) - duplicated_patient.sum()

    df_test = pd.concat([
        df[duplicated_patient],
        df[~duplicated_patient].sample(N, random_state=73)
    ])
    df_train = df.iloc[df.index.difference(df_test.index)]

    Y = df_train if is_train else df_test
    split_name = "train" if is_train else "test"

    print(f"[SPH][{split_name}] raw samples: {len(Y)}")

    # --------------------------------------------------
    # 3. Shared text cleaning
    # --------------------------------------------------
    def clean_text(t):
        if not isinstance(t, str):
            return ""
        return t.strip().lower()

    # --------------------------------------------------
    # 4. Build paired samples with debug
    # --------------------------------------------------
    X, texts = [], []
    dropped_non_str = 0
    dropped_empty = 0

    for _, row in Y.iterrows():
        # waveform path
        ecg_path = os.path.join(
            path, "records/records_wfdb", row.ECG_ID
        )

        # select text source
        if train_stage == "stage1":
            raw_txt = row.get("clip_text", None)
        elif train_stage == "stage2":
            raw_txt = row.get("caption_text", None)
        else:
            raise ValueError(f"Unknown train_stage: {train_stage}")

        if not isinstance(raw_txt, str):
            dropped_non_str += 1
            continue

        clean_txt = clean_text(raw_txt)
        if clean_txt == "":
            dropped_empty += 1
            continue

        X.append(ecg_path)
        texts.append(clean_txt)

    print(
        f"[SPH][{train_stage}][{split_name}] "
        f"kept={len(X)} "
        f"dropped_non_str={dropped_non_str} "
        f"dropped_empty={dropped_empty}"
    )

    if len(X) == 0:
        print(f"❌ [SPH][{train_stage}][{split_name}] NO VALID SAMPLES!")

    return X, texts



def load_champan_shaoxing(path, is_train, train_stage="stage1"):

    # --------------------------------------------------
    # 1. Load metadata according to training stage
    # --------------------------------------------------
    if train_stage == "stage1":
        Y = pd.read_csv(os.path.join(path, "champan_clip.csv"))
    elif train_stage == "stage2":
        Y = pd.read_csv(os.path.join(path, "champan_caption.csv"))
    else:
        raise ValueError(f"Unknown train_stage: {train_stage}")

    # --------------------------------------------------
    # 2. Stratified fold split (UNCHANGED)
    # --------------------------------------------------
    test_fold = 9
    if is_train:
        Y = Y[Y.strat_fold != test_fold]
        split = "train"
    else:
        Y = Y[Y.strat_fold == test_fold]
        split = "test"

    print(f"[CHAMPAN][{split}] raw samples: {len(Y)}")

    # --------------------------------------------------
    # 3. Shared text cleaning (same for all stages)
    # --------------------------------------------------
    def clean_text(t):
        if not isinstance(t, str):
            return ""
        t = t.strip().lower()
        return t

    # --------------------------------------------------
    # 4. Build paired ECG–text samples with debug
    # --------------------------------------------------
    X, y = [], []

    dropped_non_str = 0
    dropped_empty = 0

    for _, row in Y.iterrows():
        # ECG path (UNCHANGED)
        ecg_path = os.path.join(path, row.filename)

        # choose text source
        if train_stage == "stage1":
            raw_txt = row.get("clip_text", None)
        elif train_stage == "stage2":
            raw_txt = row.get("caption_text", None)
        else:
            raise ValueError(f"Unknown train_stage: {train_stage}")

        if not isinstance(raw_txt, str):
            dropped_non_str += 1
            continue

        txt = clean_text(raw_txt)
        if txt == "":
            dropped_empty += 1
            continue

        X.append(ecg_path)
        y.append(txt)

    # --------------------------------------------------
    # 5. Debug summary
    # --------------------------------------------------
    print(
        f"[CHAMPAN][{train_stage}][{split}] "
        f"kept={len(X)} | "
        f"dropped_non_str={dropped_non_str} | "
        f"dropped_empty={dropped_empty}"
    )

    if len(X) == 0:
        print(f"❌ [CHAMPAN][{train_stage}][{split}] NO VALID ECG–TEXT PAIRS!")

    # --------------------------------------------------
    # 6. Output unchanged
    # --------------------------------------------------
    return X, y

def load_cpsc2018(path, is_train):
    folder = "training_set" if is_train else "validation_set"
    Y = pd.read_csv(os.path.join(path, folder, "REFERENCE.csv"))

    # ECG paths
    X = Y.Recording.values
    X = [os.path.join(path, folder, x) for x in X]

    # ========= 关键修复开始 =========
    # First_label 一定存在
    lbl1 = Y.First_label.values.astype(int)

    # Second / Third 允许为空 → NaN 当作 0（表示不存在）
    lbl2 = pd.to_numeric(Y.Second_label, errors="coerce").fillna(0).astype(int).values
    lbl3 = pd.to_numeric(Y.Third_label,  errors="coerce").fillna(0).astype(int).values
    # ========= 关键修复结束 =========

    labels = np.zeros((len(X), 9), dtype=np.float32)

    for i in range(len(X)):
        # First label 一定是 1~9
        if 1 <= lbl1[i] <= 9:
            labels[i, lbl1[i] - 1] = 1

        # Second / Third label 只有在合法范围内才使用
        if 1 <= lbl2[i] <= 9:
            labels[i, lbl2[i] - 1] = 1
        if 1 <= lbl3[i] <= 9:
            labels[i, lbl3[i] - 1] = 1

    return X, labels

def make_dataloader(args, dataset, is_train, dist_sampler=True, drop_last=None):
    # 👉 统一 DataLoader 构建入口
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and dist_sampler else None
    shuffle = is_train and sampler is None
    drop_last = is_train if drop_last is None else drop_last
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=drop_last,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_ptbxl_diagnostic_dataset(args, path, preprocess_fn, is_train, epoch=0, tokenizer=None):
    sampling_rate = args.sampling_rate if args.sampling_rate is not None else 500
    datas = load_ptbxl_diagnostics(path, is_train, sampling_rate)

    dataloaders = {}
    for key in datas.keys():
        X, y = datas[key]
        dataset = ECGValDataset(path, X, y, transforms=preprocess_fn, tokenizer=tokenizer)
        dataloaders[key] = make_dataloader(args, dataset, is_train, dist_sampler=False, drop_last=False)
    return dataloaders


def get_cpsc2018_diagnostic_dataset(args, path, preprocess_fn, is_train, epoch=0, tokenizer=None):
    X, y = load_cpsc2018(path, is_train)
    dataset = ECGValDataset(path, X, y, transforms=preprocess_fn, tokenizer=tokenizer)

    return make_dataloader(args, dataset, is_train, dist_sampler=False, drop_last=False)

def _to_list(x):
    return list(x) if x is not None else []

def _build_ccar_cache_path(args, num_samples):
    """
    Must be exactly consistent with Concept-Sample/build_stage2_ccar_weights.py.
    Offline cache filename:
      stage2_train_ccar_weights_{num_samples}_{digest}.npy
    """
    cache_dir = os.path.dirname(os.path.abspath(args.concept_frequency_path))

    key = "|".join(
        [
            "stage2",
            str(num_samples),
            str(os.path.abspath(args.concept_frequency_path)),
            str(os.path.abspath(args.concept_vocab_path)),
            str(args.ccar_alpha),
            str(args.ccar_eps),
            str(args.ccar_wmin),
            str(args.ccar_wmax),
            str(args.ccar_default_weight),
        ]
    )

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    filename = f"stage2_train_ccar_weights_{num_samples}_{digest}.npy"
    return os.path.join(cache_dir, filename)


def build_or_load_ccar_weights(args, texts):
    """
    Training-time behavior:
      - never compute CCAR weights
      - never call dist.barrier()
      - never wait for rank0
      - only load the offline prebuilt .npy cache
    """
    cache_path = _build_ccar_cache_path(args, len(texts))

    rank = int(getattr(args, "rank", 0))
    world_size = int(getattr(args, "world_size", 1))

    if rank == 0:
        print(f"[CCAR][CACHE] expected file: {cache_path}", flush=True)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            "\n[CCAR][CACHE] Missing offline CCAR weight cache.\n"
            f"Expected file:\n  {cache_path}\n\n"
            "Please build it before torchrun, for example:\n"
            "  python Concept-Sample/build_stage2_ccar_weights.py ...\n"
        )

    weights = np.load(cache_path).astype(np.float32)

    if len(weights) != len(texts):
        raise RuntimeError(
            "\n[CCAR][CACHE] Length mismatch.\n"
            f"  cache file: {cache_path}\n"
            f"  weights length: {len(weights)}\n"
            f"  current stage2 train texts: {len(texts)}\n\n"
            "This usually means the offline cache was generated with different data paths, "
            "different filtering logic, or different CCAR parameters.\n"
            "Regenerate the cache with the same Stage2 data paths and same CCAR args.\n"
        )

    print(
        f"[CCAR][CACHE][rank={rank}/{world_size}] loaded {len(weights)} weights | "
        f"mean={weights.mean():.3f} "
        f"std={weights.std():.3f} "
        f"min={weights.min():.3f} "
        f"max={weights.max():.3f}",
        flush=True,
    )

    return weights

def get_all_ecg_text_dataset(args, preprocess_train, preprocess_test, epoch=0, tokenizer=None):
    datasets = {}
    stage = args.train_stage

    print("\n" + "=" * 80)
    print(f"[DATA] Building ECG-Text datasets | stage = {stage}")
    print("=" * 80)

    if args.mimic_iv_ecg_path:
        m = load_mimic_iv_ecg(args.mimic_iv_ecg_path, train_stage=stage)
        m_X_train, m_text_train, m_X_val, m_text_val, m_X_test, m_text_test = m
        print(f"[DATA][MIMIC] train={len(m_X_train)} val={len(m_X_val)} test={len(m_X_test)}")
    else:
        print("[DATA][MIMIC] skipped")
        m_X_train = m_text_train = []
        m_X_val = m_text_val = []
        m_X_test = m_text_test = []

    if args.ptbxl_path:
        p = load_ptbxl(args.ptbxl_path, sampling_rate=args.sampling_rate, train_stage=stage)
        p_X_train, p_text_train, p_X_val, p_text_val, p_X_test, p_text_test = p
        print(f"[DATA][PTBXL] train={len(p_X_train)} val={len(p_X_val)} test={len(p_X_test)}")
    else:
        p_X_train = p_text_train = []
        p_X_val = p_text_val = []
        p_X_test = p_text_test = []

    X_train = _to_list(m_X_train) + _to_list(p_X_train)
    text_train = _to_list(m_text_train) + _to_list(p_text_train)

    if args.champan_path:
        c_X, c_text = load_champan_shaoxing(args.champan_path, is_train=True, train_stage=stage)
        print(f"[DATA][CHAMPAN][train] {len(c_X)}")
        X_train += _to_list(c_X)
        text_train += _to_list(c_text)

    if args.sph_path:
        s_X, s_text = load_sph(args.sph_path, is_train=True, train_stage=stage)
        print(f"[DATA][SPH][train] {len(s_X)}")
        X_train += _to_list(s_X)
        text_train += _to_list(s_text)

    assert len(X_train) == len(text_train)
    print(f"[DATA][TRAIN][TOTAL] {len(X_train)} samples")

    concept_weighter = None
    sample_weights = None
    use_ccar_train = bool(stage == "stage2" and getattr(args, "use_ccar", False))

    if use_ccar_train:
        if is_master(args):
            print(
                f"[CCAR] enabled | loading prebuilt sample weights | "
                f"freq={args.concept_frequency_path} | "
                f"vocab={args.concept_vocab_path} | "
                f"alpha={args.ccar_alpha} eps={args.ccar_eps} "
                f"wmin={args.ccar_wmin} wmax={args.ccar_wmax} "
                f"default={args.ccar_default_weight}",
                flush=True,
            )

        sample_weights = build_or_load_ccar_weights(
            args=args,
            texts=text_train,
        )

    else:
        if is_master(args):
            print(
                f"[CCAR] disabled | stage={stage} "
                f"use_ccar={getattr(args, 'use_ccar', False)}",
                flush=True,
            )

    train_dataset = ECGTextDataset(
        X_train,
        text_train,
        transforms=preprocess_train,
        tokenizer=tokenizer,
        is_train=True,
        use_ccar=use_ccar_train,
        concept_weighter=None,
        return_weight=use_ccar_train,
        sample_weights=sample_weights,
    )

    datasets["train"] = make_dataloader(args, train_dataset, is_train=True)

    if not is_master(args):
        return datasets

    datasets["val_mimic"] = make_dataloader(
        args,
        ECGTextDataset(
            m_X_val,
            m_text_val,
            transforms=preprocess_test,
            tokenizer=tokenizer,
            is_train=False,
            use_ccar=False,
            concept_weighter=None,
            return_weight=False,
        ),
        is_train=False,
        dist_sampler=False,
    )

    if args.ptbxl_path:
        datasets["val_ptbxl"] = make_dataloader(
            args,
            ECGTextDataset(
                p_X_val,
                p_text_val,
                transforms=preprocess_test,
                tokenizer=tokenizer,
                is_train=False,
                use_ccar=False,
                concept_weighter=None,
                return_weight=False,
            ),
            is_train=False,
            dist_sampler=False,
        )

    if stage == "stage1":
        datasets["test_mimic"] = make_dataloader(
            args,
            ECGTextDataset(
                m_X_test,
                m_text_test,
                transforms=preprocess_test,
                tokenizer=tokenizer,
                is_train=False,
                use_ccar=False,
                concept_weighter=None,
                return_weight=False,
            ),
            is_train=False,
            dist_sampler=False,
        )

        if args.ptbxl_path:
            datasets["test_ptbxl"] = make_dataloader(
                args,
                ECGTextDataset(
                    p_X_test,
                    p_text_test,
                    transforms=preprocess_test,
                    tokenizer=tokenizer,
                    is_train=False,
                    use_ccar=False,
                    concept_weighter=None,
                    return_weight=False,
                ),
                is_train=False,
                dist_sampler=False,
            )

    return datasets

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    stage = args.train_stage

    data = get_all_ecg_text_dataset(
        args,
        preprocess_train,
        preprocess_val,
        epoch=epoch,
        tokenizer=tokenizer,
    )

    if not is_master(args):
        return data

    print("\n" + "-" * 80)
    print(f"[DATA] Building datasets | stage = {stage}")
    print(f"[DATA] Base keys: {list(data.keys())}")
    print("-" * 80)

    if stage == "stage2":
        print("[DATA][Stage2] Diagnostics & retrieval evaluation disabled during training")
        if "val" not in data:
            if "val_mimic" in data:
                data["val"] = data["val_mimic"]
            else:
                data["val"] = data["train"]
        print(f"[DATA][Stage2] Final keys: {list(data.keys())}")
        return data

    if stage == "stage1" and args.ptbxl_path:
        ptbxl_train = get_ptbxl_diagnostic_dataset(
            args,
            args.ptbxl_path,
            preprocess_val,
            is_train=True,
            epoch=epoch,
            tokenizer=tokenizer,
        )
        ptbxl_test = get_ptbxl_diagnostic_dataset(
            args,
            args.ptbxl_path,
            preprocess_val,
            is_train=False,
            epoch=epoch,
            tokenizer=tokenizer,
        )

        for key in ptbxl_train.keys():
            data[f"train_ptbxl_{key}"] = ptbxl_train[key]
            data[f"val_ptbxl_{key}"] = ptbxl_test[key]

        print("[DATA][PTBXL] diagnostic datasets enabled")

    if stage == "stage1" and args.cpsc2018_path:
        data["train_cpsc2018"] = get_cpsc2018_diagnostic_dataset(
            args,
            args.cpsc2018_path,
            preprocess_val,
            is_train=True,
            epoch=epoch,
            tokenizer=tokenizer,
        )
        data["val_cpsc2018"] = get_cpsc2018_diagnostic_dataset(
            args,
            args.cpsc2018_path,
            preprocess_val,
            is_train=False,
            epoch=epoch,
            tokenizer=tokenizer,
        )

        print("[DATA][CPSC2018] evaluation datasets enabled")

    print("-" * 80)
    print(f"[DATA] Final dataset keys: {list(data.keys())}")
    print("-" * 80)

    return data