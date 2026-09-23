import os
import json
import argparse
import logging
from typing import Dict

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wfdb

from open_clip.ecg_transform import ecg_transform_v2, PreprocessCfg
from open_clip.factory import create_model
from open_clip.tokenizer import HFTokenizer

# ======================================================
# Logging
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("caption_eval")


# ======================================================
# Dataset  （❗完全不改）
# ======================================================
class ECGCaptionDataset(Dataset):
    def __init__(self, csv_path: str, preprocess):
        self.df = pd.read_csv(csv_path)
        self.preprocess = preprocess
        logger.info(f"Loaded CSV: {csv_path} | samples={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def _load_ecg(self, row):
        for col in ["wfdb_path", "path_hr", "path"]:
            if col in row and pd.notna(row[col]):
                sig, _ = wfdb.rdsamp(row[col])
                return torch.from_numpy(sig.T).float()
        raise ValueError("No valid ECG path column found")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            ecg = self._load_ecg(row)
            ecg = self.preprocess(ecg.unsqueeze(0)).squeeze(0)
        except Exception as e:
            logger.error(f"[ECG LOAD ERROR] idx={idx} | {e}")
            ecg = torch.zeros((12, 5000))

        return {
            "id": str(row.get("study_id", idx)),
            "ecg": ecg,
            "text": str(row.get("caption_text", "")),
        }


# ======================================================
# Caption Generation （❗完全不改）
# ======================================================
@torch.no_grad()
def generate_caption(model, tokenizer, ecg, device):
    model.eval()
    ecg = torch.nan_to_num(ecg.unsqueeze(0).to(device), nan=0.0)

    try:
        ids = model.generate(
            ecg,
            seq_len=120,
            generation_type="beam_search",
            num_beams=4,
            repetition_penalty=2.0,
            sot_token_id=tokenizer.tokenizer.cls_token_id,
            eos_token_id=tokenizer.tokenizer.sep_token_id,
            pad_token_id=tokenizer.tokenizer.pad_token_id,
        )
    except Exception:
        ids = model.generate(
            ecg,
            seq_len=120,
            generation_type="top_k",
            top_k=10,
            repetition_penalty=1.2,
            sot_token_id=tokenizer.tokenizer.cls_token_id,
            eos_token_id=tokenizer.tokenizer.sep_token_id,
            pad_token_id=tokenizer.tokenizer.pad_token_id,
        )

    return tokenizer.tokenizer.decode(
        ids[0].tolist(),
        skip_special_tokens=True
    ).strip()


# ======================================================
# Medical Metrics
# ======================================================

CLINICAL_CONCEPTS = [
    "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
    "atrial fibrillation", "atrial flutter",
    "left axis deviation", "right axis deviation",
    "left bundle branch block", "right bundle branch block",
    "av block", "st elevation", "st depression",
    "t wave inversion", "ischemia", "myocardial infarction",
]


# ------------------------------------------------------
# Concept extraction (string-based)
# ------------------------------------------------------
def extract_concepts(text: str):
    text = text.lower()
    return {c for c in CLINICAL_CONCEPTS if c in text}


# >>> NEW <<<
# Negation-aware concept extraction
def extract_concepts_with_negation(text: str):
    """
    Detect concepts with polarity:
    returns dict {concept: +1 (present) | -1 (negated)}
    """
    text = text.lower()
    neg_words = ["no ", "without ", "absent ", "denies "]
    results = {}

    for c in CLINICAL_CONCEPTS:
        if c in text:
            negated = any(n + c in text for n in neg_words)
            results[c] = -1 if negated else 1
    return results


# ------------------------------------------------------
# Clinical Entity Precision / Recall / F1
# ------------------------------------------------------
def concept_metrics(preds, gts):
    tp = fp = fn = 0
    for k in preds:
        p, g = extract_concepts(preds[k]), extract_concepts(gts[k])
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "Clinical_Precision": precision,
        "Clinical_Recall": recall,
        "Clinical_F1": f1,
    }


# >>> NEW <<<
# Negation-aware Clinical F1
def negation_aware_concept_metrics(preds, gts):
    tp = fp = fn = 0
    for k in preds:
        p = extract_concepts_with_negation(preds[k])
        g = extract_concepts_with_negation(gts[k])

        for c in p:
            if c in g and p[c] == g[c]:
                tp += 1
            else:
                fp += 1
        for c in g:
            if c not in p:
                fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "NegationAware_Clinical_Precision": precision,
        "NegationAware_Clinical_Recall": recall,
        "NegationAware_Clinical_F1": f1,
    }


# >>> NEW <<<
# Recall@K Concepts
def recall_at_k(preds, gts, k=3):
    hit, total = 0, 0
    for sid in preds:
        gt = list(extract_concepts(gts[sid]))
        pr = list(extract_concepts(preds[sid]))[:k]
        hit += len(set(pr) & set(gt))
        total += len(gt)
    return {f"Recall@{k}_Concepts": hit / (total + 1e-8)}


# >>> NEW <<<
# Length-normalized CIDEr
def length_normalized_cider(preds, gts):
    from pycocoevalcap.cider.cider import Cider
    cider = Cider()

    preds_norm = {}
    gts_dict = {}
    for k in preds:
        gt_len = len(gts[k].split())
        preds_norm[k] = [" ".join(preds[k].split()[:gt_len])]
        gts_dict[k] = [gts[k]]

    score, _ = cider.compute_score(gts_dict, preds_norm)
    return {"CIDEr_LengthNorm": score}


# >>> NEW <<<
# Per-concept Recall / Precision
def per_concept_stats(preds, gts):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLINICAL_CONCEPTS}
    for k in preds:
        p, g = extract_concepts(preds[k]), extract_concepts(gts[k])
        for c in CLINICAL_CONCEPTS:
            if c in p and c in g:
                stats[c]["tp"] += 1
            elif c in p and c not in g:
                stats[c]["fp"] += 1
            elif c not in p and c in g:
                stats[c]["fn"] += 1

    return {
        c: {
            "Precision": stats[c]["tp"] / (stats[c]["tp"] + stats[c]["fp"] + 1e-8),
            "Recall": stats[c]["tp"] / (stats[c]["tp"] + stats[c]["fn"] + 1e-8),
        }
        for c in CLINICAL_CONCEPTS
    }


# >>> NEW <<<
# Clinical Coverage Ratio
def clinical_coverage_ratio(preds):
    ratios = []
    for k in preds:
        n_concepts = len(extract_concepts(preds[k]))
        n_words = max(len(preds[k].split()), 1)
        ratios.append(n_concepts / n_words)
    return {"Clinical_Coverage_Ratio": sum(ratios) / len(ratios)}


# >>> NEW <<<
# Hallucination Rate
def hallucination_rate(preds, gts):
    hallucinations, total = 0, 0
    for k in preds:
        p, g = extract_concepts(preds[k]), extract_concepts(gts[k])
        hallucinations += len(p - g)
        total += max(len(p), 1)
    return {"Hallucination_Rate": hallucinations / total}


# ------------------------------------------------------
# BioMedBERT Similarity（保留）
# ------------------------------------------------------
def biomedbert_similarity(preds, gts):
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/gpfsdata/home/liuxinyue/liuxinyue_lustre/ECG-CLIP/model/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    sims = []
    for k in preds:
        p = tokenizer(preds[k], return_tensors="pt", truncation=True).to(device)
        g = tokenizer(gts[k], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            p_emb = model(**p).last_hidden_state.mean(1)
            g_emb = model(**g).last_hidden_state.mean(1)
        sims.append(F.cosine_similarity(p_emb, g_emb).item())

    return {"BioMedBERT_Similarity": sum(sims) / len(sims)}


# ======================================================
# Unified metric computation
# ======================================================
def compute_metrics(preds, gts):
    results = {}
    results.update(concept_metrics(preds, gts))
    results.update(negation_aware_concept_metrics(preds, gts))
    results.update(recall_at_k(preds, gts, k=3))
    results.update(length_normalized_cider(preds, gts))
    results.update(clinical_coverage_ratio(preds))
    results.update(hallucination_rate(preds, gts))
    results.update(biomedbert_similarity(preds, gts))
    results["Per_Concept_Stats"] = per_concept_stats(preds, gts)
    return results


# ======================================================
# Main（❗完全不改）
# ======================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model("coca_ViT-B-32", pretrained=None, device=device)
    tokenizer = HFTokenizer(
        "/gpfsdata/home/liuxinyue/liuxinyue_lustre/ECG-CLIP/model/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        context_length=76,
    )

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt.get("state_dict", ckpt).items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    preprocess = ecg_transform_v2(PreprocessCfg(seq_length=5000), is_train=False)
    dataset = ECGCaptionDataset(args.csv_path, preprocess)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions, gts = {}, {}
    for batch in tqdm(loader, desc="Inference"):
        sid = batch["id"][0]
        predictions[sid] = generate_caption(model, tokenizer, batch["ecg"][0], device)
        gts[sid] = batch["text"][0]

    # ==================================================
    # Debug: Print first 5 predictions
    # ==================================================
    print("\n" + "=" * 80)
    print("🔍 Sample Predictions (First 5)")
    print("=" * 80)

    for i, sid in enumerate(predictions.keys()):
        if i >= 5:
            break
        print(f"[{i+1}] ID: {sid}")
        print(f"GT   : {gts[sid]}")
        print(f"PRED : {predictions[sid]}")

        words = predictions[sid].split()
        if len(words) > 0:
            repeat_rate = (1 - len(set(words)) / len(words)) * 100
            print(f"Word Repeat Rate: {repeat_rate:.1f}%")

        print("-" * 80)

    print("=" * 80 + "\n")

    base = args.output_json.replace(".json", "")
    with open(base + ".predictions.json", "w") as f:
        json.dump(
            {sid: {"prediction": predictions[sid], "ground_truth": gts[sid]} for sid in predictions},
            f, indent=2
        )

    metrics = compute_metrics(predictions, gts)
    with open(base + ".metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_json", required=True)
    main(parser.parse_args())
