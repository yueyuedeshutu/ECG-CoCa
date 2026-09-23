import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from training.data import (
    load_mimic_iv_ecg,
    load_ptbxl,
    load_champan_shaoxing,
    load_sph,
)


def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_regex_for_aliases(aliases):
    aliases = [normalize_text(a) for a in aliases]
    aliases = [a for a in aliases if a]
    aliases = sorted(set(aliases), key=len, reverse=True)

    if len(aliases) == 0:
        return None

    body = "|".join(re.escape(a) for a in aliases)
    return rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])"


def load_concept_probability(freq_path):
    with open(freq_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "concept_probability" in obj:
        prob = obj["concept_probability"]
    elif "concept_frequency" in obj and "total_samples" in obj:
        total = max(float(obj["total_samples"]), 1.0)
        prob = {
            k: float(v) / total
            for k, v in obj["concept_frequency"].items()
        }
    else:
        prob = obj

    return {str(k): float(v) for k, v in prob.items()}


def load_concept_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    clean_vocab = {}

    for concept, aliases in vocab.items():
        concept = str(concept)

        if isinstance(aliases, str):
            aliases = [aliases]
        elif aliases is None:
            aliases = []
        else:
            aliases = list(aliases)

        aliases = [str(a) for a in aliases]

        if concept not in aliases:
            aliases = [concept] + aliases

        clean_vocab[concept] = aliases

    return clean_vocab


def concept_weight_from_prob(prob, alpha, eps, w_min, w_max):
    raw = (1.0 / (float(prob) + float(eps))) ** float(alpha)
    return float(min(max(raw, float(w_min)), float(w_max)))


def build_cache_path(args, num_samples):
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


def collect_stage2_train_texts(args):
    X_train = []
    text_train = []

    if args.mimic_iv_ecg_path:
        m = load_mimic_iv_ecg(args.mimic_iv_ecg_path, train_stage="stage2")
        m_X_train, m_text_train, _, _, _, _ = m
        X_train += list(m_X_train)
        text_train += list(m_text_train)

    if args.ptbxl_path:
        p = load_ptbxl(args.ptbxl_path, sampling_rate=args.sampling_rate, train_stage="stage2")
        p_X_train, p_text_train, _, _, _, _ = p
        X_train += list(p_X_train)
        text_train += list(p_text_train)

    if args.champan_path:
        c_X, c_text = load_champan_shaoxing(args.champan_path, is_train=True, train_stage="stage2")
        X_train += list(c_X)
        text_train += list(c_text)

    if args.sph_path:
        s_X, s_text = load_sph(args.sph_path, is_train=True, train_stage="stage2")
        X_train += list(s_X)
        text_train += list(s_text)

    assert len(X_train) == len(text_train)

    print(f"[CCAR][OFFLINE] stage2 train samples = {len(text_train)}", flush=True)
    return text_train


def compute_weights(args, texts):
    concept_prob = load_concept_probability(args.concept_frequency_path)
    concept_vocab = load_concept_vocab(args.concept_vocab_path)

    concept_regex = {}
    concept_weight = {}

    for concept, aliases in concept_vocab.items():
        if concept not in concept_prob:
            continue

        regex = build_regex_for_aliases(aliases)
        if regex is None:
            continue

        concept_regex[concept] = regex
        concept_weight[concept] = concept_weight_from_prob(
            prob=concept_prob[concept],
            alpha=args.ccar_alpha,
            eps=args.ccar_eps,
            w_min=args.ccar_wmin,
            w_max=args.ccar_wmax,
        )

    print(
        f"[CCAR][OFFLINE] usable concepts = {len(concept_regex)}",
        flush=True,
    )

    if len(concept_regex) == 0:
        raise RuntimeError(
            "No usable concepts. Check concept_vocab.json and concept_frequency.json keys."
        )

    weights = np.full(
        len(texts),
        float(args.ccar_default_weight),
        dtype=np.float32,
    )

    n = len(texts)
    chunksize = args.chunksize

    for start in range(0, n, chunksize):
        end = min(start + chunksize, n)

        chunk = pd.Series(texts[start:end], dtype="string")
        chunk = (
            chunk.fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        chunk_weights = np.full(
            end - start,
            float(args.ccar_default_weight),
            dtype=np.float32,
        )

        for concept, regex in concept_regex.items():
            mask = chunk.str.contains(regex, regex=True, na=False).to_numpy()

            if mask.any():
                w = concept_weight[concept]
                chunk_weights[mask] = np.maximum(chunk_weights[mask], w)

        weights[start:end] = chunk_weights

        print(
            f"[CCAR][OFFLINE] processed {end}/{n} | "
            f"mean={weights[:end].mean():.3f} "
            f"min={weights[:end].min():.3f} "
            f"max={weights[:end].max():.3f}",
            flush=True,
        )

    return weights


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mimic-iv-ecg-path", type=str, default=None)
    parser.add_argument("--ptbxl-path", type=str, default=None)
    parser.add_argument("--champan-path", type=str, default=None)
    parser.add_argument("--sph-path", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=500)

    parser.add_argument("--concept-frequency-path", type=str, required=True)
    parser.add_argument("--concept-vocab-path", type=str, required=True)

    parser.add_argument("--ccar-alpha", type=float, default=0.75)
    parser.add_argument("--ccar-eps", type=float, default=1e-3)
    parser.add_argument("--ccar-wmin", type=float, default=1.0)
    parser.add_argument("--ccar-wmax", type=float, default=5.0)
    parser.add_argument("--ccar-default-weight", type=float, default=1.0)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    texts = collect_stage2_train_texts(args)

    if args.max_samples is not None:
        texts = texts[:args.max_samples]
        print(f"[CCAR][OFFLINE] max_samples enabled: {len(texts)}", flush=True)

    cache_path = build_cache_path(args, len(texts))

    print(f"[CCAR][OFFLINE] output cache = {cache_path}", flush=True)

    if os.path.exists(cache_path):
        print("[CCAR][OFFLINE] cache already exists, loading for check.", flush=True)
        weights = np.load(cache_path).astype(np.float32)
    else:
        weights = compute_weights(args, texts)
        tmp_path = cache_path + ".tmp.npy"
        np.save(tmp_path, weights)
        os.replace(tmp_path, cache_path)

    print(
        f"[CCAR][OFFLINE] done | "
        f"samples={len(weights)} "
        f"mean={weights.mean():.3f} "
        f"std={weights.std():.3f} "
        f"min={weights.min():.3f} "
        f"max={weights.max():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()