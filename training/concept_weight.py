import json
import os
import re
from typing import Dict, List, Set, Optional


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_match(pattern: str, text: str) -> bool:
    pattern = normalize_text(pattern)
    if not pattern:
        return False
    regex = r"(?<![a-z0-9])" + re.escape(pattern) + r"(?![a-z0-9])"
    return re.search(regex, text) is not None


def _load_json(path: str):
    if path is None:
        raise ValueError("JSON path is None.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ConceptWeightComputer:
    """
    CCAR: Clinical-Concept Aware Reweighting.

    caption text
        -> matched clinical concepts
        -> concept rarity
        -> scalar sample weight

    Weight:
        w_i = clip(
            max_{c in C_i} (1 / (p_c + eps)) ** alpha,
            w_min,
            w_max
        )
    """

    def __init__(
        self,
        concept_freq_path: str,
        concept_vocab_path: Optional[str] = None,
        alpha: float = 0.75,
        eps: float = 1e-3,
        w_min: float = 1.0,
        w_max: float = 5.0,
        default_weight: float = 1.0,
    ):
        freq_data = _load_json(concept_freq_path)

        if "concept_probability" not in freq_data:
            raise KeyError(
                f"'concept_probability' not found in {concept_freq_path}"
            )

        self.concept_prob: Dict[str, float] = {
            str(k): float(v)
            for k, v in freq_data["concept_probability"].items()
        }

        if concept_vocab_path is not None:
            vocab_data = _load_json(concept_vocab_path)
        elif "concept_vocab" in freq_data:
            vocab_data = freq_data["concept_vocab"]
        else:
            raise ValueError(
                "No concept vocabulary provided. "
                "Please pass concept_vocab_path or include concept_vocab in concept_frequency.json."
            )

        self.concept_vocab: Dict[str, List[str]] = self._normalize_vocab(vocab_data)

        # Keep only concepts that have frequency statistics.
        self.concept_vocab = {
            c: ps
            for c, ps in self.concept_vocab.items()
            if c in self.concept_prob
        }

        self.alpha = float(alpha)
        self.eps = float(eps)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.default_weight = float(default_weight)

        if self.w_max < self.w_min:
            raise ValueError(
                f"ccar w_max must be >= w_min, got {self.w_max} < {self.w_min}"
            )

    @staticmethod
    def _normalize_vocab(vocab_data) -> Dict[str, List[str]]:
        vocab = {}

        for concept, patterns in vocab_data.items():
            if isinstance(patterns, str):
                patterns = [patterns]

            norm_patterns = []
            for p in patterns:
                p = normalize_text(p)
                if p and p not in norm_patterns:
                    norm_patterns.append(p)

            # Add the concept key itself as an alias only if it is not too short.
            # This keeps abbreviations like AFIB/LVH usable, while reducing
            # accidental matching for very short keys.
            key_alias = normalize_text(concept)
            if len(key_alias) >= 3 and key_alias not in norm_patterns:
                norm_patterns.append(key_alias)

            vocab[str(concept)] = norm_patterns

        return vocab

    def extract_concepts(self, caption: str) -> Set[str]:
        text = normalize_text(caption)
        matched = set()

        for concept, patterns in self.concept_vocab.items():
            for p in patterns:
                if phrase_match(p, text):
                    matched.add(concept)
                    break

        if matched and matched != {"NORMAL"}:
            matched.discard("NORMAL")

        return matched

    def rarity(self, concept: str) -> float:
        p = self.concept_prob.get(concept, 0.0)
        return (1.0 / (p + self.eps)) ** self.alpha

    def compute_weight(self, caption: str) -> float:
        concepts = self.extract_concepts(caption)

        if not concepts:
            return self.default_weight

        w_raw = max(self.rarity(c) for c in concepts)
        w = min(max(w_raw, self.w_min), self.w_max)
        return float(w)

    def compute_weights(self, captions: List[str]) -> List[float]:
        return [self.compute_weight(c) for c in captions]

    def describe_caption(self, caption: str) -> Dict:
        concepts = sorted(self.extract_concepts(caption))
        if not concepts:
            return {
                "caption": caption,
                "concepts": [],
                "weight": self.default_weight,
                "rarities": {},
            }

        rarities = {c: self.rarity(c) for c in concepts}
        return {
            "caption": caption,
            "concepts": concepts,
            "weight": self.compute_weight(caption),
            "rarities": rarities,
        }


_GLOBAL_CONCEPT_WEIGHTER = None


def build_concept_weighter_from_args(args) -> ConceptWeightComputer:
    return ConceptWeightComputer(
        concept_freq_path=args.concept_frequency_path,
        concept_vocab_path=args.concept_vocab_path,
        alpha=args.ccar_alpha,
        eps=args.ccar_eps,
        w_min=args.ccar_wmin,
        w_max=args.ccar_wmax,
        default_weight=args.ccar_default_weight,
    )


def set_global_concept_weighter(weighter: ConceptWeightComputer):
    global _GLOBAL_CONCEPT_WEIGHTER
    _GLOBAL_CONCEPT_WEIGHTER = weighter


def get_caption_weight(caption: str) -> float:
    """
    Backward-compatible global API.
    Main training code should prefer passing ConceptWeightComputer explicitly.
    """
    if _GLOBAL_CONCEPT_WEIGHTER is None:
        raise RuntimeError(
            "Global ConceptWeightComputer is not initialized. "
            "Use build_concept_weighter_from_args(args) and pass the weighter explicitly."
        )
    return _GLOBAL_CONCEPT_WEIGHTER.compute_weight(caption)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept-frequency-path",
        type=str,
        default="./Concept-Sample/concept_frequency.json"
    )
    parser.add_argument(
        "--concept-vocab-path",
        type=str,
        default="./Concept-Sample/concept_vocab.json"
    )
    parser.add_argument("--ccar-alpha", type=float, default=0.75)
    parser.add_argument("--ccar-eps", type=float, default=1e-3)
    parser.add_argument("--ccar-wmin", type=float, default=1.0)
    parser.add_argument("--ccar-wmax", type=float, default=5.0)
    parser.add_argument("--ccar-default-weight", type=float, default=1.0)
    args = parser.parse_args()

    weighter = build_concept_weighter_from_args(args)

    examples = [
        "Normal ECG with sinus rhythm.",
        "Sinus rhythm with left ventricular hypertrophy and ST-T change.",
        "Atrial fibrillation with rapid ventricular response.",
        "Inferior myocardial infarction with ST elevation.",
        "This ECG demonstrates mild or borderline electrical abnormalities. Examination of the ECG discloses sinus rhythm and possible right atrial abnormality.",
    ]

    for cap in examples:
        info = weighter.describe_caption(cap)
        print(json.dumps(info, indent=2, ensure_ascii=False))