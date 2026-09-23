import re
import json
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# =============================
# 1. Concept Vocabulary
# =============================
CONCEPT_VOCAB = {
    # ========== NORMAL ==========
    "NORMAL": [
        "normal ecg",
        "normal electrocardiogram",
        "within normal limits"
    ],
    "SR": [
        "sinus rhythm",
        "normal sinus rhythm",
        "sinus node dysfunction",
        "sinus bradycardia",
        "sinus tachycardia"
    ],

    # ========== RHYTHM 节律 ==========
    "AFIB": [
        "atrial fibrillation",
        "paroxysmal atrial fibrillation",
        "persistent atrial fibrillation",
        "long-standing persistent atrial fibrillation",
        "permanent atrial fibrillation"
    ],
    "AFLT": [
        "atrial flutter",
        "paroxysmal atrial flutter",
        "persistent atrial flutter",
        "long-standing persistent atrial flutter",
        "permanent atrial flutter",
        "atrial flutter",
        "atrial flutter with variable block",
        "atrial flutter with 2:1 block",
        "atrial flutter with 3:1 block",
        "typical atrial flutter",
        "atypical atrial flutter"
    ],
    "SVT": [
        "supraventricular tachycardia",
        "supraventricular tachyarrhythmia",
        "atrial tachycardia",
        "junctional tachycardia",
        "atrioventricular nodal reentrant tachycardia (avnrt)",
        "atrioventricular reciprocating tachycardia (avrt)",
        "wolff-parkinson-white syndrome"
    ],
    "STACH": [
        "sinus tachycardia",
        "supraventricular tachycardia",
        "atrial tachycardia",
        "junctional tachycardia",
        "sinus tachycardia",
        "sinus tachycardia with atrial fibrillation",
        "sinus tachycardia with atrial flutter",
        "sinus tachycardia with premature ventricular contractions"
    ],
    "VT": [
        "ventricular tachycardia",
        "ventricular tachyarrhythmia",
        "ventricular extrasystole",
        "ventricular ectopic beat",
        "ventricular premature complex",
        "ventricular premature contraction",
        "ventricular premature beat",
        "ventricular premature rhythm"
    ],
    "SBRAD": [
        "sinus bradycardia",
        "sinus bradycardia with atrial fibrillation",
        "sinus bradycardia with atrial flutter",
        "sinus bradycardia with premature ventricular contractions",
        "sinus bradycardia",
        "sinus bradycardia with atrial fibrillation",
        "sinus bradycardia with atrial flutter",
        "sinus bradycardia with premature ventricular contractions"
    ],
    "SARRH": [
        "sinus arrhythmia",
        "sinus arrhythmias",
        "sinus rhythm irregularity",
        "wandering atrial pacemaker"
    ],

    # ========== CONDUCTION 传导 ==========
    "1AVB": [
        "first degree av block",
        "first-degree atrioventricular block",
        "prolonged pr interval"
    ],
    "2AVB": [
        "second degree av block",
        "mobitz",
        "mobitz type i second degree av block",
        "mobitz type ii second degree av block"
    ],
    "2AVB1": [
        "second degree av block (type 1)",
        "mobitz type i second-degree atrioventricular block",
        "wenckebach phenomenon"
    ],
    "2AVB2": [
        "second degree av block (type two)",
        "mobitz type ii second-degree atrioventricular block",
        "second-degree atrioventricular block type ii",
        "type ii second-degree heart block"
    ],
    "3AVB": [
        "third degree av block",
        "complete heart block",
        "high-grade av block",
        "third-degree atrioventricular block"
    ],
    "AVB": [
        "av block",
        "atrioventricular block",
        "a-v block",
        "second-degree av block",
        "third-degree av block"
    ],

    "LBBB": [
        "left bundle branch block",
        "left bundle branch block (lbbb)",
        "left anterior fascicular block",
        "left posterior fascicular block",
        "bifascicular block (lbbb + rbbb)",
        "trifascicular block (lbbb + rbbb + lphb)"
    ],
    "LBBBB": [
        "left back bundle branch block",
        "left posterior fascicular block",
        "left posterior hemiblock",
        "left bundle branch block",
        "complete left bundle branch block"
    ],
    "ILBBB": [
        "incomplete left bundle branch block",
        "left anterior fascicular block",
        "left posterior fascicular block",
        "bifascicular block"
    ],
    "CLBBB": [
        "left bundle branch block",
        "left bundle branch block (lbbb)",
        "left anterior fascicular block",
        "left posterior fascicular block",
        "bifascicular block"
    ],
    "CRBBB": [
        "right bundle branch block",
        "right bundle branch block (rbbb)",
        "complete right bundle branch block",
        "incomplete right bundle branch block"
    ],
    "IRBBB": [
        "incomplete right bundle branch block",
        "incomplete right bundle branch block type i",
        "incomplete right bundle branch block type ii"
    ],
    "RBBB": [
        "right bundle branch block"
    ],
    "LAFB": [
        "left anterior fascicular block",
        "left anterior hemiblock",
        "left bundle branch block (lafb)",
        "first-degree lafb",
        "type i lafb"
    ],
    "LPFB": [
        "left posterior fascicular block",
        "left posterior fascicular block (lpfb)",
        "left posterior hemiblock",
        "left bundle branch block (lbbb) with left anterior fascicular conduction delay"
    ],
    "LAFB/LPFB": [
        "left anterior fascicular block",
        "left posterior fascicular block",
        "left anterior hemiblock",
        "left posterior hemiblock"
    ],
    "IVCD": [
        "intraventricular conduction",
        "non-specific intraventricular conduction disturbance",
        "non-specific intraventricular conduction defect",
        "non-specific intraventricular block"
    ],
    "IDC": [
        "interior differences conduction",
        "left bundle branch block",
        "right bundle branch block",
        "left anterior fascicular block",
        "left posterior fascicular block"
    ],
    "IVB": [
        "intraventricular block",
        "complete intraventricular block",
        "high-grade intraventricular block",
        "low-grade intraventricular block",
        "left bundle branch block (lbbb)",
        "right bundle branch block (rbbb)",
        "left anterior fascicular block (lafb)",
        "left posterior fascicular block (lpfb)",
        "right anterior fascicular block (rafb)",
        "right posterior fascicular block (rpfb)"
    ],

    # ========== HYPERTROPHY 肥厚 ==========
    "LVH": [
        "left ventricular hypertrophy",
        "left ventricular hypertrophy with strain pattern",
        "asymmetric septal hypertrophy",
        "concentric left ventricular hypertrophy",
        "eccentric left ventricular hypertrophy"
    ],
    "VCLVH": [
        "voltage criteria (qrs) for left ventricular hypertrophy",
        "voltage criteria for left ventricular hypertrophy",
        "qrs complex changes for left ventricular hypertrophy"
    ],
    "RVH": [
        "right ventricle hypertrophy",
        "right ventricular hypertrophy",
        "cor pulmonale",
        "idiopathic pulmonary hypertension-related right ventricular hypertrophy",
        "chronic thromboembolic pulmonary hypertension-related right ventricular hypertrophy"
    ],
    "LAO/LAE": [
        "left atrial overload",
        "left atrial enlargement",
        "left atrial hypertrophy",
        "left atrial dilatation"
    ],
    "RAO/RAE": [
        "right atrial overload",
        "right atrial enlargement",
        "right atrial dilatation"
    ],
    "LAE": [
        "left atrial enlargement"
    ],
    "RAE": [
        "right atrial enlargement"
    ],
    "RAH": [
        "right atrial hypertrophy",
        "right atrial enlargement",
        "right atrial dilatation"
    ],
    "SEHYP": [
        "septal hypertrophy",
        "left ventricular septal hypertrophy",
        "right ventricular septal hypertrophy",
        "apical septal hypertrophy",
        "mid-septal hypertrophy"
    ],

    # ========== ISCHEMIA / MI 缺血 ==========
    "MI": [
        "myocardial infarction",
        "anterior myocardial infarction",
        "anterolateral myocardial infarction",
        "anteroseptal myocardial infarction",
        "inferior myocardial infarction",
        "inferolateral myocardial infarction",
        "inferoposterolateral myocardial infarction",
        "inferoposterior myocardial infarction",
        "lateral myocardial infarction",
        "posterior myocardial infarction",
        "right ventricular myocardial infarction",
        "subendocardial myocardial infarction"
    ],
    "AMI": [
        "anterior myocardial infarction",
        "anterolateral myocardial infarction",
        "anteroseptal myocardial infarction",
        "anterobasal myocardial infarction"
    ],
    "ASMI": [
        "anteroseptal myocardial infarction",
        "anteroseptal wall infarction",
        "anterolateral myocardial infarction",
        "anterior wall infarction",
        "subendocardial injury in anteroseptal leads"
    ],
    "MIFW": [
        "myocardial infarction in the front wall",
        "anteroseptal myocardial infarction",
        "anterior myocardial infarction",
        "anterolateral myocardial infarction"
    ],
    "INJAS": [
        "ischemic in anteroseptal leads",
        "anteroseptal ischemia",
        "anterior septal ischemia",
        "anterolateral ischemia"
    ],
    "INJAL": [
        "in anterolateral leads",
        "anterolateral injury",
        "anteroseptal injury",
        "anterior wall myocardial infarction with lateral involvement"
    ],
    "IMI": [
        "inferior myocardial infarction",
        "inferolateral myocardial infarction",
        "inferoposterolateral myocardial infarction",
        "inferoposterior myocardial infarction",
        "subendocardial injury in inferior leads",
        "posterior wall mi",
        "posterior-inferior mi",
        "posteroinferolateral mi"
    ],
    "MILW": [
        "myocardial infarction in the lower wall",
        "inferior myocardial infarction",
        "posterior myocardial infarction",
        "lateral myocardial infarction",
        "subendocardial injury in inferior and lateral leads"
    ],
    "INJIN": [
        "subendocardial injury in inferior leads",
        "subendocardial injury in inferolateral leads",
        "subendocardial injury in inferoposterolateral leads",
        "subendocardial injury in inferoposterior leads"
    ],
    "ILMI": [
        "inferolateral myocardial infarction",
        "inferoposterolateral myocardial infarction",
        "inferoposterior myocardial infarction",
        "subendocardial injury in inferolateral leads"
    ],
    "ALMI": [
        "anterolateral myocardial infarction",
        "anteroseptal myocardial infarction",
        "anterolateral subendocardial injury",
        "anterolateral subepicardial injury"
    ],
    "ISCLA": [
        "in lateral leads",
        "lateral leads",
        "lateral wall myocardial infarction",
        "inferolateral myocardial infarction"
    ],
    "INJLA": [
        "subendocardial injury in lateral leads",
        "subendocardial injury in anterolateral leads",
        "subendocardial injury in posterolateral leads"
    ],
    "MISW": [
        "myocardial infarction in the side wall",
        "lateral myocardial infarction",
        "posterolateral myocardial infarction",
        "anterolateral myocardial infarction"
    ],
    "IPMI": [
        "inferoposterior myocardial infarction",
        "posterior myocardial infarction",
        "inferoposterolateral myocardial infarction",
        "subendocardial injury in inferior and lateral leads"
    ],
    "PMI": [
        "posterior myocardial infarction",
        "posterior wall myocardial infarction",
        "posterolateral myocardial infarction",
        "posteroseptal myocardial infarction",
        "subendocardial injury in posterior leads"
    ],
    "MIBW": [
        "myocardial infarction in back wall",
        "posterior myocardial infarction",
        "posterior-inferior myocardial infarction",
        "posterolateral myocardial infarction",
        "subendocardial injury in posterior leads"
    ],
    "IPLMI": [
        "inferoposterolateral myocardial infarction",
        "inferoposterior myocardial infarction",
        "inferolateral myocardial infarction"
    ],
    "LMI": [
        "lateral myocardial infarction"
    ],

    # ========== ST-T ST-T波改变 ==========
    "NST": [
        "non-specific st changes",
        "non-specific st segment changes",
        "non-specific st elevation",
        "non-specific st depression"
    ],
    "STE": [
        "st elevation",
        "st elevation - probable extension",
        "st elevation myocardial infarction",
        "st segment elevation in anterior leads",
        "st segment elevation in lateral leads",
        "st segment elevation in inferior leads"
    ],
    "STD": [
        "st depression",
        "non-specific st depression",
        "non-specific st segment depression",
        "diffuse st depression",
        "widespread st depression"
    ],
    "STDD": [
        "st drop down",
        "st segment depression",
        "st segment elevation"
    ],
    "STTU": [
        "st tilt up",
        "st segment tilt up",
        "st segment elevation in leads ii",
        "st segment elevation in leads iii and avl"
    ],
    "NDT": [
        "non-diagnostic t abnormalities",
        "non-diagnostic t-waves",
        "non-diagnostic t-wave changes",
        "abnormal t-waves in leads i and avl",
        "abnormal t-waves in leads ii, iii and avf"
    ],
    "NT": [
        "non-specific t-wave changes",
        "non-specific t-waves",
        "nonspecific st-t changes",
        "abnormal t-waves",
        "abnormal st segment and t waves"
    ],
    "TAB": [
        "t-wave abnormality",
        "t-wave changes",
        "non-diagnostic t-waves",
        "non-diagnostic t abnormalities"
    ],
    "TWC": [
        "t wave change",
        "t-wave changes",
        "non-diagnostic t-waves",
        "abnormal t-waves",
        "st-t wave changes"
    ],
    "TWO": [
        "t wave opposite",
        "t wave inversion",
        "t wave flattening"
    ],
    "INVT": [
        "inverted t-waves",
        "inverted t-wave changes",
        "inverted t-wave pattern",
        "inverted t-waves in leads v1-v3",
        "inverted t-waves in leads ii, iii and avl"
    ],
    "LOWT": [
        "low amplitude t-waves",
        "low voltage t-waves",
        "decreased amplitude t-waves"
    ],
    "ISC": [
        "ischemic st-t",
        "ischemic st-t changes",
        "non-specific ischemic st-t changes",
        "anterior ischemic st-t changes",
        "inferior ischemic st-t changes",
        "lateral ischemic st-t changes",
        "posterior ischemic st-t changes"
    ],
    "ISCA": [
        "anterior ischemia",
        "ischemic in anterior leads",
        "ischemic in anteroseptal leads",
        "ischemic in anterolateral leads",
        "anteroseptal myocardial infarction",
        "anterior myocardial infarction"
    ],
    "ISCI": [
        "inferior ischemia",
        "ischemic in inferior leads",
        "ischemic in inferolateral leads",
        "inferoposterior myocardial infarction",
        "subendocardial injury in inferior leads"
    ],
    "ISCIN": [
        "ischemic changes in inferior leads",
        "infero-inferior ischemia",
        "infero-lateral ischemia",
        "subendocardial injury in inferior leads"
    ],
    "ISCAL": [
        "in anterolateral leads",
        "anterolateral st segment changes",
        "anterolateral t-wave changes",
        "anteroseptal myocardial infarction"
    ],
    "ISCAS": [
        "in anteroseptal leads",
        "anteroseptal st segment elevation",
        "anteroseptal t-wave inversion",
        "anterolateral st segment depression"
    ],
    "ISCIL": [
        "ischemic in inferolateral leads",
        "inferolateral ischemia",
        "inferolateral st-segment depression",
        "inferolateral t-wave inversion"
    ],
    "STTC": [
        "st-t change",
        "ischemic in anterior leads",
        "ischemic in posterior leads",
        "ischemic in inferior leads",
        "ischemic in lateral leads",
        "non-specifc st changes"
    ],

    # ========== VOLTAGE (电压) ==========
    "LOW_VOLT": [
        "low qrs voltage",
        "low voltage"
    ],
    "HIGH_VOLT": [
        "high qrs voltage"
    ],
    "LVOLT": [
        "low qrs voltages in the frontal and horizontal leads",
        "low qrs voltage",
        "low qrs amplitude",
        "reduced qrs amplitude in frontal and horizontal leads"
    ],
    "HVOLT": [
        "high qrs voltage",
        "high qrs amplitude",
        "increased qrs amplitude",
        "increased r wave amplitude"
    ],
    "LVQRSAL": [
        "lower voltage qrs in all leads",
        "low voltage qrs complex in all leads",
        "low amplitude qrs complex in all leads"
    ],
    "LVQRSCL": [
        "lower voltage qrs in chest lead",
        "low voltage qrs complex",
        "reduced amplitude of qrs complex in chest leads"
    ],
    "LVQRSLL": [
        "lower voltage qrs in limb lead",
        "low-voltage qrs complex in limb leads",
        "reduced amplitude of qrs complex in limb leads"
    ],

    # ========== AXIS (电轴) ==========
    "ALS": [
        "axis left shift",
        "left axis deviation",
        "leftward shift of the electrical axis"
    ],
    "ARS": [
        "axis right shift",
        "right axis deviation",
        "rightward axis deviation",
        "rightward displacement of the mean electrical axis"
    ],
    "CCR": [
        "counter-clockwise rotation",
        "counterclockwise axis deviation",
        "left axis deviation"
    ],
    "CR": [
        "clockwise rotation",
        "counterclockwise rotation"
    ],

    # ========== QRS MORPHOLOGY ==========
    "ABQRS": [
        "abnormal qrs",
        "abnormal qrs axis",
        "abnormal qrs complex",
        "abnormal qrs duration",
        "abnormal qrs morphology",
        "abnormal qrs transition zone"
    ],
    "AQW": [
        "abnormal q wave",
        "anterior q wave",
        "posterior q wave",
        "lateral q wave",
        "septal q wave"
    ],
    "FQRS": [
        "fqrs wave",
        "fragmented qrs complex",
        "fragmented qrs waveforms",
        "abnormal qrs morphology"
    ],

    # ========== CLINICAL & METABOLIC (临床、代谢与药物) ==========
    "EL": [
        "compatible with electrolyte abnormalities",
        "electrolyte imbalance",
        "electrolyte disturbances",
        "hypokalemia",
        "hyperkalemia",
        "hypocalcemia",
        "hypercalcemia"
    ],
    "UW": [
        "u wave",
        "u-waves",
        "abnormal u waves",
        "prominent u waves"
    ],
    "DIG": [
        "dig",
        "suggests digitalis-effect",
        "digitalis-induced changes",
        "digitalis-like effect",
        "digitalis-related changes"
    ],
    "LNGQT": [
        "long qt-interval",
        "prolonged qt interval",
        "congenital long qt syndrome",
        "acquired long qt syndrome",
        "romano-ward syndrome",
        "jervell and lange-nielsen syndrome"
    ],
    "QTIE": [
        "qt interval extension",
        "prolonged qt interval",
        "prolonged corrected qt interval",
        "torsade de pointes",
        "long qt syndrome"
    ],
    "ANEUR": [
        "st-t changes compatible with ventricular aneurysm",
        "ventricular aneurysm",
        "left ventricular aneurysm",
        "right ventricular aneurysm"
    ],
    "ERV": [
        "early repolarization of the ventricles",
        "early repolarization",
        "ventricular early repolarization",
        "ventricular premature beats with early repolarization"
    ],

    # ========== ECTOPY / PRE-EXCITATION (异位搏动与预激) ==========
    "VPC": [
        "ventricular premature complexes",
        "ventricular premature beats",
        "ventricular ectopic beats",
        "premature ventricular contractions"
    ],
    "PVC": [
        "premature ventricular",
        "ventricular ectopic",
        "ventricular premature complex (beat)",
        "ventricular premature contraction",
        "ventricular ectopic beat",
        "ventricular extrasystole"
    ],
    "VEB": [
        "ventricular escape beat",
        "ventricular premature beat",
        "ventricular ectopic beat"
    ],
    "VPB": [
        "ventricular premature beat",
        "ventricular premature complexes",
        "ventricular ectopic beats",
        "ventricular extrasystoles"
    ],
    "PAC": [
        "premature atrial",
        "atrial ectopic",
        "atrial premature complex",
        "atrial premature beat",
        "atrial ectopic beat",
        "supraventricular premature complex"
    ],
    "APB": [
        "atrial premature beats",
        "atrial premature complexes",
        "atrial ectopic beats",
        "supraventricular ectopic beats"
    ],
    "JEB": [
        "junctional escape beat",
        "junctional escape rhythm",
        "junctional premature beat",
        "junctional premature complexes"
    ],
    "JPT": [
        "junctional premature beat",
        "junctional premature complexes",
        "atrioventricular junctional rhythm",
        "supraventricular junctional rhythm"
    ],
    "BIGU": [
        "bigeminal pattern (unknown origin)",
        "bigeminal supraventricular rhythm",
        "bigeminal ventricular rhythm",
        "supraventricular bigeminus",
        "ventricular bigeminus"
    ],
    "TRIGU": [
        "trigeminal pattern (unknown origin)",
        "trigeminal pattern of unknown origin with supraventricular origin",
        "trigeminal pattern of unknown origin with ventricular origin"
    ],
    "ABI": [
        "atrial bigeminy",
        "atrial extrasystole",
        "supraventricular bigeminy"
    ],
    "VB": [
        "ventricular bigeminy",
        "ventricular extrasystoles",
        "premature ventricular contractions"
    ],
    "VET": [
        "ventricular escape trigeminy",
        "ventricular escape beats",
        "ventricular premature beats with a cycle length of 2:1"
    ],
    "WPW": [
        "wolf-parkinson-white syndrome",
        "wolff-parkinson-white pattern",
        "pre-excitation syndrome"
    ],
    "VPE": [
        "ventricular preexcitation",
        "wolff-parkinson-white syndrome",
        "accessory pathway conduction",
        "kent bundle"
    ],
    "PRC(S)": [
        "premature complex(es)",
        "premature ventricular contraction(s)",
        "premature atrial contraction(s)",
        "premature junctional complex(es)"
    ],
    "VFW": [
        "ventricular fusion wave",
        "ventricular fusion beats",
        "ventricular fusion complexes"
    ],

    # ========== PACEMAKER 起搏器 ==========
    "PACE": [
        "pacemaker",
        "normal functioning artificial pacemaker",
        "normal functioning permanent artificial cardiac pacemaker",
        "normal functioning temporary artificial cardiac pacemaker"
    ],
}

# =============================
# 2. Text Normalization
# =============================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_vocab(vocab):
    out = {}
    for concept, patterns in vocab.items():
        norm_patterns = []
        for p in patterns:
            p = normalize_text(p)
            if p and p not in norm_patterns:
                norm_patterns.append(p)

        key_alias = normalize_text(concept)
        if len(key_alias) >= 3 and key_alias not in norm_patterns:
            norm_patterns.append(key_alias)

        out[concept] = norm_patterns
    return out


def build_regex_for_patterns(patterns):
    patterns = [normalize_text(p) for p in patterns]
    patterns = [p for p in patterns if p]
    patterns = sorted(set(patterns), key=len, reverse=True)

    if not patterns:
        return None

    body = "|".join(re.escape(p) for p in patterns)
    return rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])"


def build_concept_regex(vocab):
    concept_regex = {}
    for concept, patterns in vocab.items():
        regex = build_regex_for_patterns(patterns)
        if regex is not None:
            concept_regex[concept] = regex
    return concept_regex


def count_csv_rows(csv_path):
    csv_path = Path(csv_path)
    with open(csv_path, "rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^a-z0-9\s\-]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.str.strip()


# =============================
# 3. Fast Frequency Estimation
# =============================
def process_csv_fast(
    csv_path,
    text_col="caption_text",
    concept_regex=None,
    chunksize=50000,
    show_concept_progress=False,
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    if text_col not in header.columns:
        raise ValueError(f"Column '{text_col}' not found in {csv_path}")

    total_rows = count_csv_rows(csv_path)
    freq = defaultdict(int)
    total = 0
    matched_total = 0

    concepts = list(concept_regex.keys())
    non_normal_concepts = [c for c in concepts if c != "NORMAL"]

    n_chunks = (total_rows + chunksize - 1) // chunksize if chunksize > 0 else None
    reader = pd.read_csv(csv_path, usecols=[text_col], chunksize=chunksize)

    iterator = reader
    if tqdm is not None:
        iterator = tqdm(
            reader,
            total=n_chunks,
            desc=f"{csv_path.name}",
            dynamic_ncols=True,
        )

    for chunk_idx, chunk in enumerate(iterator):
        texts = normalize_series(chunk[text_col])
        bsz = len(texts)
        total += bsz

        matched_any = pd.Series(False, index=texts.index)
        non_normal_any = pd.Series(False, index=texts.index)
        normal_mask = None

        concept_iter = concepts
        if show_concept_progress and tqdm is not None:
            concept_iter = tqdm(
                concepts,
                desc=f"concepts chunk {chunk_idx}",
                leave=False,
                dynamic_ncols=True,
            )

        for concept in concept_iter:
            regex = concept_regex[concept]
            mask = texts.str.contains(regex, regex=True, na=False)

            if concept == "NORMAL":
                normal_mask = mask
            else:
                cnt = int(mask.sum())
                if cnt > 0:
                    freq[concept] += cnt
                    matched_any |= mask
                    non_normal_any |= mask

        if normal_mask is not None:
            normal_only = normal_mask & (~non_normal_any)
            cnt_normal = int(normal_only.sum())
            if cnt_normal > 0:
                freq["NORMAL"] += cnt_normal
                matched_any |= normal_only

        matched_total += int(matched_any.sum())

        if tqdm is None:
            print(
                f"[{csv_path.name}] chunk={chunk_idx + 1} "
                f"processed={total}/{total_rows} "
                f"matched={matched_total} "
                f"ratio={matched_total / total if total else 0:.4f}",
                flush=True,
            )

    return freq, total, matched_total


# =============================
# 4. Optional Slow Debug Matcher
# =============================
def extract_concepts_slow(text: str, vocab):
    text = normalize_text(text)
    matched = set()

    for concept, patterns in vocab.items():
        for p in patterns:
            p = normalize_text(p)
            if not p:
                continue
            regex = rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])"
            if re.search(regex, text):
                matched.add(concept)
                break

    if matched and matched != {"NORMAL"}:
        matched.discard("NORMAL")

    return matched


def sanity_check_examples(vocab):
    examples = [
        "Normal ECG with sinus rhythm.",
        "Sinus rhythm with left ventricular hypertrophy and ST-T change.",
        "Atrial fibrillation with rapid ventricular response.",
        "Inferior myocardial infarction with ST elevation.",
        "This ECG demonstrates mild or borderline electrical abnormalities. Examination of the ECG discloses sinus rhythm and possible right atrial abnormality.",
    ]

    print("\n[Sanity check]", flush=True)
    for x in examples:
        concepts = sorted(extract_concepts_slow(x, vocab))
        print(f"{concepts} | {x}", flush=True)
    print("", flush=True)


# =============================
# 5. Main
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--text-col", type=str, default="caption_text")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--show-concept-progress", action="store_true", default=False)
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--sanity-check", action="store_true", default=True)

    parser.add_argument(
        "--mimic-csv",
        type=str,
        default="/gpfsdata/home/liuxinyue/liuxinyue_lustre/ECG-CLIP/ECG-CoCa/data/mimic-iv-ecg/mimic_iv_ecg_caption_new.csv"
    )
    parser.add_argument(
        "--ptbxl-csv",
        type=str,
        default="/gpfsdata/home/liuxinyue/liuxinyue_lustre/ECG-CLIP/ECG-CoCa/data/ptb-xl/ptbxl_database_caption_new.csv"
    )
    parser.add_argument(
        "--champan-csv",
        type=str,
        default="/gpfsdata/home/liuxinyue/liuxinyue_lustre/ECG-CLIP/ECG-CoCa/data/champan-shaoxing/champan_caption.csv"
    )
    parser.add_argument(
        "--sph-csv",
        type=str,
        default="/gpfsdata/home/liuxinyue/liuxinyue_lustre/ECG-CLIP/ECG-CoCa/data/sph/sph_caption.csv"
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = normalize_vocab(CONCEPT_VOCAB)
    concept_regex = build_concept_regex(vocab)

    print("=" * 80, flush=True)
    print("[CCAR] Build concept vocabulary and frequency", flush=True)
    print(f"[CCAR] num_concepts={len(vocab)}", flush=True)
    print(f"[CCAR] chunksize={args.chunksize}", flush=True)
    print(f"[CCAR] out_dir={out_dir}", flush=True)
    print("=" * 80, flush=True)

    if args.sanity_check:
        sanity_check_examples(vocab)

    datasets = {
        "mimic": args.mimic_csv,
        "ptbxl": args.ptbxl_csv,
        "champan": args.champan_csv,
        "sph": args.sph_csv,
    }

    global_freq = defaultdict(int)
    global_total = 0
    global_matched = 0
    dataset_stats = {}

    for name, path in datasets.items():
        path = Path(path)

        if not path.exists():
            msg = f"[{name}] CSV not found: {path}"
            if args.skip_missing:
                print(msg + " | skipped", flush=True)
                dataset_stats[name] = {
                    "path": str(path),
                    "skipped": True,
                    "total_samples": 0,
                    "matched_samples": 0,
                    "matched_ratio": 0.0,
                    "num_concepts": 0,
                }
                continue
            raise FileNotFoundError(msg)

        print("-" * 80, flush=True)
        print(f"[{name}] start: {path}", flush=True)

        freq, total, matched_total = process_csv_fast(
            path,
            text_col=args.text_col,
            concept_regex=concept_regex,
            chunksize=args.chunksize,
            show_concept_progress=args.show_concept_progress,
        )

        global_total += total
        global_matched += matched_total

        for k, v in freq.items():
            global_freq[k] += v

        dataset_stats[name] = {
            "path": str(path),
            "skipped": False,
            "total_samples": total,
            "matched_samples": matched_total,
            "matched_ratio": matched_total / total if total > 0 else 0.0,
            "num_concepts": len(freq),
            "top_concepts": dict(sorted(freq.items(), key=lambda x: -x[1])[:20]),
        }

        print(
            f"[{name}] done | total={total} matched={matched_total} "
            f"ratio={dataset_stats[name]['matched_ratio']:.4f} "
            f"concepts={len(freq)}",
            flush=True,
        )
        print(f"[{name}] top20={dataset_stats[name]['top_concepts']}", flush=True)

    concept_frequency = dict(sorted(global_freq.items(), key=lambda x: -x[1]))
    concept_probability = {
        k: v / global_total for k, v in concept_frequency.items()
    } if global_total > 0 else {}

    freq_out = {
        "total_samples": global_total,
        "matched_samples": global_matched,
        "matched_ratio": global_matched / global_total if global_total > 0 else 0.0,
        "dataset_stats": dataset_stats,
        "concept_frequency": concept_frequency,
        "concept_probability": concept_probability,
    }

    vocab_path = out_dir / "concept_vocab.json"
    freq_path = out_dir / "concept_frequency.json"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    with open(freq_path, "w", encoding="utf-8") as f:
        json.dump(freq_out, f, indent=2, ensure_ascii=False)

    print("=" * 80, flush=True)
    print(f"Saved {vocab_path}", flush=True)
    print(f"Saved {freq_path}", flush=True)
    print(
        f"[GLOBAL] total={global_total} matched={global_matched} "
        f"ratio={freq_out['matched_ratio']:.4f} concepts={len(concept_frequency)}",
        flush=True,
    )
    print(f"[GLOBAL] top30={dict(list(concept_frequency.items())[:30])}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()