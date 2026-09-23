import logging
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..precision import get_autocast, get_input_dtype
from .metrics import auroc, accuracy_and_f1
from .metadata import zero_shot_class


def unwrap_model(model):
    """Unwrap DistributedDataParallel if needed."""
    return model.module if hasattr(model, "module") else model


def build_zero_shot_classifier(args, model, tokenizer, dataset=""):
    texts = zero_shot_class[dataset]
    device = args.device

    with open('./training/evaluation/CKEPE_prompt.json', 'r', encoding='utf-8') as f:
        prompt = json.load(f)

    texts_encoded = []
    for text in texts:
        text = text.replace('_', "").replace("(s)", "")
        texts_encoded.append(tokenizer([prompt[text]])[0])

    texts_encoded = torch.stack(texts_encoded).to(device)

    # ✅ unwrap DDP
    model_ = unwrap_model(model)
    model_.eval()

    with torch.no_grad():
        class_embedding = model_.encode_text(texts_encoded)

    # shape: [embed_dim, num_classes]
    return class_embedding.T


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # ✅ unwrap once，彻底绕开 DDP forward
    model_ = unwrap_model(model)
    model_.eval()

    num_sample = dataloader.num_samples
    num_classes = classifier.shape[1]

    y_true = np.zeros((num_sample, num_classes))
    y_pred = np.zeros_like(y_true)

    idx = 0
    with torch.no_grad():
        for ecgs, targets in dataloader:
            batch_size = ecgs.size(0)

            ecgs = ecgs.to(device=args.device, dtype=input_dtype)

            with autocast():
                # ✅ 核心修改：只走 encoder，绝不调用 model(...)
                ecg_features = model_.encode_ecg(ecgs)

                logits = model_.logit_scale.exp() * ecg_features @ classifier

            probs = torch.sigmoid(logits).cpu().numpy()

            y_true[idx:idx + batch_size, :] = targets
            y_pred[idx:idx + batch_size, :] = probs
            idx += batch_size

    acc, f1, _, _ = accuracy_and_f1(y_true, y_pred)
    auc, _ = auroc(y_true, y_pred)
    return acc, f1, auc


def zero_shot_eval(model, data, args, tokenizer, dataset=""):
    logging.info(f'Starting zero-shot {dataset}.')
    assert tokenizer is not None

    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            args=args,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset
        )

    results = {}
    acc, f1, auc = run(model, classifier, data, args)
    results[f'{dataset}-zeroshot-val-acc'] = acc
    results[f'{dataset}-zeroshot-val-f1-score'] = f1
    results[f'{dataset}-zeroshot-val-auc'] = auc
    return results
