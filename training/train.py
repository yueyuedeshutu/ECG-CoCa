import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from .distributed import is_master
from .evaluation.zero_shot import zero_shot_eval
from .precision import get_autocast
from .evaluation.linear_probe import linear_probe_eval
from .logger import write_eval_log


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "ecg_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2],
    }


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def unpack_batch(batch):
    if len(batch) == 3:
        ecgs, texts, sample_weights = batch
    elif len(batch) == 2:
        ecgs, texts = batch
        sample_weights = None
    else:
        raise RuntimeError(f"Unexpected batch format with {len(batch)} elements.")
    return ecgs, texts, sample_weights


def compute_contrastive_loss(model_out, clip_loss_fn):
    ecg_features = model_out["ecg_features"]
    text_features = model_out["text_features"]
    logit_scale = model_out["logit_scale"]

    logits_per_ecg, logits_per_text = clip_loss_fn.get_logits(
        ecg_features,
        text_features,
        logit_scale,
    )

    labels = clip_loss_fn.get_ground_truth(
        ecg_features.device,
        logits_per_ecg.shape[0],
    )

    contrastive_loss = (
        F.cross_entropy(logits_per_ecg, labels)
        + F.cross_entropy(logits_per_text, labels)
    ) / 2

    weight = getattr(clip_loss_fn, "clip_loss_weight", 1.0)
    return contrastive_loss * weight


def compute_generative_loss(
    model_out,
    pad_id=0,
    sample_weights=None,
    caption_loss_weight=1.0,
):
    if "logits" not in model_out or "labels" not in model_out:
        return None

    token_logits = model_out["logits"]
    token_labels = model_out["labels"]

    token_loss = F.cross_entropy(
        token_logits.permute(0, 2, 1),
        token_labels,
        ignore_index=pad_id,
        reduction="none",
    )

    valid = token_labels.ne(pad_id).float()
    sample_loss = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    if sample_weights is not None:
        sample_weights = sample_weights.to(device=sample_loss.device, dtype=sample_loss.dtype)
        sample_loss = sample_loss * sample_weights

    return sample_loss.mean() * caption_loss_weight


def compute_stage_loss(
    model_out,
    clip_loss_fn,
    train_stage,
    sample_weights=None,
    pad_id=0,
):
    losses = {}

    if train_stage == "stage1":
        losses["contrastive_loss"] = compute_contrastive_loss(
            model_out,
            clip_loss_fn,
        )

    elif train_stage == "stage2":
        gen_loss = compute_generative_loss(
            model_out,
            pad_id=pad_id,
            sample_weights=sample_weights,
            caption_loss_weight=getattr(clip_loss_fn, "caption_loss_weight", 1.0),
        )

        if gen_loss is None:
            raise RuntimeError("Stage2 requires model_out to contain logits and labels.")

        losses["gen_loss"] = gen_loss

    else:
        raise ValueError(f"Unknown train_stage: {train_stage}")

    total_loss = sum(losses.values())
    losses["loss"] = total_loss
    return losses, total_loss


def model_forward_by_stage(model, ecgs, texts, train_stage):
    """
    Stage1:
        contrastive-only, no decoder forward.
    Stage2:
        caption generation, decoder forward enabled.
    """
    return model(
        ecgs,
        texts,
        output_labels=(train_stage == "stage2"),
    )


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    if args.train_stage == "stage2" and args.accum_freq > 1:
        logging.warning("[Stage2] Gradient accumulation disabled for generative-only training.")
        args.accum_freq = 1

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    model.train()

    if epoch == 0 and args.train_stage == "stage2" and is_master(args):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"[Stage2] Trainable params: {trainable_params:,}")

    data["train"].set_epoch(epoch)
    dataloader = data["train"].dataloader
    train_dataset = dataloader.dataset
    pad_id = getattr(unwrap_model(model), "pad_id", 0)

    if epoch == 0 and is_master(args):
        has_weights = getattr(train_dataset, "sample_weights", None) is not None
        n_weights = len(train_dataset.sample_weights) if has_weights else 0
        logging.info(
            f"[CCAR] dataset_has_weights={has_weights} len={n_weights} "
            f"stage={args.train_stage} use_ccar={getattr(args, 'use_ccar', False)}"
        )

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_ecgs, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler and args.train_stage != "stage2":
            scheduler(step)

        ecgs, texts, sample_weights = unpack_batch(batch)
        ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        if sample_weights is not None:
            sample_weights = sample_weights.to(device=device, dtype=torch.float32, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model_forward_by_stage(
                    model,
                    ecgs,
                    texts,
                    args.train_stage,
                )

                logit_scale = model_out["logit_scale"]

                use_weights = (
                    args.train_stage == "stage2"
                    and getattr(args, "use_ccar", False)
                    and sample_weights is not None
                )

                losses, total_loss = compute_stage_loss(
                    model_out=model_out,
                    clip_loss_fn=loss,
                    train_stage=args.train_stage,
                    sample_weights=sample_weights if use_weights else None,
                    pad_id=pad_id,
                )

                if i == 0 and epoch == 0 and is_master(args):
                    if use_weights:
                        logging.info(
                            f"[CCAR][Batch0] enabled | "
                            f"mean={sample_weights.mean().item():.3f} "
                            f"min={sample_weights.min().item():.3f} "
                            f"max={sample_weights.max().item():.3f}"
                        )
                    else:
                        logging.info("[CCAR][Batch0] disabled for this batch.")

            backward(total_loss, scaler)

        else:
            if args.train_stage != "stage1":
                raise RuntimeError("Gradient accumulation is only supported for Stage1 contrastive training.")

            with torch.no_grad():
                with autocast():
                    model_out = model(
                        ecgs,
                        texts,
                        output_labels=False,
                    )

                    model_out.pop("logit_scale", None)
                    model_out.pop("logit_bias", None)

                    for key in ("ecg_features", "text_features"):
                        val = model_out[key]
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_ecgs.append(ecgs)
                accum_texts.append(texts)

            if ((i + 1) % args.accum_freq) > 0:
                continue

            optimizer.zero_grad()

            for j in range(args.accum_freq):
                ecgs_j = accum_ecgs[j]
                texts_j = accum_texts[j]

                with autocast():
                    model_out = model(
                        ecgs_j,
                        texts_j,
                        output_labels=False,
                    )

                    logit_scale = model_out.pop("logit_scale")
                    inputs_no_accum = {"logit_scale": logit_scale}

                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}

                    for key, accumulated in accum_features.items():
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1:]
                        )

                    losses, total_loss = compute_stage_loss(
                        model_out={**inputs, **inputs_no_accum},
                        clip_loss_fn=loss,
                        train_stage=args.train_stage,
                        sample_weights=None,
                        pad_id=pad_id,
                    )

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)

                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip_norm,
                        norm_type=2.0,
                    )

                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip_norm,
                        norm_type=2.0,
                    )

                scaler.step(optimizer)

            scaler.update()

        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm,
                    norm_type=2.0,
                )

            optimizer.step()

        if args.accum_freq > 1:
            accum_ecgs, accum_texts, accum_features = [], [], {}

        if args.train_stage != "stage2":
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(ecgs)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()

                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )

            samples_per_second = (
                args.accum_freq
                * args.batch_size
                * args.world_size
                / batch_time_m.val
            )

            samples_per_second_per_gpu = (
                args.accum_freq
                * args.batch_size
                / batch_time_m.val
            )

            logging.info(
                f"Train Epoch: {epoch} "
                f"[{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, "
                f"{samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                + loss_log
            )

            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }

            log_data.update({name: val.val for name, val in losses_m.items()})
            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step
                wandb.log(log_data, step=step)

            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}

    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples
        cumulative_loss = 0.0
        all_ecg_features, all_text_features = [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                ecgs, texts, _ = unpack_batch(batch)
                ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(
                        ecgs,
                        texts,
                        output_labels=False,
                    )

                    ecg_features = model_out["ecg_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"].mean()

                    all_ecg_features.append(ecg_features.cpu())
                    all_text_features.append(text_features.cpu())

                    logits_per_ecg = logit_scale * ecg_features @ text_features.t()
                    logits_per_text = logits_per_ecg.t()

                    batch_size = ecgs.shape[0]
                    labels = torch.arange(batch_size, device=device).long()

                    total_loss = (
                        F.cross_entropy(logits_per_ecg, labels)
                        + F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

            val_metrics = get_clip_metrics(
                ecg_features=torch.cat(all_ecg_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )

            loss_val = cumulative_loss / num_samples

            metrics.update(
                {
                    **val_metrics,
                    "clip_val_loss": loss_val.item(),
                    "epoch": epoch,
                    "num_samples": num_samples,
                }
            )

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}
    write_eval_log(args, log_data, data, epoch, metrics, tb_writer)
    return metrics


def test(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}

    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    test_datasets = ["test_mimic", "test_ptbxl"]

    for test_set in test_datasets:
        if test_set not in data or data[test_set] is None:
            logging.warning(f"[Test] {test_set} not found, skipping.")
            continue

        dataloader = data[test_set].dataloader
        num_batches = len(dataloader)
        num_samples = getattr(dataloader, "num_samples", 0)

        logging.info(
            f"[Test] {test_set}: num_samples={num_samples}, num_batches={num_batches}"
        )

        if num_batches == 0 or num_samples == 0:
            logging.warning(f"[Test] {test_set} is empty, skipping evaluation.")
            metrics[f"{test_set}_num_samples"] = 0
            continue

        all_ecg_features, all_text_features = [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                ecgs, texts, _ = unpack_batch(batch)
                ecgs = ecgs.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(
                        ecgs,
                        texts,
                        output_labels=False,
                    )

                    ecg_features = model_out["ecg_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]

                all_ecg_features.append(ecg_features.cpu())
                all_text_features.append(text_features.cpu())

                if is_master(args) and (i % 100) == 0:
                    logging.info(f"Test {test_set}: [{i * ecgs.size(0)} / {num_samples}]")

        if len(all_ecg_features) == 0 or len(all_text_features) == 0:
            logging.warning(f"[Test] {test_set}: no features collected, skipping metrics.")
            metrics[f"{test_set}_num_samples"] = num_samples
            continue

        test_metrics = get_clip_metrics(
            ecg_features=torch.cat(all_ecg_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
            set_name=test_set + "_",
        )

        test_metrics[f"{test_set}_num_samples"] = num_samples
        metrics.update(test_metrics)

    logging.info(
        f"Test Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"test/" + name: val for name, val in metrics.items()}
    write_eval_log(args, log_data, data, epoch, metrics, tb_writer)
    return metrics


def get_clip_metrics(ecg_features, text_features, logit_scale, set_name=""):
    metrics = {}

    logits_per_ecg = (logit_scale * ecg_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_ecg.t().detach().cpu()

    logits = {
        f"{set_name}ecg_to_text": logits_per_ecg,
        f"{set_name}text_to_ecg": logits_per_text,
    }

    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()

        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1

        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out, pad_id=0):
    return compute_generative_loss(
        model_out,
        pad_id=pad_id,
        sample_weights=None,
        caption_loss_weight=1.0,
    )