import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_model_config
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate, test
from training.file_utils import pt_load, start_sync_process, remote_sync
from training.evaluate import zero_shot_evaluatation, linear_probing_evaluation

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(" ")[-1]) for x in result.stdout.decode().split("\n")[:-1]]
    else:
        checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def set_dcm_dropout(model, p: float):
    if not hasattr(model, "dcm_adapter"):
        return
    for m in model.dcm_adapter.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = float(p)


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    if args.distributed:
        logging.info(
            f"[DDP Init] rank={args.rank} local_rank={args.local_rank} "
            f"world_size={args.world_size} device={args.device}"
        )
    else:
        logging.info("[DDP Init] Single process training")

    if args.name is None:
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            date_str = broadcast_object(args, date_str)
        args.name = "-".join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None

    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print("Error. Experiment already exists. Use --name to specify a new experiment.")
            return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path

        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print("Error. Cannot use save-most-recent with remote_sync and resume latest.")
                return -1
            if args.remote_sync_protocol != "s3":
                print("Error. Sync protocol not supported when using resume latest.")
                return -1

        if is_master(args):
            if args.save_most_recent:
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    resume_from = None
            else:
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)

            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")

        if args.distributed:
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("remote sync successful.")
        else:
            logging.info("Error: remote sync failed. Exiting.")
            return -1

        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        remote_sync_process.start()

    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning."
        )

    if args.horovod:
        logging.info(
            f"Running in horovod mode. Device: {args.device}. "
            f"Process global={args.rank}, local={args.local_rank}, total={args.world_size}."
        )
    elif args.distributed:
        logging.info(
            f"Running in distributed mode. Device: {args.device}. "
            f"Process global={args.rank}, local={args.local_rank}, total={args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    random_seed(args.seed, 0)
    model_kwargs = {}

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_ecg_size=args.force_ecg_size,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        **model_kwargs,
    )

    if args.use_bnb_linear is not None:
        print(
            "=> using a layer from bitsandbytes.\n"
            "   this is an experimental feature which requires two extra pip installs\n"
            "   pip install bitsandbytes triton\n"
            "   please make sure to use triton 2.0.0"
        )
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f"=> replacing linear layers with {args.use_bnb_linear}")
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_ecg:
        model.lock_ecg_tower(
            unlocked_groups=args.lock_ecg_unlocked_groups,
            freeze_bn_stats=args.lock_ecg_freeze_bn_stats,
        )

    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm,
        )

    use_dcm = bool(args.train_stage == "stage2" and getattr(args, "use_dcm", False))

    if hasattr(model, "set_dcm_enabled"):
        model.set_dcm_enabled(use_dcm)

    if hasattr(model, "dcm_adapter"):
        set_dcm_dropout(model, getattr(args, "dcm_dropout", 0.0))

    if hasattr(model, "apply_train_stage"):
        logging.info(
            f"Applying train stage: {args.train_stage} | "
            f"use_dcm={use_dcm} | dcm_dropout={getattr(args, 'dcm_dropout', 0.0)}"
        )
        try:
            model.apply_train_stage(args.train_stage, use_dcm=use_dcm)
        except TypeError:
            model.apply_train_stage(args.train_stage)
    else:
        logging.warning(f"Model {type(model)} has no apply_train_stage(), skipping stage control.")

    if args.train_stage == "stage2" and is_master(args):
        logging.info("Stage2 trainable parameters:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                logging.info(f"  {name} - {p.shape}")

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    else:
        params_file = ""

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        ddp_args = {"find_unused_parameters": True}

        if args.ddp_static_graph:
            ddp_args["static_graph"] = True

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            **ddp_args,
        )

    optimizer = None
    scaler = None

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if not args.eval:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    start_epoch = 0

    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location="cpu")
        sd = checkpoint["state_dict"]

        if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        model.load_state_dict(sd, strict=False)

        ckpt_stage = checkpoint.get("train_stage", None)

        if ckpt_stage == args.train_stage:
            start_epoch = checkpoint.get("epoch", 0)
            logging.info(f"=> Same-stage resume: stage={ckpt_stage}, start_epoch={start_epoch}")
        else:
            start_epoch = 0
            logging.info(
                f"=> Cross-stage initialization: checkpoint stage={ckpt_stage}, "
                f"current stage={args.train_stage}. Reset start_epoch to 0."
            )

        if optimizer is not None and ckpt_stage == args.train_stage and "optimizer" in checkpoint:
            logging.info(f"=> Resuming optimizer & scaler (same stage: {ckpt_stage})")
            optimizer.load_state_dict(checkpoint["optimizer"])

            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        else:
            logging.info(
                f"=> Loaded model weights only. "
                f"(checkpoint stage={ckpt_stage}, current stage={args.train_stage})"
            )

        logging.info(f"=> loaded checkpoint '{args.resume}'")

    tokenizer = get_tokenizer(args.model)

    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )

    assert len(data), "At least one train or eval dataset must be specified."

    scheduler = None

    if "train" in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs

        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, (
                "Please specify the number of cooldown epochs for this lr schedule."
            )
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer,
                args.lr,
                args.warmup,
                total_steps,
                cooldown_steps,
                args.lr_cooldown_power,
                args.lr_cooldown_end,
            )
        else:
            logging.error(
                f"Unknown scheduler {args.lr_scheduler}. "
                "Available options are: cosine, const, const-cooldown."
            )
            exit(1)

    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None

    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples

        if hasattr(args, "val_data") and args.val_data is not None and "val" in data:
            args.val_sz = data["val"].dataloader.num_samples

        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume="auto" if args.resume == "latest" else None,
            config=vars(args),
        )

        if args.debug:
            wandb.watch(model, log="all")

        if params_file:
            wandb.save(params_file)

        logging.debug("Finished loading wandb.")

    original_model = model

    if args.torchcompile:
        logging.info("Compiling model...")
        model = torch.compile(original_model)

    if args.use_bnb_linear is not None:
        from open_clip.utils import convert_int8_model_to_inference_mode
        convert_int8_model_to_inference_mode(model)

    if args.eval:
        test(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        zero_shot_evaluatation(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        linear_probing_evaluation(model, data, start_epoch, args, tb_writer=writer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        train_one_epoch(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            args,
            tb_writer=writer,
        )

        completed_epoch = epoch + 1

        if is_master(args):
            logging.info(f"[Epoch {completed_epoch}] TRAIN finished, checkpoint next")

        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "train_stage": args.train_stage,
                "use_ccar": getattr(args, "use_ccar", False),
                "use_dcm": getattr(args, "use_dcm", False),
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and completed_epoch % args.save_frequency == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )

            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

            logging.info(f"[Checkpoint] Epoch {completed_epoch} saved successfully")

        if args.train_stage == "stage1":
            if args.distributed:
                if is_master(args):
                    logging.info(
                        f"[Eval] Skipping in-training Stage1 evaluation under DDP at epoch {completed_epoch}. "
                        f"Please run evaluation separately from saved checkpoint."
                    )
            else:
                if is_master(args):
                    logging.info(f"[Eval] Starting evaluation at epoch {completed_epoch}")
                    try:
                        evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

                        if completed_epoch % args.val_frequency == 0:
                            logging.info("[Eval] Running test()")
                            test(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

                            logging.info("[Eval] Running zero-shot evaluation")
                            zero_shot_evaluatation(
                                model,
                                data,
                                completed_epoch,
                                args,
                                tb_writer=writer,
                                tokenizer=tokenizer,
                            )

                            logging.info("[Eval] Running linear probe evaluation")
                            linear_probing_evaluation(model, data, completed_epoch, args, tb_writer=writer)

                    except Exception as e:
                        logging.exception(f"[Eval Error] Epoch {completed_epoch} evaluation failed: {e}")
                        
        elif args.train_stage == "stage2":
            if is_master(args):
                logging.info("[Stage2] Skipping retrieval/diagnostic evaluation by design.")

        else:
            raise ValueError(f"Unknown train_stage: {args.train_stage}")

    if args.wandb and is_master(args):
        wandb.finish()

    if remote_sync_process is not None:
        logging.info("Final remote sync.")
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("Final remote sync successful.")
        else:
            logging.info("Final remote sync failed.")


def get_ecg_encoder(model_name, checkpoint_path, device):
    model_kwargs = {}
    model_config = get_model_config(model_name)

    model, _, preprocess_val = create_model_and_transforms(
        model_name,
        "",
        precision="amp",
        device=device,
        jit=False,
        output_dict=True,
        **model_kwargs,
    )

    model.to_empty(device=device)
    model = model.ecg
    checkpoint = pt_load(checkpoint_path, map_location="cpu")

    sd = checkpoint["state_dict"]
    sd_new = {}

    for k, v in sd.items():
        if k.startswith("module.ecg"):
            sd_new[k[len("module.ecg."):]] = v
        elif k.startswith("ecg."):
            sd_new[k[len("ecg."):]] = v

    model.load_state_dict(sd_new)
    logging.info(f"=> loaded checkpoint '{checkpoint_path}'")
    model.lock()
    return model, preprocess_val, model_config


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")

    if os.path.exists(new_code_path):
        print(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
        return -1

    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)

    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)

    copytree(current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb"))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])