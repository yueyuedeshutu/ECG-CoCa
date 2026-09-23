from typing import Optional

import numpy as np
import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import (
    CLIPTextCfg,
    _build_text_tower,
    CLIPEcgCfg,
    _build_ecg_tower,
)

try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList,
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search",
    }
    _has_transformers = True
except ImportError:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search",
    }
    _has_transformers = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8

    # Diagnostic Context Memory (DCM)
    use_dcm: bool = False
    dcm_dropout: float = 0.0

    # Backward-compatible aliases for old configs
    use_diag_soft_prompt: bool = False
    diag_prompt_dropout: float = 0.0


def _build_text_decoder_tower(
    embed_dim,
    multimodal_cfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    multimodal_cfg = (
        MultimodalCfg(**multimodal_cfg)
        if isinstance(multimodal_cfg, dict)
        else multimodal_cfg
    )

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32
        if cast_dtype in (torch.float16, torch.bfloat16)
        else LayerNorm
    )

    return MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )


class CoCa(nn.Module):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        text_cfg: CLIPTextCfg,
        ecg_cfg: CLIPEcgCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
    ):
        super().__init__()

        multimodal_cfg = (
            MultimodalCfg(**multimodal_cfg)
            if isinstance(multimodal_cfg, dict)
            else multimodal_cfg
        )
        text_cfg = (
            CLIPTextCfg(**text_cfg)
            if isinstance(text_cfg, dict)
            else text_cfg
        )
        ecg_cfg = (
            CLIPEcgCfg(**ecg_cfg)
            if isinstance(ecg_cfg, dict)
            else ecg_cfg
        )

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = text_cfg.vocab_size

        self.ecg = _build_ecg_tower(
            embed_dim=embed_dim,
            ecg_cfg=ecg_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * init_logit_scale
        )
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * init_logit_bias)
            if init_logit_bias is not None
            else None
        )

        self.pad_id = int(pad_id)
        self.context_length = multimodal_cfg.context_length

        # DCM switch. Old config fields remain accepted.
        self.use_dcm = bool(
            multimodal_cfg.use_dcm
            or multimodal_cfg.use_diag_soft_prompt
        )

        dcm_dropout = multimodal_cfg.dcm_dropout

        if (
            dcm_dropout == 0.0
            and multimodal_cfg.diag_prompt_dropout > 0.0
        ):
            dcm_dropout = multimodal_cfg.diag_prompt_dropout

        self.dcm_adapter = nn.Sequential(
            nn.Linear(embed_dim, multimodal_cfg.width),
            nn.GELU(),
            nn.Dropout(dcm_dropout),
        )

        # Backward-compatible aliases.
        self.use_soft_prompt = self.use_dcm
        self.diag_prompt_adapter = self.dcm_adapter

    def set_dcm_enabled(self, enabled: bool):
        self.use_dcm = bool(enabled)
        self.use_soft_prompt = self.use_dcm

    @torch.jit.ignore
    def set_grad_checkpointing(
        self,
        enable: bool = True,
    ):
        if hasattr(self.ecg, "set_grad_checkpointing"):
            self.ecg.set_grad_checkpointing(enable)

        if hasattr(self.text, "set_grad_checkpointing"):
            self.text.set_grad_checkpointing(enable)

        if hasattr(
            self.text_decoder,
            "set_grad_checkpointing",
        ):
            self.text_decoder.set_grad_checkpointing(enable)

    def _encode_ecg(
        self,
        ecgs,
        normalize: bool = True,
    ):
        ecg_latent, token_embs = self.ecg(ecgs)

        if normalize:
            ecg_latent = F.normalize(
                ecg_latent,
                dim=-1,
            )

        return ecg_latent, token_embs

    def _encode_text(
        self,
        text,
        normalize: bool = True,
    ):
        text_latent, token_embs = self.text(text)

        if normalize:
            text_latent = F.normalize(
                text_latent,
                dim=-1,
            )

        return text_latent, token_embs

    def _get_caption_token_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return non-contextual BERT embeddings for autoregressive captioning.

        The caption decoder must not consume BiomedBERT last_hidden_state,
        because bidirectional contextual states can see future tokens during
        teacher forcing. This method calls only the Hugging Face embedding
        module, which normally contains word, position and token-type
        embeddings plus LayerNorm/dropout.
        """
        if input_ids.ndim != 2:
            raise RuntimeError(
                "Caption input_ids must have shape [B,L], "
                f"got {tuple(input_ids.shape)}."
            )

        transformer = getattr(
            self.text,
            "transformer",
            None,
        )

        if transformer is None:
            raise RuntimeError(
                "HF text encoder has no 'transformer' attribute."
            )

        embeddings_module = getattr(
            transformer,
            "embeddings",
            None,
        )

        if embeddings_module is None:
            base_model = getattr(
                transformer,
                "base_model",
                None,
            )

            if base_model is not None:
                embeddings_module = getattr(
                    base_model,
                    "embeddings",
                    None,
                )

        if embeddings_module is not None:
            try:
                token_embs = embeddings_module(
                    input_ids=input_ids,
                )
            except TypeError:
                token_embs = embeddings_module(input_ids)

        elif hasattr(
            transformer,
            "get_input_embeddings",
        ):
            embedding_layer = (
                transformer.get_input_embeddings()
            )

            if embedding_layer is None:
                raise RuntimeError(
                    "HF transformer returned no input embedding layer."
                )

            token_embs = embedding_layer(input_ids)

        else:
            raise RuntimeError(
                "Could not locate the BiomedBERT embedding module."
            )

        if token_embs.ndim != 3:
            raise RuntimeError(
                "Caption embedding module must return [B,L,D], "
                f"got {tuple(token_embs.shape)}."
            )

        # BERT embeddings retain positional values at PAD positions.
        # Mask them explicitly so padding cannot become decoder content.
        token_mask = input_ids.ne(
            self.pad_id
        ).unsqueeze(-1)

        token_embs = token_embs * token_mask.to(
            token_embs.dtype
        )

        return token_embs

    def encode_ecg(
        self,
        ecgs,
        normalize: bool = True,
    ):
        ecg_latent, _ = self._encode_ecg(
            ecgs,
            normalize=normalize,
        )
        return ecg_latent

    def encode_text(
        self,
        text,
        normalize: bool = True,
    ):
        text_latent, _ = self._encode_text(
            text,
            normalize=normalize,
        )
        return text_latent

    def lock_ecg_tower(
        self,
        unlocked_groups=0,
        freeze_bn_stats=False,
    ):
        self.ecg.lock(
            unlocked_groups=unlocked_groups,
            freeze_bn_stats=freeze_bn_stats,
        )

    def lock_text_tower(
        self,
        unlocked_layers: int = 0,
        freeze_layer_norm: bool = True,
    ):
        self.text.lock(
            unlocked_layers,
            freeze_layer_norm,
        )

    def _augment_ecg_memory_with_dcm(
        self,
        ecg_latent: torch.Tensor,
        ecg_embs: torch.Tensor,
    ) -> torch.Tensor:
        if ecg_latent.ndim != 2:
            raise RuntimeError(
                "ecg_latent must have shape [B,D], "
                f"got {tuple(ecg_latent.shape)}."
            )

        if ecg_embs.ndim != 3:
            raise RuntimeError(
                "ecg_embs must have shape [B,N,D], "
                f"got {tuple(ecg_embs.shape)}."
            )

        if ecg_latent.shape[0] != ecg_embs.shape[0]:
            raise RuntimeError(
                "DCM batch mismatch: "
                f"ecg_latent={tuple(ecg_latent.shape)}, "
                f"ecg_embs={tuple(ecg_embs.shape)}."
            )

        dcm_token = self.dcm_adapter(
            ecg_latent
        ).unsqueeze(1)

        if dcm_token.shape[-1] != ecg_embs.shape[-1]:
            raise RuntimeError(
                "DCM width mismatch: "
                f"dcm_token={tuple(dcm_token.shape)}, "
                f"ecg_embs={tuple(ecg_embs.shape)}."
            )

        return torch.cat(
            [dcm_token, ecg_embs],
            dim=1,
        )

    # Backward-compatible method name.
    def _augment_ecg_memory_with_soft_prompt(
        self,
        ecg_latent,
        ecg_embs,
    ):
        return self._augment_ecg_memory_with_dcm(
            ecg_latent,
            ecg_embs,
        )

    def forward(
        self,
        ecg,
        text: Optional[torch.Tensor] = None,
        ecg_latent: Optional[torch.Tensor] = None,
        ecg_embs: Optional[torch.Tensor] = None,
        output_labels: bool = True,
    ):
        if ecg_latent is None or ecg_embs is None:
            ecg_latent, ecg_embs = self._encode_ecg(ecg)

        if text is None:
            return {
                "ecg_features": ecg_latent,
                "ecg_embs": ecg_embs,
            }

        if text.ndim != 2:
            raise RuntimeError(
                "Text IDs must have shape [B,L], "
                f"got {tuple(text.shape)}."
            )

        # Full bidirectional BiomedBERT output is retained only for
        # contrastive/retrieval text features.
        text_latent, _ = self._encode_text(text)

        out_dict = {
            "ecg_features": ecg_latent,
            "text_features": text_latent,
            "logit_scale": self.logit_scale.exp(),
        }

        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias

        # Stage1 / retrieval / contrastive-only.
        if not output_labels:
            return out_dict

        if text.shape[1] < 2:
            raise RuntimeError(
                "Caption forward requires at least two tokens, "
                f"got {tuple(text.shape)}."
            )

        # Autoregressive right shift.
        decoder_input_ids = text[:, :-1]
        labels = text[:, 1:]

        # Important: do not use bidirectional contextual token states here.
        token_embs = self._get_caption_token_embeddings(
            decoder_input_ids
        )

        # DCM is applied exactly once in forward, in both training and eval.
        if self.use_dcm:
            ecg_memory = self._augment_ecg_memory_with_dcm(
                ecg_latent=ecg_latent,
                ecg_embs=ecg_embs,
            )

            if (
                self.training
                and torch.rand(1).item() < 0.001
            ):
                print(
                    f"[DCM] enabled | "
                    f"token_norm="
                    f"{ecg_memory[:, 0].norm(dim=-1).mean().item():.3f} | "
                    f"memory_len={ecg_memory.shape[1]}"
                )
        else:
            ecg_memory = ecg_embs

        logits = self.text_decoder(
            ecg_memory,
            token_embs,
        )

        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "Caption batch mismatch: "
                f"logits={tuple(logits.shape)}, "
                f"labels={tuple(labels.shape)}."
            )

        if logits.shape[1] != labels.shape[1]:
            raise RuntimeError(
                "Caption sequence mismatch: "
                f"logits={tuple(logits.shape)}, "
                f"labels={tuple(labels.shape)}."
            )

        out_dict["logits"] = logits
        out_dict["labels"] = labels

        return out_dict


def _append_generation_placeholder(
    text: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    if text.ndim != 2:
        raise RuntimeError(
            "Generation text must have shape [B,L], "
            f"got {tuple(text.shape)}."
        )

    placeholder = torch.full(
        (text.shape[0], 1),
        fill_value=int(pad_token_id),
        dtype=text.dtype,
        device=text.device,
    )

    return torch.cat(
        [text, placeholder],
        dim=1,
    )


def generate(
    self,
    ecg,
    text=None,
    seq_len=30,
    max_seq_len=77,
    temperature=1.0,
    generation_type="beam_search",
    top_p=0.1,
    top_k=1,
    pad_token_id=None,
    eos_token_id=None,
    sot_token_id=None,
    num_beams=6,
    num_beam_groups=1,
    min_seq_len=5,
    stopping_criteria=None,
    repetition_penalty=1.0,
    fixed_output_length=False,
):
    assert (
        _has_transformers
    ), "Please install transformers for generate functionality."

    if seq_len <= min_seq_len:
        raise ValueError(
            f"seq_len={seq_len} must exceed min_seq_len={min_seq_len}."
        )

    if generation_type == "beam_search" and num_beam_groups != 1:
        raise ValueError(
            "This repository implementation supports standard beam search "
            "only. Set num_beam_groups=1."
        )

    with torch.no_grad():
        sot_token_id = (
            49406
            if sot_token_id is None
            else int(sot_token_id)
        )
        eos_token_id = (
            49407
            if eos_token_id is None
            else int(eos_token_id)
        )
        pad_token_id = (
            self.pad_id
            if pad_token_id is None
            else int(pad_token_id)
        )

        logit_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(
                    min_seq_len,
                    eos_token_id,
                ),
                RepetitionPenaltyLogitsProcessor(
                    repetition_penalty
                ),
            ]
        )

        if stopping_criteria is None:
            stopping_criteria = [
                MaxLengthCriteria(
                    max_length=seq_len
                )
            ]

        stopping_criteria = StoppingCriteriaList(
            stopping_criteria
        )

        device = ecg.device

        if generation_type == "beam_search":
            output = self._generate_beamsearch(
                ecg_inputs=ecg,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                sot_token_id=sot_token_id,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                min_seq_len=min_seq_len,
                stopping_criteria=stopping_criteria,
                logit_processor=logit_processor,
            )

            if (
                fixed_output_length
                and output.shape[1] < seq_len
            ):
                pad_len = seq_len - output.shape[1]

                return torch.cat(
                    (
                        output,
                        torch.full(
                            (
                                output.shape[0],
                                pad_len,
                            ),
                            fill_value=self.pad_id,
                            device=device,
                            dtype=output.dtype,
                        ),
                    ),
                    dim=1,
                )

            return output

        if generation_type == "top_p":
            logit_warper = GENERATION_TYPES[
                generation_type
            ](top_p)
        elif generation_type == "top_k":
            logit_warper = GENERATION_TYPES[
                generation_type
            ](top_k)
        else:
            raise ValueError(
                "generation_type has to be one of "
                f"{list(GENERATION_TYPES.keys())}"
            )

        ecg_latent, ecg_embs = self._encode_ecg(ecg)

        if text is None:
            text = torch.full(
                (ecg.shape[0], 1),
                fill_value=sot_token_id,
                device=device,
                dtype=torch.long,
            )

        was_training = self.training
        num_dims = len(text.shape)

        if num_dims == 1:
            text = text[None, :]

        self.eval()
        out = text

        while True:
            x = out[:, -max_seq_len:]
            cur_len = x.shape[1]

            generation_text = _append_generation_placeholder(
                x,
                pad_token_id,
            )

            logits = self(
                ecg,
                generation_text,
                ecg_latent=ecg_latent,
                ecg_embs=ecg_embs,
                output_labels=True,
            )["logits"][:, -1]

            mask = (
                (out[:, -1] == eos_token_id)
                | (out[:, -1] == pad_token_id)
            )

            sample = torch.full(
                (out.shape[0], 1),
                fill_value=pad_token_id,
                device=device,
                dtype=torch.long,
            )

            if mask.all():
                if not fixed_output_length:
                    break
            else:
                active_logits = logits[~mask, :]

                filtered_logits = logit_processor(
                    x[~mask, :],
                    active_logits,
                )

                filtered_logits = logit_warper(
                    x[~mask, :],
                    filtered_logits,
                )

                probs = F.softmax(
                    filtered_logits / temperature,
                    dim=-1,
                )

                if cur_len + 1 == seq_len:
                    sample[~mask, :] = eos_token_id
                else:
                    sample[~mask, :] = torch.multinomial(
                        probs,
                        1,
                    )

            out = torch.cat(
                (out, sample),
                dim=-1,
            )

            if stopping_criteria(out, None):
                break

        if num_dims == 1:
            out = out.squeeze(0)

        self.train(was_training)
        return out


def _generate_beamsearch(
    self,
    ecg_inputs,
    pad_token_id=None,
    eos_token_id=None,
    sot_token_id=None,
    num_beams=6,
    num_beam_groups=1,
    min_seq_len=5,
    stopping_criteria=None,
    logit_processor=None,
    logit_warper=None,
):
    del logit_warper

    if num_beam_groups != 1:
        raise ValueError(
            "Grouped/diverse beam search is not implemented correctly. "
            "Set num_beam_groups=1."
        )

    device = ecg_inputs.device
    original_batch_size = ecg_inputs.shape[0]

    ecg_inputs = torch.repeat_interleave(
        ecg_inputs,
        num_beams,
        dim=0,
    )

    ecg_latent, ecg_embs = self._encode_ecg(
        ecg_inputs
    )

    input_ids = torch.full(
        (
            original_batch_size * num_beams,
            1,
        ),
        fill_value=int(sot_token_id),
        device=device,
        dtype=torch.long,
    )

    beam_scorer = BeamSearchScorer(
        batch_size=original_batch_size,
        num_beams=num_beams,
        device=device,
        num_beam_groups=1,
    )

    logits_processor = (
        LogitsProcessorList(
            [
                MinLengthLogitsProcessor(
                    min_seq_len,
                    eos_token_id=eos_token_id,
                )
            ]
        )
        if logit_processor is None
        else logit_processor
    )

    num_beams = beam_scorer.num_beams
    batch_size = len(
        beam_scorer._beam_hyps
    )

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            "Beam dimension mismatch: "
            f"num_beams={num_beams}, "
            f"batch_size={batch_size}, "
            f"input_ids={tuple(input_ids.shape)}."
        )

    beam_scores = torch.full(
        (batch_size, num_beams),
        -1e9,
        device=device,
    )
    beam_scores[:, 0] = 0
    beam_scores = beam_scores.view(-1)

    next_tokens = None
    next_indices = None

    while True:
        model_inputs = prepare_inputs_for_generation(
            input_ids=input_ids,
            ecg_inputs=ecg_inputs,
        )

        generation_text = model_inputs["text"]

        generation_text_with_placeholder = (
            _append_generation_placeholder(
                generation_text,
                int(pad_token_id),
            )
        )

        outputs = self(
            model_inputs["ecgs"],
            generation_text_with_placeholder,
            ecg_latent=ecg_latent,
            ecg_embs=ecg_embs,
            output_labels=True,
        )

        if not isinstance(outputs, dict):
            raise RuntimeError(
                "Beam-search forward expected a dictionary, "
                f"got {type(outputs).__name__}."
            )

        if "logits" not in outputs:
            raise RuntimeError(
                "Beam-search forward did not return caption logits. "
                f"Returned keys: {list(outputs.keys())}"
            )

        if outputs["logits"].ndim != 3:
            raise RuntimeError(
                "Caption logits must have shape [B,L,V], "
                f"got {tuple(outputs['logits'].shape)}."
            )

        if outputs["logits"].shape[1] == 0:
            raise RuntimeError(
                "Caption decoder returned zero-length logits."
            )

        next_token_logits = outputs[
            "logits"
        ][:, -1, :]

        vocab_size = next_token_logits.shape[-1]

        next_token_scores = F.log_softmax(
            next_token_logits,
            dim=-1,
        )

        next_token_scores = logits_processor(
            input_ids,
            next_token_scores,
        )

        next_token_scores = (
            next_token_scores
            + beam_scores.unsqueeze(-1)
        )

        next_token_scores = next_token_scores.view(
            batch_size,
            num_beams * vocab_size,
        )

        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            2 * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = torch.div(
            next_tokens,
            vocab_size,
            rounding_mode="floor",
        )

        next_tokens = next_tokens % vocab_size

        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        beam_scores = beam_outputs[
            "next_beam_scores"
        ]
        beam_next_tokens = beam_outputs[
            "next_beam_tokens"
        ]
        beam_idx = beam_outputs[
            "next_beam_indices"
        ]

        input_ids = input_ids[beam_idx]

        input_ids = torch.cat(
            [
                input_ids,
                beam_next_tokens.unsqueeze(-1),
            ],
            dim=-1,
        )

        cur_len += 1

        if (
            beam_scorer.is_done
            or stopping_criteria(input_ids, None)
        ):
            break

    if next_tokens is None or next_indices is None:
        raise RuntimeError(
            "Beam search terminated before producing candidates."
        )

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    return sequence_outputs["sequences"]


def apply_train_stage(
    self,
    train_stage: str,
    use_dcm: bool = False,
):
    if train_stage == "stage1":
        # CLIP only.
        for parameter in self.text_decoder.parameters():
            parameter.requires_grad = False

        for parameter in self.ecg.parameters():
            parameter.requires_grad = True

        for parameter in self.text.parameters():
            parameter.requires_grad = True

        for parameter in self.dcm_adapter.parameters():
            parameter.requires_grad = False

        self.set_dcm_enabled(False)

        self.logit_scale.requires_grad = True

        if self.logit_bias is not None:
            self.logit_bias.requires_grad = True

    elif train_stage == "stage2":
        # Caption only.
        for parameter in self.text.parameters():
            parameter.requires_grad = False

        self.logit_scale.requires_grad = False

        if self.logit_bias is not None:
            self.logit_bias.requires_grad = False

        if hasattr(
            self.ecg,
            "lock_for_caption_token_level",
        ):
            self.ecg.lock_for_caption_token_level(
                freeze_bn_stats=True
            )
        else:
            raise RuntimeError(
                "ECG encoder lacks lock_for_caption_token_level"
            )

        for parameter in self.text_decoder.parameters():
            parameter.requires_grad = True

        self.set_dcm_enabled(use_dcm)

        for parameter in self.dcm_adapter.parameters():
            parameter.requires_grad = self.use_dcm

        print(
            f"[Stage2] use_dcm={self.use_dcm} | "
            f"dcm_trainable="
            f"{any(p.requires_grad for p in self.dcm_adapter.parameters())}"
        )

    else:
        raise ValueError(
            f"Unknown train stage {train_stage}"
        )


def prepare_inputs_for_generation(
    input_ids,
    ecg_inputs,
    past=None,
    **kwargs,
):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get(
        "attention_mask",
        None,
    )
    position_ids = kwargs.get(
        "position_ids",
        None,
    )

    if (
        attention_mask is not None
        and position_ids is None
    ):
        position_ids = (
            attention_mask.long().cumsum(-1) - 1
        )
        position_ids.masked_fill_(
            attention_mask == 0,
            1,
        )
    else:
        position_ids = None

    return {
        "text": input_ids,
        "ecgs": ecg_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
