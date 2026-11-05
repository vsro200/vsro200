"""
Beam search with confidence-weighted fusion of Whisper (audio) and VSRo (video).

Decoding modes:
  - "fusion": at every step, both branches give a distribution over the next
              token. The weight of the audio branch is

                  alpha = H(P_video) / (H(P_audio) + H(P_video))

              where H is the Shannon entropy, and the fused distribution is

                  P_fused = alpha * P_audio + (1 - alpha) * P_video.

              The branch that is less certain at that step counts for less,
              and alpha is recomputed for every token and every beam.
  - "audio":  audio-only baseline (Whisper alone).
  - "video":  video-only baseline (the VSR model alone).

All three modes share this decoder, so differences between them come from
the information available, not from the search procedure.

Decoder structure follows MultiVSR's beam search (search.py), with the
modification that the next-token distribution is computed from both
modalities at each step.
"""

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("fusion", "audio", "video")

# Added inside every log and every entropy, so that a token with zero
# probability in both branches does not produce -inf or NaN.
EPS = 1e-8


def tile(x, count, dim=0):
    """Replicate the tensor `count` times along dimension `dim`."""
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def subsequent_mask(size: int) -> torch.Tensor:
    """Upper-triangular causal mask for the decoder."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def beam_search_fusion(
    whisper_model,        # WhisperForConditionalGeneration.model (encoder + decoder)
    whisper_proj,         # WhisperForConditionalGeneration.proj_out
    audio_embeds,         # (1, T_a, D_a) : Whisper encoder output
    multivsr_model,       # the encoder-decoder VSR model
    video_memory,         # (1, T_v, D_v) : VSR encoder output
    src_mask,             # (1, 1, T_v) : VSR encoder mask
    bos_indices,          # list of BOS token IDs (start-of-transcript prompt)
    eos_index,            # EOS token ID
    pad_index,            # PAD token ID (typically 0)
    max_output_length,    # maximum tokens to generate
    size=5,               # beam size
    length_penalty_alpha=1.0,  # GNMT length penalty exponent, not the fusion weight
    n_best=1,             # number of hypotheses to return
    mode="fusion",        # "fusion" | "audio" | "video"
    stats=None,           # optional dict; in fusion mode, collects per-step
                          # mean alpha and entropies of both branches
):
    """
    Beam search decoding over the audio, video or fused distribution.

    - "fusion": P_fused = alpha * P_audio + (1 - alpha) * P_video, with
                alpha = H(P_video) / (H(P_audio) + H(P_video)) per step
    - "audio":  log_softmax(whisper_logits)
    - "video":  log_softmax(vsr_logits)

    `length_penalty_alpha` is the exponent of the GNMT length normalisation
    applied when ranking hypotheses. It has nothing to do with the fusion
    weight alpha above.

    Currently supports batch_size=1 only.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    batch_size = src_mask.size(0)
    assert batch_size == 1, "Currently only batch_size=1 is supported"

    device = src_mask.device

    # Replicate encoder outputs `size` times for parallel beam search
    audio_embeds = tile(audio_embeds.contiguous(), size, dim=0)
    video_memory = tile(video_memory.contiguous(), size, dim=0)
    src_mask = tile(src_mask, size, dim=0)

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
    beam_offset = torch.arange(0, batch_size * size, step=size,
                               dtype=torch.long, device=device)

    # Initialize sequences with BOS prompt
    alive_seq = torch.zeros([batch_size * size, len(bos_indices)],
                            dtype=torch.long, device=device)
    for i, tok in enumerate(bos_indices):
        alive_seq[:, i] = tok

    # First beam gets log-prob 0, the rest -inf so they don't compete on step 1
    topk_log_probs = torch.zeros(batch_size, size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    hypotheses = [[] for _ in range(batch_size)]
    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
    }

    for step in range(max_output_length):
        decoder_input = alive_seq

        # ── Whisper logits ────────────────────────────────────────────
        if mode != "video":
            out_audio = whisper_model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=audio_embeds,
            )
            logits_audio = whisper_proj(out_audio.last_hidden_state[:, -1])
        else:
            logits_audio = None

        # ── VSR logits ───────────────────────────────────────────────
        if mode != "audio":
            tgt_mask = subsequent_mask(decoder_input.size(1)).to(device).long()
            logits_video = multivsr_model.decode(
                video_memory, src_mask, decoder_input, tgt_mask
            )
            logits_video = logits_video.reshape(
                (decoder_input.size(0), -1, logits_video.size(-1))
            )
            logits_video = logits_video[:, -1]
        else:
            logits_video = None

        # ── Next-token distribution ──────────────────────────────────
        if mode == "audio":
            log_probs = F.log_softmax(logits_audio, dim=-1)
        elif mode == "video":
            log_probs = F.log_softmax(logits_video, dim=-1)
        else:
            # Both models use the same multilingual Whisper tokenizer, so the
            # two vocabularies are index for index the same. If they ever were
            # not, averaging probabilities would mix unrelated tokens, which is
            # worse than failing.
            if logits_audio.size(-1) != logits_video.size(-1):
                raise ValueError(
                    f"Vocabulary sizes differ: audio {logits_audio.size(-1)}, "
                    f"video {logits_video.size(-1)}. Fusion needs the same "
                    "tokenizer on both branches."
                )

            probs_audio = F.softmax(logits_audio, dim=-1)
            probs_video = F.softmax(logits_video, dim=-1)

            # Entropy of each branch at this step, one value per beam.
            ent_audio = -torch.sum(probs_audio * torch.log(probs_audio + EPS),
                                   dim=-1, keepdim=True)
            ent_video = -torch.sum(probs_video * torch.log(probs_video + EPS),
                                   dim=-1, keepdim=True)

            # Weight of the audio branch: large when the video branch is the
            # more uncertain one, small when audio is.
            alpha_fuse = ent_video / (ent_audio + ent_video + EPS)

            if stats is not None:
                stats.setdefault("alpha", []).append(alpha_fuse.mean().item())
                stats.setdefault("ent_audio", []).append(ent_audio.mean().item())
                stats.setdefault("ent_video", []).append(ent_video.mean().item())

            probs_fused = alpha_fuse * probs_audio + (1 - alpha_fuse) * probs_video
            log_probs = torch.log(probs_fused + EPS)

        # ── Standard beam search expansion ───────────────────────────
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        output_size = log_probs.size(-1)

        if length_penalty_alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** length_penalty_alpha
            curr_scores /= length_penalty

        curr_scores = curr_scores.reshape(-1, size * output_size)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if length_penalty_alpha > -1:
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        try:
            topk_beam_index = topk_ids.div(output_size, rounding_mode="floor")
        except TypeError:
            topk_beam_index = topk_ids.div(output_size)
        topk_ids = topk_ids.fmod(output_size)

        batch_index = (
            topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
        )
        select_indices = batch_index.view(-1)

        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )

        # ── Termination ──────────────────────────────────────────────
        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        end_condition = is_finished[:, 0].eq(True)

        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                for j in finished_hyp:
                    if (predictions[i, j, 1:] == eos_index).nonzero(
                        as_tuple=False
                    ).numel() < 2:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:])
                        )
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

            non_finished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            if len(non_finished) == 0:
                break
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        select_indices = batch_index.view(-1)
        audio_embeds = audio_embeds.index_select(0, select_indices)
        video_memory = video_memory.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

    return results["predictions"], results["scores"]
