"""Qwen3.5 video input and action-generation utilities."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.video_utils import VideoMetadata

from core import prompt_text


def frames_from_tar(path, max_frames):
    with tarfile.open(path) as tar:
        meta = json.load(tar.extractfile("metadata.json"))
        if meta.get("format") != "easyvideor1_preprocessed_video":
            raise ValueError(f"Unsupported frame cache: {path}")
        count = int(meta["num_frames"])
        if count < 1:
            raise ValueError(f"Empty frame cache: {path}")
        chosen = np.linspace(0, count - 1, min(count, max_frames)).round().astype(int).tolist()
        frames = np.stack([
            np.asarray(Image.open(io.BytesIO(tar.extractfile(f"frames/{i:06d}.jpg").read())).convert("RGB"))
            for i in chosen
        ])
        source = meta["metadata"]
        metadata = VideoMetadata(
            total_num_frames=source["total_num_frames"],
            fps=source["fps"],
            frames_indices=[int(source["frames_indices"][i]) for i in chosen],
            video_backend="tar_cache",
        )
        return frames, metadata


def frames_from_file(path, max_frames):
    """Seek uniformly through an mp4, retaining timestamps without writing frames."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 1.0
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration) / 1_000_000
        if duration <= 0:
            raise ValueError(f"Unknown video duration: {path}")
        frames, indices = [], []
        for i in range(max_frames):
            seconds = duration * (i + 0.5) / max_frames
            timestamp = int(seconds / float(stream.time_base))
            container.seek(timestamp, stream=stream, backward=True)
            picked = None
            for frame in container.decode(stream):
                picked = frame
                if frame.pts is not None and float(frame.pts * stream.time_base) >= seconds:
                    break
            if picked is None:
                raise ValueError(f"Could not decode video frame: {path}")
            frames.append(picked.to_ndarray(format="rgb24"))
            actual_seconds = float(picked.pts * stream.time_base) if picked.pts is not None else seconds
            indices.append(int(round(actual_seconds * fps)))
        total = stream.frames or int(round(duration * fps))
    metadata = VideoMetadata(
        total_num_frames=max(total, max(indices)+1), fps=fps,
        frames_indices=indices, video_backend="pyav_seek",
    )
    return np.stack(frames), metadata


def prepare_video(processor, row, max_frames=8, max_pixels=65536, prompt_override=None):
    cache = row.get("frame_cache_path")
    if cache:
        frames, metadata = frames_from_tar(cache, max_frames)
        video = {"type": "video", "video": frames}
        kwargs = {"do_sample_frames": False, "video_metadata": [metadata], "max_pixels": max_pixels}
    else:
        video_path = Path(row["video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        frames, metadata = frames_from_file(video_path, max_frames)
        video = {"type": "video", "video": frames}
        kwargs = {"do_sample_frames": False, "video_metadata": [metadata], "max_pixels": max_pixels}
    messages = [{"role": "user", "content": [video, {"type": "text", "text": prompt_override if prompt_override is not None else prompt_text(row)}]}]
    batch = dict(processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs=kwargs,
    ))
    batch.pop("video_metadata", None)
    if "pixel_values_videos" not in batch or "video_grid_thw" not in batch:
        raise RuntimeError("Video was not encoded; refusing text-only training")
    return batch


def to_device(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def append_tokens(batch, tokens):
    """Extend the encoded multimodal prompt with text tokens for a full forward."""
    result = dict(batch)
    tokens = torch.as_tensor(tokens, dtype=batch["input_ids"].dtype, device=batch["input_ids"].device).reshape(1, -1)
    result["input_ids"] = torch.cat([batch["input_ids"], tokens], dim=1)
    result["attention_mask"] = torch.cat(
        [batch["attention_mask"], torch.ones_like(tokens)], dim=1
    )
    if "mm_token_type_ids" in batch:
        result["mm_token_type_ids"] = torch.cat(
            [batch["mm_token_type_ids"], torch.zeros_like(tokens)], dim=1
        )
    return result


def completion_logprobs(model, batch, completion_ids):
    prompt_len = batch["input_ids"].shape[1]
    full = append_tokens(batch, completion_ids)
    outputs = model(**full, use_cache=False, logits_to_keep=0)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].float()
    token_ids = full["input_ids"][:, prompt_len:]
    return logits.log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)[0]


def generate_completion(model, processor, batch, max_new_tokens=24, temperature=1.0, top_p=0.95, sample=True):
    with torch.no_grad():
        ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    completion = ids[0, batch["input_ids"].shape[1]:]
    eos = processor.tokenizer.eos_token_id
    if eos is not None:
        stops = (completion == eos).nonzero(as_tuple=True)[0]
        if len(stops):
            completion = completion[:int(stops[0])+1]
    text = processor.tokenizer.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return completion.detach(), text


def action_label_tokens(processor, num_choices):
    """Tokenize labels at their actual position after ``Action:``."""
    tokenizer = processor.tokenizer
    prefix = tokenizer.encode('Action:', add_special_tokens=False)
    ids = []
    for i in range(num_choices):
        completed = tokenizer.encode(f'Action: {chr(65+i)}', add_special_tokens=False)
        if completed[:len(prefix)] != prefix or len(completed) != len(prefix)+1:
            raise ValueError('Action label must be one token at the answer position')
        ids.append(completed[-1])
    if len(set(ids)) != len(ids):
        raise ValueError('Action label token IDs must be distinct')
    return prefix, ids


def native_action_scores(model, processor, batch, num_choices):
    prefix, ids = action_label_tokens(processor, num_choices)
    with torch.no_grad():
        full_logits = model(**append_tokens(batch, prefix), use_cache=False,
                            logits_to_keep=1).logits[0, -1].float()
        label_logits = full_logits[ids]
        label_mass = (label_logits.logsumexp(0) - full_logits.logsumexp(0)).exp()
    return {'probabilities': label_logits.softmax(-1).tolist(),
            'label_mass': float(label_mass)}


def native_action_probs(model, processor, batch, num_choices):
    return native_action_scores(model, processor, batch, num_choices)['probabilities']


def load_model(model_path, device, lora_rank=0):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    )
    if lora_rank:
        from peft import LoraConfig, get_peft_model
        model.requires_grad_(False)
        language = model.model.language_model
        targets = [name for name, layer in language.named_modules() if isinstance(layer, torch.nn.Linear)]
        model.model.language_model = get_peft_model(
            language,
            LoraConfig(r=lora_rank, lora_alpha=2*lora_rank, target_modules=targets, lora_dropout=0.0, bias="none"),
        )
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("No LoRA parameters are trainable")
    model.to(device)
    return processor, model
