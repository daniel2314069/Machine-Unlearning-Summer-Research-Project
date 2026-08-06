from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor

from .utils import package_versions, tensor_sha256


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def original_projection_modules(pipe, projection: str = "to_v"):
    suffix = f"attn2.{projection}"
    modules = [(name, module) for name, module in pipe.unet.named_modules() if name.endswith(suffix)]
    if not modules:
        raise RuntimeError(f"No original UNet modules matched {suffix}")
    return modules


def original_w0_fingerprint(pipe, projection: str = "to_v") -> str:
    return tensor_sha256((f"{name}.weight", module.weight) for name, module in original_projection_modules(pipe, projection))


def load_original_pipeline(
    config: dict[str, Any], purpose: str, include_vae: bool = True
):
    model = config["model"]
    dtype_key = model["generation_dtype"] if purpose == "generation" else model["embedding_dtype"]
    kwargs: dict[str, Any] = {"torch_dtype": DTYPES[dtype_key]}
    if model.get("disable_safety_checker", True):
        kwargs["safety_checker"] = None
    if not include_vae:
        kwargs["vae"] = None
    pipe = DiffusionPipeline.from_pretrained(model["model_id"], **kwargs)
    pipe = pipe.to(model["device"])
    for component_name in ("unet", "text_encoder", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.eval()
            component.requires_grad_(False)
    return pipe


def model_metadata(pipe, config: dict[str, Any], projection: str = "to_v") -> dict[str, Any]:
    scheduler = pipe.scheduler
    modules = original_projection_modules(pipe, projection)
    model_id = config["model"]["model_id"]
    try:
        checkpoint_path = Path(snapshot_download(model_id, local_files_only=True)).resolve()
        checkpoint_revision = checkpoint_path.name if checkpoint_path.parent.name == "snapshots" else str(checkpoint_path)
    except Exception:
        checkpoint_path = Path(model_id).expanduser().resolve() if Path(model_id).exists() else None
        checkpoint_revision = str(checkpoint_path) if checkpoint_path else "unresolved"
    return {
        "model_id": model_id,
        "checkpoint_revision": checkpoint_revision,
        "pipeline_class": type(pipe).__name__,
        "tokenizer_class": type(pipe.tokenizer).__name__,
        "tokenizer_model_max_length": pipe.tokenizer.model_max_length,
        "text_encoder_class": type(pipe.text_encoder).__name__,
        "text_hidden_size": pipe.text_encoder.config.hidden_size,
        "scheduler_class": type(scheduler).__name__,
        "scheduler_config": dict(scheduler.config),
        "projection": projection,
        "projection_layer_count": len(modules),
        "projection_layers": {name: list(module.weight.shape) for name, module in modules},
        "original_w0_in_memory_fingerprint_sha256": original_w0_fingerprint(pipe, projection),
        "original_w0_in_memory_dtype": str(modules[0][1].weight.dtype),
        "package_versions": package_versions(),
    }


class ClipEnsembleClassifier:
    """Closed-set CLIP scorer with explicit ambiguity checks handled by the caller."""

    def __init__(self, config: dict[str, Any], concepts: list[str]):
        classifier = config["classifier"]
        if classifier["backend"] != "clip":
            raise ValueError(f"Unsupported classifier backend: {classifier['backend']}")
        self.device = torch.device(classifier["device"])
        self.concepts = concepts
        self.templates = classifier["templates"]
        self.model = CLIPModel.from_pretrained(classifier["model_id"]).to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(classifier["model_id"])
        self.text_features = self._build_text_features()
        self.logit_scale = self.model.logit_scale.exp().detach()

    @torch.inference_mode()
    def _build_text_features(self) -> torch.Tensor:
        per_concept = []
        for concept in self.concepts:
            texts = [template.format(concept=concept) for template in self.templates]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            per_concept.append(features)
        return torch.stack(per_concept)

    @torch.inference_mode()
    def score(self, image) -> dict[str, Any]:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Average class logits across the configured prompt templates before softmax.
        template_logits = self.logit_scale * torch.einsum("bd,ctd->bct", image_features, self.text_features)
        logits = template_logits.mean(dim=-1).squeeze(0)
        probabilities = logits.softmax(dim=0)
        order = torch.argsort(probabilities, descending=True)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        normalized_entropy = float(entropy / np.log(len(self.concepts)))
        return {
            "class_logits": {name: float(logits[i]) for i, name in enumerate(self.concepts)},
            "class_probabilities": {name: float(probabilities[i]) for i, name in enumerate(self.concepts)},
            "top1_concept": self.concepts[int(order[0])],
            "top1_probability": float(probabilities[order[0]]),
            "runner_up_concept": self.concepts[int(order[1])],
            "runner_up_probability": float(probabilities[order[1]]),
            "normalized_entropy": normalized_entropy,
        }
