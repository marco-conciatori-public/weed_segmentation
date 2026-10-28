"""Species classifier: a DINOv3 or DINOv2 vision transformer (timm, Apache-2.0) with a linear
head. DINOv2's weights are Apache-2.0 too, DINOv3's under Meta's DINOv3 License."""

from typing import Any

import timm
import torch
from torch import nn

from weedseg.paths import resolve


def build_classifier(num_classes: int, config: dict[str, Any], input_size: int) -> nn.Module:
    """A timm backbone (`backbone`, e.g. vit_base_patch14_dinov2.lvd142m) with a linear head
    for `num_classes`, at `input_size` px (position embeddings are resized to it).

    With `pretrained`, the backbone weights are read from `pretrained_path` when set, else
    downloaded once from Hugging Face to its cache. `freeze_blocks` freezes the patch
    embedding and that many transformer blocks (0 = train everything).
    """
    options: dict[str, Any] = {"num_classes": num_classes, "img_size": input_size,
                               "drop_path_rate": config["drop_path"]}
    if config["pretrained"] and config["pretrained_path"]:
        path = resolve(config["pretrained_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained weights not found: {path}. Set "
                                    "model.pretrained_path to null to download them")
        options["pretrained_cfg_overlay"] = {"file": str(path)}
    model = timm.create_model(config["backbone"], pretrained=config["pretrained"], **options)
    if config["freeze_blocks"]:
        frozen = [model.patch_embed, *model.blocks[:config["freeze_blocks"]]]
        for module in frozen:
            for parameter in module.parameters():
                parameter.requires_grad = False
        for name in ("cls_token", "reg_token", "pos_embed"):
            if getattr(model, name, None) is not None:
                getattr(model, name).requires_grad = False
    return model


def parameter_groups(model: nn.Module, lr_backbone: float, lr_head: float,
                     weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: the head at `lr_head`, the backbone at `lr_backbone`; no weight decay on
    biases, norms and embeddings (1-D tensors and tokens)."""
    head = {id(p) for p in model.get_classifier().parameters()}
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim <= 1 or name.endswith(("token", "pos_embed"))
        groups.setdefault((id(parameter) in head, no_decay), []).append(parameter)
    return [{"params": params, "lr": lr_head if is_head else lr_backbone,
             "weight_decay": 0.0 if no_decay else weight_decay}
            for (is_head, no_decay), params in groups.items()]


def load_classifier(path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Classifier from a checkpoint saved by scripts/train_classifier.py, plus the checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_classifier(len(checkpoint["classes"]), {**config["model"], "pretrained": False},
                             config["crops"]["input_size"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
