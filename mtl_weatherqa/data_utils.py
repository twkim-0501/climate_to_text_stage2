"""
Common data utilities for the mtl_weatherqa workflows.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from torchvision import transforms


def build_image_transform(image_size: int, imagenet_norm: bool = True):
    """Resize + normalize transform."""
    norm = (
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if imagenet_norm
        else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ]
    )


def resolve_image_path(rel_path: str, image_root: str) -> Optional[str]:
    """Resolve image path for WeatherQA-style folders."""
    rel_clean = rel_path.lstrip("./")
    candidates = [
        os.path.join(image_root, rel_clean),
        os.path.join(image_root, "WeatherQA_MD_2014-2019", rel_clean),
        os.path.join(image_root, "WeatherQA_MD_2020", rel_clean),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def select_samples_gemini(
    json_path: str, image_root: str, num_samples: int, distinct_cond: bool = True
) -> List[Dict]:
    """
    Select samples from a Gemini JSON.
    - Picks at most one per cond_name if distinct_cond=True.
    - Returns dicts with keys: image, cond_name, summary, caption.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[Dict] = []
    seen = set()

    for case in data:
        summary = (case.get("global_summary") or "").strip()
        for el in case.get("elements", []):
            cond = el.get("cond_name") or el.get("type_name")
            rel = el.get("image_rel_path")
            cap = (el.get("caption") or "").strip()
            if not (summary and cond and rel):
                continue
            if distinct_cond and cond in seen:
                continue
            img_path = resolve_image_path(rel, image_root)
            if not img_path:
                continue
            seen.add(cond)
            out.append({"image": img_path, "cond_name": cond, "summary": summary, "caption": cap})
            if len(out) >= num_samples:
                return out
    return out
