from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from src.data.dataset import default_image_loader


@dataclass
class MTLSample:
    """
    단일 multi-task 샘플.

    - task: "caption" | "summary" | "cot_link" | "text_summary"
    - image_path: 이미지가 필요한 태스크는 실제 경로, text-only 태스크는 None
    - prompt: LLM 입력 프롬프트 텍스트
    - target: LLM이 생성해야 할 타겟 텍스트
    """

    task: str
    image_path: Optional[str]
    prompt: str
    target: str


class ElementCaptionDataset(Dataset):
    """
    Task A: 요소 이미지 → Gemini caption 재구성.

    - 원본 JSON: weatherqa_gemini_batch.json 형식
    - 각 element에 대해:
        - prompt: 4문장 설명을 요구하는 meteorology caption 프롬프트
        - target: Gemini caption 전체 (4 sentences)
    """

    def __init__(self, json_path: str, image_root: str) -> None:
        super().__init__()
        self.samples: List[MTLSample] = []

        root = Path(os.path.expanduser(image_root)).resolve()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data:
            elements = case.get("elements", [])
            if not elements:
                continue
            for el in elements:
                if el.get("error") is not None:
                    continue
                caption: str = (el.get("caption") or "").strip()
                if not caption:
                    continue
                image_rel: str = el.get("image_rel_path") or ""
                if not image_rel:
                    continue

                rel_path = Path(image_rel)
                if rel_path.is_absolute():
                    image_path = rel_path
                else:
                    image_path = root / rel_path

                prompt = (
                    "You are a meteorologist.\n"
                    "Describe this meteorological element image in exactly 4 sentences.\n"
                    "Sentence 1: State which variables are shown by the color fill and contours.\n"
                    "Sentence 2: Describe the dominant visual feature, its location, and wind flow.\n"
                    "Sentence 3: Explain the physical meaning of these patterns.\n"
                    "Sentence 4: Briefly connect this element to the broader weather context.\n\n"
                    "Caption:"
                )

                self.samples.append(
                    MTLSample(
                        task="caption",
                        image_path=str(image_path),
                        prompt=prompt,
                        target=caption,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MTLSample:
        return self.samples[idx]


class SummaryFromImageDataset(Dataset):
    """
    Task B: MD-style summary (이미지 → 요약).

    - Gemini batch JSON (weatherqa_gemini_batch.json)을 사용한다.
      각 case의 `global_summary`를 해당 case의 모든 요소 이미지(image_rel_path)와 pairing하여
      (image, summary) 샘플을 구성한다.
    """

    def __init__(self, json_path: str, image_root: str) -> None:
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples: List[MTLSample] = []

        root = Path(os.path.expanduser(image_root)).resolve()
        for case in data:
            global_summary = (case.get("global_summary") or "").strip()
            elements = case.get("elements", [])
            if not global_summary or not elements:
                continue

            for el in elements:
                if el.get("error") is not None:
                    continue
                image_rel = el.get("image_rel_path") or ""
                if not image_rel:
                    continue

                rel_path = Path(image_rel)
                if rel_path.is_absolute():
                    image_path = rel_path
                else:
                    image_path = root / rel_path

                prompt = (
                    "You are a meteorologist at SPC.\n"
                    "Given this mesoscale diagnostic image, write a concise severe weather SUMMARY (2–3 sentences).\n"
                    "Must clearly state hazards (hail/wind/tornado), affected regions, timing/evolution, and whether a watch is needed.\n\n"
                    "Summary:"
                )

                self.samples.append(
                    MTLSample(
                        task="summary",
                        image_path=str(image_path),
                        prompt=prompt,
                        target=global_summary,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MTLSample:
        return self.samples[idx]


def _split_sentences(text: str) -> List[str]:
    """
    Gemini caption이 4문장이라고 가정하고, 간단한 정규식으로 문장 분리.
    완벽하지는 않지만, S4(요약과의 연결)를 대략 분리하는 용도로 사용.
    """
    # 마침표/느낌표/물음표 뒤의 공백 기준으로 split
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # 빈 문자열 제거
    parts = [p.strip() for p in parts if p.strip()]
    return parts


class ElementSummaryLinkDataset(Dataset):
    """
    Task C: 요소 caption + global summary → connection/explanation 한 문장.

    - caption에서 마지막 문장(S4)을 summary와 연결하는 문장으로 보고 target으로 사용.
    - prompt에는 caption 전체와 global_summary를 포함.
    """

    def __init__(self, json_path: str, image_root: str) -> None:
        super().__init__()
        self.samples: List[MTLSample] = []

        root = Path(os.path.expanduser(image_root)).resolve()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data:
            global_summary: str = (case.get("global_summary") or "").strip()
            elements = case.get("elements", [])
            if not global_summary or not elements:
                continue

            for el in elements:
                if el.get("error") is not None:
                    continue
                caption: str = (el.get("caption") or "").strip()
                image_rel: str = el.get("image_rel_path") or ""
                if not caption or not image_rel:
                    continue

                rel_path = Path(image_rel)
                if rel_path.is_absolute():
                    image_path = rel_path
                else:
                    image_path = root / rel_path

                sentences = _split_sentences(caption)
                if not sentences:
                    continue
                explanation = sentences[-1]  # S4로 가정
                # 너무 짧게 잘리면 마지막 두 문장을 합쳐 사용
                if len(explanation.split()) < 4 and len(sentences) >= 2:
                    explanation = " ".join(sentences[-2:])

                # 프롬프트: 지시문을 앞에 두고, caption은 앞부분만 사용해 길이 절약
                cap_sentences = _split_sentences(caption)
                cap_brief = " ".join(cap_sentences[:2]) if cap_sentences else caption
                prompt = (
                    "You are a meteorologist. Explain how this element supports or refines the forecast summary.\n\n"
                    f"Forecast summary:\n{global_summary}\n\n"
                    f"Element description:\n{cap_brief}\n\n"
                    "Link explanation:\n"
                )

                self.samples.append(
                    MTLSample(
                        task="cot_link",
                        image_path=str(image_path),
                        prompt=prompt,
                        target=explanation,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MTLSample:
        return self.samples[idx]


class CaseSummaryFromCaptionsDataset(Dataset):
    """
    Task D (선택적): 여러 요소 caption 리스트 → global forecast summary.

    이 태스크는 text-only로도 구성할 수 있지만, 간단히 case마다 대표 요소
    이미지 하나를 image_path로 사용하고, LLM은 caption들을 이용해 summary를
    재구성하도록 학습한다.
    """

    def __init__(self, json_path: str, image_root: str) -> None:
        super().__init__()
        self.samples: List[MTLSample] = []

        root = Path(os.path.expanduser(image_root)).resolve()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data:
            global_summary: str = (case.get("global_summary") or "").strip()
            elements = case.get("elements", [])
            if not global_summary or not elements:
                continue

            # 요소 caption 여러 개를 numbering해서 prompt에 넣는다.
            captions = []
            for el in elements:
                if el.get("error") is not None:
                    continue
                cap = (el.get("caption") or "").strip()
                if cap:
                    captions.append(cap)
            if not captions:
                continue

            # 대표 이미지: 첫 번째 요소 이미지 사용
            first_image_rel = elements[0].get("image_rel_path") or ""
            if first_image_rel:
                rel_path = Path(first_image_rel)
                if rel_path.is_absolute():
                    image_path = rel_path
                else:
                    image_path = root / rel_path
            else:
                image_path = None

            captions_block = ""
            for i, cap in enumerate(captions[:10], start=1):  # 너무 길어지지 않게 최대 10개
                captions_block += f"{i}. {cap}\n"

            prompt = (
                "You are a meteorologist.\n"
                "You are given several element-level descriptions from diagnostic plots.\n\n"
                "Element descriptions:\n"
                f"{captions_block}\n"
                "Write a concise forecast summary in the style of SPC Mesoscale Discussions,\n"
                "focusing on the key hazards, affected regions, and timing.\n\n"
                "Summary:"
            )

            self.samples.append(
                MTLSample(
                    task="text_summary",
                    image_path=str(image_path) if image_path is not None else None,
                    prompt=prompt,
                    target=global_summary,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MTLSample:
        return self.samples[idx]


def build_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def mtl_collate_fn(
    batch: List[MTLSample],
    tokenizer,
    image_transform,
    max_seq_len: int,
) -> Dict[str, Any]:
    """
    Multi-task용 collate 함수.

    - prompt와 target을 분리 토큰화하여 prompt 부분 label은 -100으로 마스킹,
      target 부분만 loss에 사용되도록 한다.
    - 이미지가 없는 text-only 태스크(text_summary)는 image_path가 None일 수 있으며,
      이 경우 zero tensor 이미지를 사용하거나, 상위 코드에서 Stage1 encoder를 생략할 수 있다.
    """
    pixel_values_list: List[torch.Tensor] = []
    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    tasks: List[str] = []

    for sample in batch:
        tasks.append(sample.task)

        # 이미지 로딩 (없으면 zero tensor)
        if sample.image_path is not None and os.path.exists(sample.image_path):
            img = default_image_loader(sample.image_path)
            pv = image_transform(img)
        else:
            pv = torch.zeros(3, image_transform.transforms[0].size[0], image_transform.transforms[0].size[1])
        pixel_values_list.append(pv)

        # prompt/target 토큰화 (target이 짤리지 않도록 prompt 길이를 강제로 자른다)
        prompt_ids = tokenizer(
            sample.prompt, add_special_tokens=True, truncation=True, max_length=max_seq_len
        )["input_ids"]
        target_ids = tokenizer(
            sample.target, add_special_tokens=False, truncation=True, max_length=max_seq_len
        )["input_ids"]
        # target이 비면 의미 없는 학습이 되므로 최소 1토큰 보장
        if len(target_ids) == 0:
            target_ids = [tokenizer.eos_token_id]
        # target을 위해 최소 공간 확보
        min_target = min(len(target_ids), max_seq_len // 2)
        allow_prompt = max_seq_len - min_target
        prompt_ids = prompt_ids[:allow_prompt]

        ids = prompt_ids + target_ids
        ids = ids[:max_seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long)

        # attention mask
        attn = torch.ones_like(input_ids)

        # labels: prompt 부분은 -100, target 부분만 실제 토큰
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(ids))
        labels[:prompt_len] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attn)
        labels_list.append(labels)

    # padding
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )
    pixel_values = torch.stack(pixel_values_list, dim=0)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "tasks": tasks,
    }


def build_mtl_dataset_and_loader(
    gemini_jsons: List[str],
    image_root: str,
    tokenizer,
    image_size: int,
    max_seq_len: int,
    batch_size: int,
    train_val_split: float,
    num_workers: int,
    include_caption: bool = True,
    include_summary: bool = True,
    include_cot_link: bool = True,
    include_text_summary: bool = False,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    """
    여러 태스크용 dataset을 생성하고 하나의 DataLoader로 합쳐 반환한다.
    """
    datasets: List[Dataset] = []

    if include_caption:
        for gj in gemini_jsons:
            datasets.append(ElementCaptionDataset(gj, image_root))
    if include_cot_link:
        for gj in gemini_jsons:
            datasets.append(ElementSummaryLinkDataset(gj, image_root))
    if include_text_summary:
        for gj in gemini_jsons:
            datasets.append(CaseSummaryFromCaptionsDataset(gj, image_root))
    if include_summary:
        for gj in gemini_jsons:
            datasets.append(SummaryFromImageDataset(gj, image_root))

    if not datasets:
        raise ValueError("No tasks selected for MTL (all include_* flags are False).")

    full_dataset = ConcatDataset(datasets)

    total_len = len(full_dataset)
    train_len = int(total_len * train_val_split)
    val_len = total_len - train_len
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_len, val_len])

    image_transform = build_image_transform(image_size)

    collate = lambda batch: mtl_collate_fn(
        batch,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=max_seq_len,
    )

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return train_loader, val_loader, train_sampler, val_sampler
