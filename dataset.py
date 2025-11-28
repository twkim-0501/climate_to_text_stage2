import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset


@dataclass
class ClimateSample:
    """
    하나의 데이터 샘플을 표현하는 단순 자료 구조.

    - image_path : 이미지 파일의 절대/상대 경로
    - annotation : 이미지에 대한 텍스트 설명 (타겟)
    - cond_name  : 이미지 종류 텍스트 (예: 'wqa:shr6' 또는 'WeatherQA: Rainmap')
    """

    image_path: str
    annotation: str
    cond_name: str


class ClimateToTextDataset(Dataset):
    """
    Stage1 전처리 결과(JSON)를 읽어서 PyTorch Dataset으로 만드는 클래스.

    JSON 예시 (kse/ClimateToText/stage1_preprocessed_items.json):
    [
        {
            "image": "/path/to/image.gif",
            "annotation": "텍스트 설명 ...",
            "cond_name": "wqa:shr6"
        },
        ...
    ]

    이 Dataset은 샘플의 "원시 정보"만 반환하고,
    토크나이징/패딩/이미지 변환(Resize/Normalize 등)은 collate_fn 또는 상위 코드에서 처리한다.
    """

    def __init__(
        self,
        json_path: str,
        cond_name_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            json_path: 샘플이 저장된 JSON 파일 경로.
            cond_name_to_id: cond_name(string)를 class id(int)로 매핑하는 딕셔너리.
                             None이면 __init__에서 자동으로 생성한다.
        """
        super().__init__()

        with open(json_path, "r", encoding="utf-8") as f:
            raw_items: List[Dict[str, Any]] = json.load(f)

        self.samples: List[ClimateSample] = []
        for item in raw_items:
            self.samples.append(
                ClimateSample(
                    image_path=item["image"],
                    annotation=item["annotation"],
                    cond_name=item["cond_name"],
                )
            )

        # cond_name → class id 매핑이 주어지지 않으면, 지금 데이터에서 자동 생성
        if cond_name_to_id is None:
            unique_names = sorted({s.cond_name for s in self.samples})
            cond_name_to_id = {name: idx for idx, name in enumerate(unique_names)}

        self.cond_name_to_id: Dict[str, int] = cond_name_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        반환값은 나중에 collate_fn에서 처리하기 쉽게 dict 형태로 구성한다.

        - image_path : 이미지 경로 (collate_fn에서 실제 이미지를 로딩)
        - annotation : 타겟 텍스트
        - cond_name  : 클래스 이름 (예: 'wqa:shr6')
        - cond_id    : cond_name을 숫자 index로 매핑한 값
        """
        sample = self.samples[idx]
        cond_id = self.cond_name_to_id[sample.cond_name]

        return {
            "image_path": sample.image_path,
            "annotation": sample.annotation,
            "cond_name": sample.cond_name,
            "cond_id": cond_id,
        }


def default_image_loader(image_path: str) -> Image.Image:
    """
    디스크에서 이미지를 읽어 PIL.Image 객체로 반환하는 유틸 함수.

    - GIF/PNG/JPEG 등 대부분의 포맷을 지원하며,
    - 일관성을 위해 항상 RGB 모드로 변환한다.
    """
    img = Image.open(image_path)
    # 팔레트/그레이스케일 이미지를 포함해 항상 RGB로 통일
    return img.convert("RGB")


def build_cond_name_mapping(json_path: str) -> Dict[str, int]:
    """
    JSON 전체를 스캔하여 cond_name → class id 매핑을 생성하는 헬퍼 함수.

    Stage2 전체에서 동일한 매핑을 써야 하므로,
    학습 스크립트에서 이 함수를 사용해 매핑을 저장/로드하는 것을 권장한다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_items: List[Dict[str, Any]] = json.load(f)

    unique_names = sorted({item["cond_name"] for item in raw_items})
    return {name: idx for idx, name in enumerate(unique_names)}


def climate_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    tokenizer,
    image_transform,
    max_seq_len: int,
) -> Dict[str, Any]:
    """
    DataLoader에 넘길 collate_fn 구현.

    이 함수에서:
    - 이미지를 디스크에서 읽고, 주어진 image_transform을 적용해 tensor로 변환
    - annotation 텍스트를 토크나이저로 변환 (padding/truncation)
    - cond_id를 tensor로 묶어서 배치로 만든다.
    """
    # 이미지 로딩 및 변환
    images: List[torch.Tensor] = []
    cond_ids: List[int] = []
    annotations: List[str] = []
    image_paths: List[str] = []

    for item in batch:
        img = default_image_loader(item["image_path"])
        img_tensor = image_transform(img)
        images.append(img_tensor)
        cond_ids.append(int(item["cond_id"]))
        annotations.append(item["annotation"])
        image_paths.append(item["image_path"])

    pixel_values = torch.stack(images, dim=0)  # [B, C, H, W]
    cond_ids_tensor = torch.tensor(cond_ids, dtype=torch.long)

    # 텍스트 토크나이징
    tokenized = tokenizer(
        annotations,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # labels: padding 위치는 -100으로 마스킹하여 loss에서 제외
    labels = input_ids.masked_fill(attention_mask == 0, -100)

    return {
        "pixel_values": pixel_values,
        "cond_ids": cond_ids_tensor,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_paths": image_paths,
    }

