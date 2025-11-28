Multi-task WeatherQA Stage2 (Image → Text) Training
===================================================

개요
----

이 디렉토리는 WeatherQA + Gemini 요소 캡션 데이터셋을 활용해,
여러 태스크를 하나의 Stage1 이미지 인코더 + LLM(Mistral-7B + LoRA)로
동시에 학습하기 위한 Multi-task Stage2 파이프라인이다.

Stage1:
  - Patch Perceiver 기반 vision encoder (latent_dim=768)
  - EfficientNet-B0 classifier (WeatherQA cond 20 클래스)

Stage2:
  - Mistral-7B (또는 호환 HF causal LLM)
  - Stage1 latent + cond id를 LLM prefix 토큰으로 주입
  - LoRA로 경량 미세조정

데이터 구조
-----------

현재 MTL 파이프라인은 주로 다음 두 종류의 JSON 포맷을 대상으로 한다.

1. `weatherqa_gemini_batch.json`

   - 각 case:

     - `case_id`
     - `global_summary`: 해당 사례 전체에 대한 사람/MD 스타일 요약 텍스트
     - `elements`: 요소 리스트

   - 각 element:

     - `image_rel_path`: 요소 이미지 경로 (루트 기준 상대 경로)
     - `type_name`: 요소 타입 (예: `thea`, `mcon`, `ttd`)
     - `caption`: Gemini가 생성한 4문장 reasoning 텍스트
       - Sentence 1: color fill/contours가 나타내는 변수 설명
       - Sentence 2: 지배적인 시각 패턴, 위치, 바람 흐름
       - Sentence 3: 물리적 의미 (예: moisture pooling, instability 등)
       - Sentence 4: Global Summary와의 연결
     - `error`: 캡션 생성에 문제가 있으면 값이 채워지고, Dataset에서는 `error is None`인 것만 사용

2. (선택) summary-only JSON

   - 기존 Stage2 summary-only 학습에서 사용하던 포맷
   - 각 item: `{ "image": ..., "annotation": ..., "cond_name": ... }`
   - 이 JSON이 없더라도, MTL 파이프라인은 `weatherqa_gemini_batch.json`의 `global_summary`를
     각 요소 이미지와 pairing하여 (image, summary) 데이터를 자동으로 구성한다.

지원 태스크
-----------

현재 4가지 태스크를 지원한다:

1. Task A: Element Caption (요소 캡션 재구성)

   - 데이터: `weatherqa_gemini_batch.json`의 각 element
   - 입력: 요소 이미지
   - 프롬프트:

     - "4문장으로 color fill/contours, 지배적 패턴, 물리 의미, broader context를 설명하라"는
       내용의 고정 프롬프트

   - 타겟: Gemini가 생성한 `caption` 전체 (4 sentences)
   - 목적:

     - Stage1 latent + LLM이 기상 요소 이미지를 도메인 용어로 잘 설명하도록 align

2. Task B: MD-style Summary (이미지 → 요약)

   - 데이터:

     - Gemini JSON만 사용하며, 각 case의 `global_summary`를 모든 element 이미지(`image_rel_path`)와 pairing한 (image, summary) 집합을 구성한다.

   - 입력: WeatherQA 요소/MD 이미지
   - 프롬프트:

     - SPC Mesoscale Discussion 스타일의 summary를 쓰라는 간단한 인스트럭션

   - 타겟:

     - `global_summary` (case별로 동일 summary가 여러 요소 이미지와 페어링됨)
   - 목적:

     - 최종 목표인 "이미지 → summary description" 태스크 자체를 학습

3. Task C: Element–Summary Link (요소 ↔ Global Summary 연결 설명)

   - 데이터: `weatherqa_gemini_batch.json`의 각 case의 요소와 `global_summary`
   - 입력:

     - 요소 이미지
     - 프롬프트에 element caption 전체와 global_summary를 포함:

       - Element description: {caption}
       - Forecast summary: {global_summary}
       - Q: 이 요소가 summary를 어떻게 support/refine/contradict하는지 1–2문장으로 설명

   - 타겟:

     - caption의 마지막 문장(S4)을 "summary와의 연결/설명" 문장으로 간주해 사용

   - 목적:

     - LLM이 "local 요소 ↔ 전체 요약" 사이의 관계를 텍스트로 풀어내는 패턴 학습

4. Task D (옵션): Text Summary from Captions (여러 caption → summary)

   - 데이터: `weatherqa_gemini_batch.json`
   - 입력:

     - 한 case 내 여러 요소 caption의 리스트
     - 프롬프트에 numbered captions를 넣고, SPC 스타일 summary를 쓰도록 지시
     - 대표 이미지(첫 요소 이미지)를 image_path로 사용 (Stage1 prefix는 크게 중요치 않음)

   - 타겟:

     - `global_summary`

   - 목적:

     - 여러 요소 정보로부터 전체 forecast summary를 구성하는 패턴 학습
     - text-only 요약에 가까운 보조 태스크

구현 구조
---------

파일 구성:

- `datasets.py`

  - `MTLSample`: task/prompt/target/image_path를 담는 dataclass
  - `ElementCaptionDataset`: Task A용 Dataset
  - `SummaryFromImageDataset`: Task B용 Dataset
  - `ElementSummaryLinkDataset`: Task C용 Dataset
  - `CaseSummaryFromCaptionsDataset`: Task D용 Dataset (옵션)
  - `mtl_collate_fn`: prompt + target을 분리 토큰화해 prompt 구간 label을 -100으로 마스킹
  - `build_mtl_dataset_and_loader`: 여러 Dataset을 합쳐 train/val DataLoader 생성

- `train_mtl_weatherqa.py`

  - 공통 Stage1/Stage2 구조:

    - `Stage1ImageEncoderForStage2`:

      - Stage1 Perceiver encoder + EfficientNet classifier를 래핑
      - `encode_and_classify(images) -> (latents, cond_ids_stage1)`
      - `stage1-trainable-blocks` 인자로 마지막 K 블록만 finetune 가능

    - `ImageToTextModelStage1`:

      - 이미지 latent + Stage1 cond id를 LLM prefix 토큰으로 변환해 주입
      - 텍스트 부분은 HF `AutoModelForCausalLM` (예: Mistral-7B) 사용
      - 학습 시 prefix 구간 label은 -100으로 마스킹

  - multi-task 학습:

    - 하나의 DataLoader에서 Task A/B/C/D 샘플이 섞여 들어옴
    - 모든 태스크가 같은 Stage1 encoder + LLM(LoRA)을 공유
    - optimizer에서 Stage1 encoder와 나머지(LLM/Stage2)를 다른 lr로 설정 가능

  - wandb 로깅, checkpoint 저장:

    - `model.llm.save_pretrained(output_dir)`
    - 토크나이저 저장
    - `stage2_prefix_modules.pt` (class embedding + image_projection)
    - Stage1 encoder를 finetune한 경우 `stage1_encoder_finetuned.pt`

사용 예시
---------

1) 기본 Multi-task 학습 (Task A+B+C, Task D 비활성화)

```bash
cd /home/agi592
conda activate tw-stage2

CUDA_VISIBLE_DEVICES=0 python -m twkim.climate_to_text_stage2.mtl_weatherqa.train_mtl_weatherqa \
  --gemini-json kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
                kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --model-path /home/agi592/models/mistral-7b \
  --output-dir twkim/checkpoints/stage2_mtl_weatherqa_ep1 \
  --batch-size 1 \
  --num-epochs 1 \
  --lr 5e-5 \
  --encoder-lr 5e-6 \
  --warmup-ratio 0.1 \
  --max-seq-len 512 \
  --image-size 224 \
  --use-lora \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --num-image-tokens 4 \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_mtl_weatherqa_three_patch/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --stage1-trainable-blocks 1 \
  --use-wandb \
  --wandb-project climate-to-text-stage2-mtl \
  --wandb-run-name "stage2_mtl_weatherqa_ep1"
```

- `--encoder-lr`: Stage1 encoder 뒷단 학습률 (LLM lr보다 작게 설정하는 것을 권장)
- `--stage1-trainable-blocks`: Stage1 Perceiver 마지막 몇 개 block을 열지 결정 (0이면 완전 동결)

2) Task 구성 조절

- caption 태스크만 끄고 싶을 때:

```bash
... train_mtl_weatherqa.py \
  --no-caption \
  --summary-json ... \
  ...
```

- summary 태스크를 빼고 CoT 중심으로만 돌릴 때:

```bash
... train_mtl_weatherqa.py \
  --no-summary \
  ...
```

- text_summary(Task D)를 추가로 켜고 싶을 때:

```bash
... train_mtl_weatherqa.py \
  --text-summary \
  ...
```

3) 분산(DDP) 학습 예시 (torchrun)

```bash
cd /home/agi592
conda activate tw-stage2

torchrun --nproc_per_node=2 twkim/climate_to_text_stage2/mtl_weatherqa/train_mtl_weatherqa.py \
  --gemini-json kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
                kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --model-path /home/agi592/models/mistral-7b \
  --output-dir twkim/checkpoints/stage2_mtl_weatherqa_ep1_ddp \
  --batch-size 1 \
  --num-epochs 1 \
  --lr 5e-5 \
  --encoder-lr 5e-6 \
  --warmup-ratio 0.1 \
  --max-seq-len 512 \
  --image-size 224 \
  --use-lora \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --num-image-tokens 4 \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_mtl_weatherqa_three_patch/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --stage1-trainable-blocks 1 \
  --distributed
```

이 설정에서는:

- Task A: 요소 이미지 → Gemini caption (4문장)
- Task B: 요소 이미지 → case의 global_summary
- Task C: caption + global_summary → 연결/설명
- Task D: `--text-summary`를 추가로 켜면 caption list → global_summary 텍스트 요약까지 함께 학습된다.

주의사항 / 팁
-------------

- GPU 메모리:

  - Mistral-7B + Stage1 encoder + LoRA + multi-task는 24GB급 GPU에서 돌아갈 수 있도록
    fp16/bf16, gradient clipping 등을 포함하고 있음.
  - 그래도 여유가 부족하다면 `max-seq-len`을 낮추거나, batch size를 1로 유지하는 것을 권장.

- Stage1 finetune:

  - `--stage1-trainable-blocks=0`부터 시작해 전체 파이프라인이 안정적으로 돌아가는지 확인한 뒤,
    1 또는 2 블록만 여는 방식으로 점진적으로 늘리는 것이 좋다.
  - `--encoder-lr`는 LLM lr보다 5~10배 정도 작게 잡는 것이 안전하다 (예: LLM lr=1e-5, encoder lr=5e-6).

- 태스크 비율:

  - 현재 구현은 여러 Dataset을 Concat하고 random_split만 수행하므로, 태스크별 비율은
    원본 데이터셋 크기에 따라 결정된다.
  - 필요하다면 Dataset을 subsample하거나, 별도 샘플링 로직을 추가해 summary 태스크 비중을
    조금 더 높게 가져가는 것도 가능하다.

이 디렉토리의 코드를 바탕으로, WeatherQA + Gemini 요소 캡션 데이터셋을 활용한
이미지→텍스트 multi-task 학습 실험을 손쉽게 수행할 수 있다.
