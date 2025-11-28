"""
Multi-task WeatherQA Stage2 training package.

이 패키지는 WeatherQA + Gemini 요소 캡션 데이터셋을 활용해
다양한 태스크(요소 캡션 재구성, MD-style summary 생성, 요소-요약 연결 CoT 등)를
하나의 Stage1 encoder + LLM(LoRA)로 함께 학습하기 위한 코드 모음이다.
"""

