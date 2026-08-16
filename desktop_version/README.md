<!-- Created: 2026-08-16 22:26 -->
# 손글씨 숫자 인식 - 데스크톱 버전

Tkinter로 만든 데스크톱 GUI에서 마우스로 숫자(0~9)를 그리면, 학습된 신경망 모델이 어떤 숫자인지 실시간으로 인식해주는 애플리케이션입니다.

같은 프로젝트의 웹 버전은 [`../web_version`](../web_version)에 있습니다. 두 버전은 서로 독립적으로 개발되며 코드를 공유하지 않습니다.

## 주요 기능

- 캔버스에 숫자를 그리고 마우스를 떼면 **버튼 없이 자동으로 인식**
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 데이터셋으로 학습한 `scikit-learn` MLP 분류기 사용 (테스트 정확도 약 97.8%)
- 캔버스 이미지를 실제 MNIST 데이터 형식과 동일하게 전처리(잉크 영역 크롭 → 20x20 비율 유지 스케일 → 무게중심 기준 28x28 중앙 정렬)해서 정확도를 높임
- Windows에서 `Run_Digit_Recognizer.bat`을 더블클릭하면 콘솔 창 없이 바로 실행

## 실행 방법

```bash
# 1. 의존성 설치
python -m pip install numpy pillow scikit-learn joblib

# 2. 모델 학습 (최초 1회, MNIST 데이터를 자동 다운로드하며 2~3분 소요)
#    digit_model.joblib / digit_scaler.joblib이 이미 있다면 생략 가능
python train_model.py

# 3. GUI 실행
python draw_and_recognize.py
```

Windows에서는 `Run_Digit_Recognizer.bat`을 더블클릭해서 실행할 수도 있습니다 (모델 파일이 없으면 안내 메시지를 띄웁니다).

## 폴더 구조

```
desktop_version/
├── draw_and_recognize.py   # Tkinter GUI 앱
├── train_model.py          # MNIST 학습 스크립트
├── Run_Digit_Recognizer.bat # Windows 더블클릭 실행 파일
├── digit_model.joblib       # 학습된 분류기
├── digit_scaler.joblib      # 입력 정규화용 스케일러
└── mnist_data/               # 학습용 MNIST 원본 파일 캐시
```

## 동작 원리

1. 캔버스(280x280, MNIST 원본 28x28의 10배 크기)에 그린 숫자를 마우스를 떼는 순간 캡처합니다.
2. 학습 데이터와 동일한 방식으로 전처리한 뒤, 저장된 모델로 예측합니다.
3. 예측한 숫자와 신뢰도(%)를 화면에 바로 표시합니다.
