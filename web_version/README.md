<!-- Created: 2026-08-16 22:19 -->
# 손글씨 숫자 인식 - 웹 버전

브라우저에서 마우스나 손가락으로 숫자(0~9)를 그리면, 학습된 신경망 모델이 어떤 숫자인지 실시간으로 인식해주는 웹 애플리케이션입니다.

같은 프로젝트의 데스크톱(Tkinter) 버전은 [`../desktop_version`](../desktop_version)에 있습니다. 두 버전은 서로 독립적으로 개발되며 코드를 공유하지 않습니다.

## 주요 기능

- 캔버스에 숫자를 그리고 마우스(또는 손가락)를 떼면 **버튼 없이 자동으로 인식**
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 데이터셋으로 학습한 `scikit-learn` MLP 분류기 사용 (테스트 정확도 약 97.8%)
- 캔버스 이미지를 실제 MNIST 데이터 형식과 동일하게 전처리(잉크 영역 크롭 → 20x20 비율 유지 스케일 → 무게중심 기준 28x28 중앙 정렬)해서 정확도를 높임
- 데스크톱 앱 없이 브라우저만으로 동작

## 기술 스택

- **Backend**: Flask
- **모델 학습/추론**: scikit-learn, NumPy, Pillow, joblib
- **Frontend**: HTML5 Canvas + Vanilla JavaScript (별도 빌드 과정 없음)

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 모델 학습 (최초 1회, MNIST 데이터를 자동 다운로드하며 2~3분 소요)
#    digit_model.joblib / digit_scaler.joblib이 이미 있다면 생략 가능
python train_model.py

# 3. 서버 실행
python app.py
```

서버가 뜨면 브라우저에서 `http://127.0.0.1:5000` 으로 접속합니다.

## 폴더 구조

```
web_version/
├── app.py                 # Flask 서버 (/, /predict 라우트)
├── train_model.py         # MNIST 학습 스크립트
├── templates/
│   └── index.html         # 캔버스 UI (한글, 구글 스타일 디자인)
├── digit_model.joblib      # 학습된 분류기
├── digit_scaler.joblib     # 입력 정규화용 스케일러
├── mnist_data/             # 학습용 MNIST 원본 파일 캐시
└── requirements.txt
```

## 동작 원리

1. 브라우저 캔버스에 그린 숫자를 PNG로 캡처해 `/predict`로 전송합니다.
2. 서버(`app.py`)가 이미지를 받아 학습 데이터와 동일한 방식으로 전처리한 뒤, 저장된 모델로 예측합니다.
3. 예측한 숫자와 신뢰도(%)를 JSON으로 응답받아 화면에 표시합니다.
