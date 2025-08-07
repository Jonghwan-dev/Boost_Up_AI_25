# **CYP3A4 효소 저해 예측 모델**

2025 신약개발 경진대회 'Boost up AI 2025' 출품작입니다. 이 프로젝트는 화합물의 2D 구조 정보(SMILES)로부터 인체 내 주요 약물 대사 효소인 CYP3A4의 저해율을 예측하는 머신러닝 파이프라인을 포함합니다.

---

## **핵심 전략**

본 모델은 **"더 풍부한 분자 표현형이 더 나은 예측을 만든다"** 는 가설 아래, 다음과 같은 전략을 채택했습니다.

1.  **다각적 피처 생성 (Rich Molecular Representation):**
    * 분자의 국소적 구조(Morgan FP), 약물유사체 패턴(Avalon FP), 물리화학적 특성(RDKit Descriptors), 그리고 사전 훈련된 GNN의 잠재 표현(GIN Pre-trained)을 **조합**하여 분자를 다각적으로 표현했습니다.
2.  **강력한 모델링 및 실험 기반 최적화:**
    * 고차원 데이터에 강하고 성능이 입증된 **CatBoost** 모델을 사용했습니다.
    * 실험을 통해 피처 선택이나 타겟 로그 변환과 같은 전처리 단계를 생략하고, 모델이 데이터의 복잡성을 직접 학습하도록 하는 것이 최적의 성능을 냄을 확인했습니다.

---

## **프로젝트 구조**

```
.
├── cache/                  # 피처 생성 시 캐시 파일이 저장되는 폴더 (자동 생성)
├── data/                   # 학습/테스트 데이터 폴더
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── submission/             # 결과 파일이 저장되는 폴더 (자동 생성)
├── features.py             # 분자 피처 생성 스크립트
├── main.py                 # 메인 실행 스크립트 (학습 및 예측)
├── utils.py                # 유틸리티 함수 (평가 지표, 스케일러)
└── requirements.txt        # 프로젝트 의존성 패키지 목록
```

---

## **설치 (Installation)**

이 가이드는 `pip`과 Python 가상환경을 사용하여 프로젝트 실행 환경을 구성하는 방법을 안내합니다.

### **1. 가상환경 생성 및 활성화**

프로젝트의 의존성을 독립적으로 관리하기 위해 새로운 가상환경을 생성합니다.

**On macOS / Linux:**
```bash
python3 -m venv boostup25
source boostup25/bin/activate
```

**On Windows:**
```bash
python -m venv boostup25
.\boostup25\Scripts\activate
```

### **2. 패키지 설치**

아래 `requirements.txt` 파일의 내용을 프로젝트 루트에 동일한 이름으로 저장한 후, 다음 명령어를 실행하여 모든 패키지를 설치합니다.

**설치 명령어:**
```bash
pip install -r requirements.txt
```
**⚠️ 중요:** `rdkit-pypi`는 일부 시스템에서 C++ 빌드 도구가 없어 설치에 실패할 수 있습니다. 오류 발생 시, Mamba/Conda를 통해 `rdkit`을 설치하는 것이 가장 안정적인 대안입니다.

---

## **실행 (How to Run)**

### **1. 데이터 준비**

`data` 폴더에 `train.csv`, `test.csv`, `sample_submission.csv` 파일을 위치시킵니다.

"update : data 폴더는 public data이므로 clone으로 간편하게 사용할 수 있도록 조치"

### **2. 모델 학습 및 예측**

터미널에서 `main.py` 스크립트를 실행합니다. 다양한 옵션을 통해 실험을 제어할 수 있습니다.

**기본 실행 (최고 성능 조합):**
* GIN 피처를 포함하고, 타겟 변환 및 피처 선택을 하지 않는 기본 설정입니다.
* CPU를 사용하여 학습합니다.
```bash
python main.py --use_gin_features --no_log_transform
```

**GPU를 사용하여 학습 속도 향상:**
* NVIDIA GPU가 설치된 환경에서 사용 가능합니다.
```bash
python main.py --use_gin_features --no_log_transform --use_gpu
```

**피처 캐시 재생성:**
* 피처를 처음부터 다시 생성하고 싶을 때 사용합니다.
```bash
python main.py --use_gin_features --no_log_transform --force_feature_regen
```

**주요 실행 옵션:**
* `--use_gin_features`: 사전 훈련된 GIN 피처를 포함합니다. (권장)
* `--no_log_transform`: 타겟 변수에 로그 변환을 적용하지 않습니다. (권장)
* `--feature_selection_method`: 피처 선택 방법을 지정합니다. (`none`, `variance`, `correlation`, `all`)
* `--use_gpu`: GPU를 사용하여 CatBoost 모델을 학습합니다.
* `--force_feature_regen`: 캐시를 무시하고 피처를 강제로 다시 생성합니다.
* 자세한 옵션은 `python main.py --help` 명령어로 확인할 수 있습니다.

### **3. 결과 확인**

실행이 완료되면 `submission` 폴더에 최종 예측 결과 파일(`Final_..._submission.csv`)이 생성됩니다.

---
## **데이터 출처**

본 연구에 사용된 데이터는 과학기술정보통신부의 후원으로 한국화학연구원, 한국생명공학연구원이 주최하고 한국화합물은행, 국가생명연구자원정보센터(KOBIC)가 주관한 '2025 신약개발 경진대회'를 통해 제공받았습니다. 

대회 운영을 맡아주신 데이콘에 감사드립니다.

---

## **라이선스 (License)**

이 프로젝트는 Apache License 2.0을 따릅니다. 자세한 내용은 [🎗️License](https://github.com/Jonghwan-dev/Boost_Up_AI_25/blob/main/LICENSE)를 참고하십시오.
