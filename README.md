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
├── cache/                  # 피처 캐시 폴더 (자동 생성)
├── data/                   # 데이터 폴더
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/                 # 모델 가중치 폴더 (train.py 실행 시 자동 생성)
├── submission/             # 결과 파일 폴더 (predict.py 또는 main.py 실행 시 자동 생성)
├── features.py             # 분자 피처 생성 스크립트
├── utils.py                # 유틸리티 함수 (평가 지표, 스케일러)
├── train.py                # 모델 학습 및 가중치 저장 스크립트
├── predict.py              # 저장된 가중치로 추론 및 제출 파일 생성 스크립트
├── main.py                 # 학습과 추론을 한번에 실행하는 통합 스크립트
└── requirements.txt        # 프로젝트 의존성 패키지 목록
```

---

## **개발 및 실행 환경**

본 프로젝트는 아래 환경에서 개발 및 테스트되었습니다.

* **OS:** Ubuntu 22.04 LTS
* **Python:** 3.10.18 (Mamba로 구성)
* **GPU:** NVIDIA GeForce GTX 5090
* **CUDA Version:** 12.8 nightly

---

## **설치 (Installation)**

이 가이드는 `pip`과 Python 가상환경을 사용하여 프로젝트 실행 환경을 구성하는 방법을 안내합니다.

### **1. 가상환경 생성 및 활성화**

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

### **2. GPU 사용자 PyTorch/DGL 설치 (권장)**

`molfeat[dgl]`은 PyTorch에 의존합니다. GPU 가속을 사용하려면 시스템의 CUDA 버전에 맞는 PyTorch와 DGL을 먼저 설치해야 합니다.

**본 개발 환경(CUDA 12.8)에서는 아래 명령어를 사용했습니다:**
```bash
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
pip3 install dgl -f [https://data.dgl.ai/wheels/cu128/repo.html](https://data.dgl.ai/wheels/cu128/repo.html)
```
**⚠️ 중요:** 사용자의 GPU 또는 CUDA 버전이 다른 경우, [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)와 [DGL 공식 홈페이지](https://www.dgl.ai/pages/start.html)에서 자신의 환경에 맞는 설치 명령어를 확인하여 실행하세요

### **3. 나머지 패키지 설치**

`requirements.txt` 파일을 사용하여 나머지 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```
**⚠️ 참고:** `rdkit-pypi`는 일부 시스템에서 C++ 빌드 도구가 없어 설치에 실패할 수 있습니다. 오류 발생 시, Mamba/Conda를 통해 `rdkit`을 설치하는 것이 가장 안정적인 대안입니다.

---

## **실행 (How to Run)**

실행은 **학습**과 **추론**의 두 단계로 나뉩니다.

### **1. 데이터 준비**

`data` 폴더에 `train.csv`, `test.csv`, `sample_submission.csv` 파일을 위치시킵니다.
(참고: `data` 폴더는 public 데이터이므로, 이 저장소를 clone하면 바로 사용할 수 있습니다.)

### **2. 모델 학습 (`train.py`)**

`train.py`는 K-Fold 교차 검증으로 모델을 학습시키고, 각 Fold의 모델 가중치를 `models/` 폴더에 저장합니다.

**최고 성능 옵션으로 학습 실행:**
```bash
python train.py --use_gin_features --no_log_transform
```

**GPU를 사용하여 학습 속도 향상:**
```bash
python train.py --use_gin_features --no_log_transform --use_gpu
```

### **3. 추론 및 제출 파일 생성 (`predict.py`)**

`predict.py`는 `models/` 폴더에 저장된 가중치들을 불러와 테스트 데이터에 대한 예측을 수행하고, 최종 `submission.csv` 파일을 `submission/` 폴더에 생성합니다.

**저장된 모델로 추론 실행:**
```bash
python predict.py --use_gin_features
```
* **주의:** `predict.py`의 옵션(`--use_gin_features` 등)은 `train.py` 실행 시 사용했던 옵션과 반드시 일치해야 합니다.

---
## **데이터 출처 및 謝辭**

본 연구에 사용된 데이터는 과학기술정보통신부의 후원으로 한국화학연구원, 한국생명공학연구원이 주최하고 한국화합물은행, 국가생명연구자원정보센터(KOBIC)가 주관한 '2025 신약개발 경진대회'를 통해 제공받았습니다. 대회 운영을 맡아주신 데이콘에 감사드립니다.

---

## **참고 문헌 (References)**

1.  Stavropoulou, E., et al. (2018). The Role of Cytochromes P450 in Infection. *Frontiers in Immunology*. https://doi.org/10.3389/fimmu.2018.00089
2.  Zhang, L., et al. (2021). Nanoparticulate Drug Delivery Strategies to Address Intestinal Cytochrome P450 CYP3A4 Metabolism towards Personalized Medicine. *Pharmaceutics*. https://doi.org/10.3390/pharmaceutics13081261
3.  Notwell, J. H., & Wood, M. W. (2023). ADMET property prediction through combinations of molecular fingerprints. *arXiv*. https://doi.org/10.48550/arXiv.2310.00174
4.  Fabian, B., et al. (2020). Molecular representation learning with language models and domain-relevant auxiliary tasks. *arXiv*. https://doi.org/10.48550/arXiv.2011.13230
5.  Hu, W., et al. (2020). STRATEGIES FOR PRE-TRAINING GRAPH NEURAL NETWORKS. *ICLR 2020*. https://doi.org/10.48550/arXiv.1905.12265
6.  Bento, A. P. G., et al. (2020). An open source chemical structure curation pipeline using RDKit. *Journal of Cheminformatics*. https://doi.org/10.1186/s13321-020-00456-1
7.  Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 2018*. https://doi.org/10.48550/arXiv.1706.09516
8.  Prusty, K. B., et al. (2022). SKCV: Stratified K-fold cross-validation on ML classifiers for predicting cervical cancer. *Biomedical Nanotechnology*. https://doi.org/10.3389/fnano.2022.972421

---

## **라이선스 (License)**

이 프로젝트는 Apache License 2.0을 따릅니다. 자세한 내용은 [LICENSE](https://github.com/Jonghwan-dev/Boost_Up_AI_25/blob/main/LICENSE) 파일을 참고하십시오.