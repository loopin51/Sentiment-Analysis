# 감성 분석 모델 학습: GoEmotions 및 Financial PhraseBank

이 프로젝트는 Hugging Face Transformers 라이브러리를 활용하여 두 가지 다른 데이터셋에 대한 감성 분석 모델을 구축하고 평가하는 과정을 담고 있습니다.
각각의 작업은 별도의 Jupyter Notebook으로 구성되어 있습니다:

1.  **GoEmotions 데이터셋 기반 다중 감정 분류:** Reddit 댓글에서 27가지의 다양한 감정을 인식하는 다중 레이블 분류 모델을 학습합니다.
2.  **Financial PhraseBank 데이터셋 기반 금융 감성 분류:** 금융 뉴스 헤드라인의 감성을 긍정, 중립, 부정으로 분류하는 모델을 학습합니다.

## 🚀 프로젝트 목표

*   다양한 텍스트 데이터에 대한 감성 분류 모델 구축 능력 시연.
*   Hugging Face Transformers, `datasets`, `scikit-learn` 등 최신 NLP 라이브러리 활용 능력 제시.
*   일반적인 감정(GoEmotions)과 특정 도메인(금융)의 감정 분석 태스크 수행.
*   모델 학습, 평가, 결과 분석 및 시각화의 전체 파이프라인 이해 및 구현.

## 📂 파일 구성

*   `go_emotions_sentiment_analysis.ipynb`: GoEmotions 데이터셋을 사용한 다중 감정 분류 모델 학습 및 평가 노트북. (데이터 분포 및 학습 과정 시각화 포함)
*   `financial_sentiment_analysis.ipynb`: Financial PhraseBank 데이터셋을 사용한 금융 감성 분류 모델 학습 및 평가 노트북. (데이터 분포 및 학습 과정 시각화 포함)
*   `README.md`: 이 프로젝트에 대한 설명 파일.

## 🛠️ 사용 기술

*   Python 3.x
*   Hugging Face Transformers (PyTorch 백엔드)
*   Hugging Face Datasets
*   Scikit-learn
*   Pandas
*   NumPy
*   Matplotlib & Seaborn (시각화)
*   Jupyter Notebook
*   TensorBoard (로깅 및 시각화에 활용 가능)

## ⚙️ 환경 설정 및 실행 방법

1.  **저장소 복제 (Clone the repository):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **필수 라이브러리 설치:**
    각 노트북 상단에 필요한 라이브러리를 설치하는 셀이 포함되어 있습니다. (`!pip install ...`)
    또는 다음 명령어로 주요 라이브러리를 미리 설치할 수 있습니다:
    ```bash
    pip install -q datasets transformers[torch] scikit-learn pandas numpy matplotlib seaborn tensorboard
    ```

3.  **데이터셋 준비:**
    *   **GoEmotions:** `datasets` 라이브러리를 통해 자동으로 다운로드됩니다. 별도의 준비 과정이 필요 없습니다.
    *   **Financial PhraseBank:**
        *   데이터셋은 일반적으로 `Sentences_AllAgree.txt`, `Sentences_75Agree.txt`, `Sentences_66Agree.txt`, `Sentences_50Agree.txt` 등의 파일로 제공됩니다.
        *   예를 들어, [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data) 등에서 "Sentiment Analysis for Financial News" 데이터셋을 다운로드 받을 수 있습니다.
        *   `financial_sentiment_analysis.ipynb` 노트북에서는 기본적으로 `Sentences_50Agree.txt` 파일을 현재 작업 디렉토리에서 찾도록 설정되어 있습니다. 다른 파일을 사용하거나 경로가 다른 경우 노트북 내의 `file_path` 변수를 수정해야 합니다.

4.  **Jupyter Notebook 실행:**
    ```bash
    jupyter notebook
    ```
    이후 브라우저에서 `go_emotions_sentiment_analysis.ipynb` 또는 `financial_sentiment_analysis.ipynb` 파일을 열어 실행합니다. Google Colab과 같은 클라우드 기반 노트북 환경 사용도 권장됩니다 (특히 GPU 사용 시).

##  modelu1: GoEmotions 다중 감정 분류

### 1.1. 데이터셋 설명

*   **데이터셋:** GoEmotions (raw 버전)
*   **출처:** [Hugging Face Datasets - go_emotions](https://huggingface.co/datasets/go_emotions)
*   **내용:** Reddit 댓글에 대해 27개의 감정 레이블로 주석이 달린 데이터. 다중 레이블 분류 문제로, 하나의 댓글이 여러 감정을 가질 수 있습니다.
*   **레이블 예시:** `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`.
*   **시각화:** 노트북에는 전체 데이터셋의 각 감정별 샘플 수 분포를 보여주는 막대 그래프가 포함되어 있습니다.

### 1.2. 모델 아키텍처

*   **기본 모델:** `distilbert-base-uncased`
*   **선택 이유:** DistilBERT는 BERT의 경량화 버전으로, 더 적은 파라미터로 유사한 성능을 내며 학습 속도가 빠릅니다. 감성 분류와 같은 일반적인 NLP 작업에 효과적입니다.
*   **분류 방식:** 다중 레이블 분류 (Multi-label classification). 모델의 출력층은 각 레이블에 대한 로짓을 생성하며, 시그모이드 함수를 통과한 후 임계값을 기준으로 각 감정의 존재 유무를 판단합니다.

### 1.3. 학습 과정

1.  **데이터 로드 및 전처리:** `datasets` 라이브러리 사용, 토큰화 및 multi-hot 인코딩 수행.
2.  **모델 로드:** `AutoModelForSequenceClassification` (`problem_type="multi_label_classification"`).
3.  **학습 설정 (`TrainingArguments`):** 배치 크기, 학습률, 에포크 등 설정. `logging_strategy="epoch"`로 설정하여 에포크별 로깅.
4.  **메트릭 정의 (`compute_metrics`):** F1-score (macro, micro, weighted), Subset Accuracy 등.
5.  **Trainer API 사용:** 모델 학습 및 평가.
6.  **시각화:**
    *   학습 중 Training/Validation Loss 변화 라인 그래프.
    *   Validation F1 Macro 및 Subset Accuracy 변화 라인 그래프.

### 1.4. 결과 및 분석

*   테스트셋에 대한 분류 보고서 및 주요 성능 지표 제시.
*   각 감정 레이블별 혼동 행렬 시각화.
*   샘플 텍스트에 대한 예측 결과 시연.

## 🤖 모델 2: Financial PhraseBank 금융 감성 분류

### 2.1. 데이터셋 설명

*   **데이터셋:** Financial PhraseBank
*   **출처:** [Malo, P., et al. (2014). Good debt or bad debt...](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)
*   **내용:** 금융 뉴스 문장에 긍정(positive), 중립(neutral), 부정(negative) 레이블 부착.
*   **레이블:** `positive`, `neutral`, `negative`.
*   **시각화:** 노트북에는 데이터셋의 긍정/중립/부정 감성 분포를 보여주는 파이 차트가 포함되어 있습니다.

### 2.2. 모델 아키텍처

*   **기본 모델:** `distilbert-base-uncased`
*   **분류 방식:** 다중 클래스 분류 (Multi-class classification).

### 2.3. 학습 과정

1.  **데이터 로드 및 전처리:** `pandas`로 파일 로드, 레이블 인코딩, 데이터 분할, Hugging Face `Dataset` 변환, 토큰화.
2.  **모델 로드:** `AutoModelForSequenceClassification`.
3.  **학습 설정 (`TrainingArguments`):** GoEmotions와 유사하게 설정, `logging_strategy="epoch"`.
4.  **메트릭 정의 (`compute_metrics`):** 정확도(Accuracy), F1-score (macro).
5.  **Trainer API 사용:** 모델 학습 및 평가.
6.  **시각화:**
    *   학습 중 Training/Validation Loss 변화 라인 그래프.
    *   Validation F1 Macro 및 Accuracy 변화 라인 그래프.

### 2.4. 결과 및 분석

*   테스트셋에 대한 분류 보고서 및 주요 성능 지표 제시.
*   3x3 혼동 행렬 시각화.
*   샘플 금융 뉴스 헤드라인에 대한 예측 결과 시연.

## 💡 결론 및 향후 개선 방향

이 프로젝트는 두 가지 다른 감성 분석 문제를 해결하기 위한 트랜스포머 기반 모델의 적용 가능성을 보여줍니다. 노트북에 포함된 데이터 및 학습 과정 시각화는 모델의 행동과 데이터셋의 특성을 이해하는 데 도움을 줍니다.

**향후 개선 아이디어:**

*   **더 큰 사전 학습 모델 사용:** `BERT-base`, `BERT-large`, `RoBERTa` 등.
*   **하이퍼파라미터 튜닝:** Optuna, Ray Tune 등 사용.
*   **도메인 특화 모델 활용 (Financial PhraseBank):** `FinBERT` 등.
*   **데이터 증강.**
*   **오분류 분석 심층화.**
*   **임계값 조정 (GoEmotions):** 다중 레이블 분류의 시그모이드 출력 임계값 조정.
---
