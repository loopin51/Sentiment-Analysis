# 감성 분석 모델 학습: GoEmotions 및 Financial PhraseBank

이 프로젝트는 Hugging Face Transformers 라이브러리를 활용하여 두 가지 다른 데이터셋에 대한 감성 분석 모델을 구축하고 평가하는 과정을 담고 있습니다.
각각의 작업은 별도의 Jupyter Notebook으로 구성되어 있습니다:

1.  **GoEmotions 데이터셋 기반 다중 감정 분류:** Reddit 댓글에서 27가지의 다양한 감정을 인식하는 다중 레이블 분류 모델을 학습합니다.
2.  **Financial PhraseBank 데이터셋 기반 금융 감성 분류:** 금융 뉴스 헤드라인의 감성을 긍정, 중립, 부정으로 분류하는 모델을 학습합니다.

## 🚀 프로젝트 목표

*   다양한 텍스트 데이터에 대한 감성 분류 모델 구축 능력 시연.
*   Hugging Face Transformers, `datasets`, `scikit-learn` 등 최신 NLP 라이브러리 활용 능력 제시.
*   일반적인 감정(GoEmotions)과 특정 도메인(금융)의 감정 분석 태스크 수행.
*   모델 학습, 평가, 결과 분석의 전체 파이프라인 이해 및 구현.

## 📂 파일 구성

*   `go_emotions_sentiment_analysis.ipynb`: GoEmotions 데이터셋을 사용한 다중 감정 분류 모델 학습 및 평가 노트북.
*   `financial_sentiment_analysis.ipynb`: Financial PhraseBank 데이터셋을 사용한 금융 감성 분류 모델 학습 및 평가 노트북.
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
*   **내용:** Reddit 댓글에 대해 27개의 감정 레이블 + `neutral`로 주석이 달린 데이터. 다중 레이블 분류 문제로, 하나의 댓글이 여러 감정을 가질 수 있습니다. (본 프로젝트에서는 27개 감정 + 'neutral'을 포함할 수 있으나, 'raw' 설정 사용 시 제공되는 27개 기본 감정 레이블을 사용합니다. `neutral`은 별도 처리나 포함 여부를 결정할 수 있습니다. 노트북에서는 27개 감정으로 처리합니다.)
*   **레이블 예시:** `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`. (실제 'raw' 버전은 27개 레이블을 포함)

### 1.2. 모델 아키텍처

*   **기본 모델:** `distilbert-base-uncased`
*   **선택 이유:** DistilBERT는 BERT의 경량화 버전으로, 더 적은 파라미터로 유사한 성능을 내며 학습 속도가 빠릅니다. 감성 분류와 같은 일반적인 NLP 작업에 효과적입니다.
*   **분류 방식:** 다중 레이블 분류 (Multi-label classification). 모델의 출력층은 각 레이블에 대한 로짓을 생성하며, 시그모이드 함수를 통과한 후 임계값을 기준으로 각 감정의 존재 유무를 판단합니다.

### 1.3. 학습 과정

1.  **데이터 로드:** `datasets` 라이브러리를 사용하여 `go_emotions` 데이터셋의 `raw` 구성을 로드합니다.
2.  **전처리 및 토큰화:**
    *   `AutoTokenizer`를 사용하여 텍스트를 토큰화합니다 (`distilbert-base-uncased`).
    *   레이블을 다중 레이블 형식에 맞게 변환합니다 (multi-hot encoding). 각 텍스트 샘플에 대해 27개의 감정 각각에 대해 해당 감정이 존재하면 1, 아니면 0으로 표시되는 벡터를 생성합니다.
3.  **모델 로드:** `AutoModelForSequenceClassification`을 사용하되, `problem_type="multi_label_classification"`으로 설정하고, 레이블 수(27개)와 ID-레이블명 매핑 정보를 전달합니다.
4.  **학습 설정 (`TrainingArguments`):**
    *   배치 크기, 학습률, 에포크 수 등 하이퍼파라미터 설정.
    *   평가 전략, 저장 전략, 로깅 설정.
    *   `fp16=True` (사용 가능한 경우 혼합 정밀도 학습).
5.  **메트릭 정의 (`compute_metrics`):**
    *   다중 레이블 분류에 적합한 F1-score (macro, micro, weighted), Subset Accuracy 등을 계산합니다.
    *   예측 시 로짓에 시그모이드 함수를 적용하고, 임계값(예: 0.5)을 기준으로 이진 예측을 수행합니다.
6.  **Trainer API 사용:** `Trainer` 객체에 모델, 학습 설정, 데이터셋, 토크나이저, 메트릭 함수를 전달하여 학습(`trainer.train()`)을 진행합니다.
7.  **평가:** 학습된 모델을 테스트 데이터셋으로 평가합니다.

### 1.4. 결과 및 분석 (예상)

*   테스트셋에 대한 F1-score (macro, micro), Subset Accuracy 등의 성능 지표를 제시합니다.
*   각 감정 레이블별로 개별적인 혼동 행렬(True Positive, False Positive, True Negative, False Negative)을 시각화하여 모델이 어떤 감정을 잘 분류하고 어떤 감정을 어려워하는지 분석합니다.
*   샘플 텍스트에 대한 실제 예측 결과를 보여주어 모델의 작동 방식을 설명합니다.

## 🤖 모델 2: Financial PhraseBank 금융 감성 분류

### 2.1. 데이터셋 설명

*   **데이터셋:** Financial PhraseBank
*   **출처:** [Malo, P., Sinha, A., Korhonen, A., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796.](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) (또는 Kaggle 등에서 재배포된 버전)
*   **내용:** 금융 뉴스 및 보고서에서 발췌한 문장들에 대해 긍정(positive), 중립(neutral), 부정(negative) 감성 레이블이 부착된 데이터셋. 주석자의 동의 수준에 따라 여러 버전이 존재합니다 (예: AllAgree, 50Agree 등).
*   **레이블:** `positive`, `neutral`, `negative`.

### 2.2. 모델 아키텍처

*   **기본 모델:** `distilbert-base-uncased`
*   **선택 이유:** GoEmotions와 동일하게 DistilBERT를 사용하여 효율적인 학습과 좋은 성능을 기대합니다. 금융 도메인 텍스트에도 일반화 성능이 좋을 것으로 예상됩니다.
*   **분류 방식:** 다중 클래스 분류 (Multi-class classification). 모델은 입력 텍스트가 세 가지 감성(긍정, 중립, 부정) 중 어느 것에 속하는지 예측합니다.

### 2.3. 학습 과정

1.  **데이터 로드:** `pandas`를 사용하여 `.txt` 파일 (예: `Sentences_50Agree.txt`)을 로드합니다. 텍스트와 감성 레이블을 분리합니다.
2.  **전처리 및 토큰화:**
    *   감성 레이블을 정수(0: positive, 1: neutral, 2: negative)로 인코딩합니다.
    *   데이터를 학습, 검증, 테스트 세트로 분할합니다 (`train_test_split`).
    *   `Dataset.from_pandas`를 사용하여 Hugging Face `Dataset` 객체로 변환합니다.
    *   `AutoTokenizer`를 사용하여 텍스트를 토큰화합니다.
3.  **모델 로드:** `AutoModelForSequenceClassification`을 사용하고, 레이블 수(3개)와 ID-레이블명 매핑 정보를 전달합니다. (기본 `problem_type`은 다중 클래스 분류)
4.  **학습 설정 (`TrainingArguments`):**
    *   GoEmotions와 유사하게 하이퍼파라미터 설정.
5.  **메트릭 정의 (`compute_metrics`):**
    *   정확도(Accuracy)와 F1-score (macro)를 주요 평가지표로 사용합니다.
6.  **Trainer API 사용:** `Trainer` 객체를 설정하고 학습(`trainer.train()`)을 진행합니다.
7.  **평가:** 학습된 모델을 테스트 데이터셋으로 평가합니다.

### 2.4. 결과 및 분석 (예상)

*   테스트셋에 대한 정확도, F1-score (macro) 등의 성능 지표를 제시합니다.
*   3x3 혼동 행렬을 시각화하여 모델이 각 감성을 얼마나 잘 구분하는지 분석합니다. (예: 중립적인 문장을 긍정/부정으로 오분류하는 경향 등)
*   샘플 금융 뉴스 헤드라인에 대한 실제 예측 결과를 보여줍니다.

## 💡 결론 및 향후 개선 방향

이 프로젝트는 두 가지 다른 감성 분석 문제를 해결하기 위한 트랜스포머 기반 모델의 적용 가능성을 보여줍니다.

**향후 개선 아이디어:**

*   **더 큰 사전 학습 모델 사용:** `BERT-base`, `BERT-large`, `RoBERTa` 등 더 큰 모델을 사용하여 성능 향상을 시도할 수 있습니다 (더 많은 컴퓨팅 자원 필요).
*   **하이퍼파라미터 튜닝:** Optuna, Ray Tune과 같은 라이브러리를 사용하여 최적의 하이퍼파라미터 조합을 탐색합니다.
*   **도메인 특화 모델 활용 (Financial PhraseBank):** 금융 도메인 텍스트로 사전 학습된 모델(예: `FinBERT`)을 사용하면 성능이 더 향상될 수 있습니다.
*   **데이터 증강:** 특히 레이블이 불균형하거나 데이터가 적은 경우, 데이터 증강 기법을 적용하여 모델의 일반화 성능을 높일 수 있습니다.
*   **오분류 분석 심층화:** 잘못 분류된 샘플들을 심층적으로 분석하여 모델의 약점을 파악하고 개선 전략을 수립합니다.
*   **임계값 조정 (GoEmotions):** 다중 레이블 분류에서 사용되는 시그모이드 출력에 대한 임계값을 조정하여 특정 레이블의 정밀도/재현율을 조절할 수 있습니다.
