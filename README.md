# Credit Default Risk Prediction - Comparison of Five Classifiers

## a. Problem Statement
<WRITE IN YOUR OWN WORDS, 3-4 sentences: predict whether a credit-card client will default on next month's payment (binary classification), why missing a defaulter is costly in credit-risk work, and that you compare Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM using Accuracy, AUC, Precision, Recall, F1 and MCC, deployed as a Streamlit app.>

## b. Dataset Description  [1 mark]
- **Source:** Default of Credit Card Clients dataset (UCI Machine Learning Repository; CSV copy from Kaggle) - <add the link you downloaded it from>
- **Size:** 30,000 clients, 23 input features + 1 target (an `ID` column was dropped). Requirement: at least 12 features and 500 instances.
- **Subsample used:** 10,000 randomly sampled rows (random_state=42) to keep SVM training and the Streamlit free tier manageable; split 80/20 stratified into 8,000 train and 2,000 test rows.
- **Target:** `target` - 1 = client defaults next month, 0 = does not default. About 22% of clients default, so the classes are imbalanced.
- **Features:** credit limit (`LIMIT_BAL`), demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`), six months of repayment status (`PAY_0`, `PAY_2`-`PAY_6`), six months of bill amounts (`BILL_AMT1`-`6`) and six months of previous payments (`PAY_AMT1`-`6`).
- **Preprocessing:** no missing values; numeric columns scaled with StandardScaler inside each model pipeline (so the saved models include their preprocessing); target label-encoded.

## c. GitHub Repository  [1 mark]
<PASTE YOUR GITHUB REPO LINK>

Live Streamlit app: <PASTE YOUR STREAMLIT LINK>

```
project-folder/
|-- app.py
|-- evaluation.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_comparison.csv
|-- .streamlit/config.toml
|-- model/   (logistic_regression, decision_tree, knn, naive_bayes,
|             support_vector_machine .joblib + label_encoder.joblib + meta.json)
```

## d. Models Used  [5 marks]

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6850 | 0.7208 | 0.3714 | 0.6422 | 0.4706 | 0.2889 |
| Decision Tree | 0.7840 | 0.7502 | 0.5043 | 0.5344 | 0.5189 | 0.3801 |
| KNN | 0.8060 | 0.7357 | 0.6165 | 0.2913 | 0.3956 | 0.3271 |
| Naive Bayes | 0.7275 | 0.7299 | 0.4183 | 0.6399 | 0.5059 | 0.3431 |
| Support Vector Machine | 0.7705 | 0.7583 | 0.4782 | 0.5780 | 0.5234 | 0.3769 |

### Output of train_models.py (run on BITS Virtual Lab)

```
saved model/logistic_regression.joblib
saved model/decision_tree.joblib
saved model/knn.joblib
saved model/naive_bayes.joblib
saved model/support_vector_machine.joblib

 | ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.685  | 0.7208 |      0.3714 |   0.6422 | 0.4706 | 0.2889 |
| Decision Tree          |     0.784  | 0.7502 |      0.5043 |   0.5344 | 0.5189 | 0.3801 |
| KNN                    |     0.806  | 0.7357 |      0.6165 |   0.2913 | 0.3956 | 0.3271 |
| Naive Bayes            |     0.7275 | 0.7299 |      0.4183 |   0.6399 | 0.5059 | 0.3431 |
| Support Vector Machine |     0.7705 | 0.7583 |      0.4782 |   0.578  | 0.5234 | 0.3769 |
```

Settings used: Logistic Regression (C=0.5, balanced class weights), Decision Tree (max_depth=5, min_samples_leaf=20, balanced), KNN (k=15, distance-weighted), Gaussian Naive Bayes (var_smoothing=1e-8), SVM (RBF, C=2.0, balanced).

### Observations on model performance  [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <YOUR WORDS: high recall (0.64) but low precision (0.37) and lowest MCC; linear boundary too simple?> |
| Decision Tree | <YOUR WORDS: best MCC (0.380), balanced precision/recall, depth limit controls overfitting, interpretable> |
| KNN | <YOUR WORDS: highest accuracy but recall only 0.29 - accuracy misleading with ~78% non-defaulters; no class weighting> |
| Naive Bayes | <YOUR WORDS: recall 0.64 but precision 0.42, lowest accuracy; independence assumption broken by correlated BILL_AMT/PAY features> |
| Support Vector Machine | <YOUR WORDS: best AUC and F1, MCC almost equal to tree, non-linear RBF boundary, slowest to train> |
| **Overall winner** | <YOUR CHOICE + reason: Decision Tree (0.380 MCC, interpretable) or SVM (best AUC/F1) - justify with your priority> |

## How to run
```
pip install -r requirements.txt
python train_models.py        # trains, saves models to model/, writes test_data.csv
streamlit run app.py
```
