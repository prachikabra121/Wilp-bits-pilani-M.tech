# Credit Default Risk Prediction - Comparison of Five Classifiers

## a. Problem Statement
Credit card lenders need to know in advance which clients are likely to miss their next month's payment, because an undetected default means a direct financial loss. This project treats the task as a binary classification problem: given a client's credit limit, demographics, six months of repayment status, bill amounts and previous payments, predict whether the client will default next month (1) or not (0). Five classification models are trained on the same dataset: Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine. They are compared using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). Because only about 22% of clients default, accuracy alone is misleading, so recall and MCC are given extra weight when choosing the best model. The trained models are deployed in an interactive Streamlit app where a test CSV can be uploaded to see each model's metrics and confusion matrix.

## b. Dataset Description 
- **Source:** Default of Credit Card Clients dataset (UCI Machine Learning Repository; CSV copy from Kaggle) - https://www.kaggle.com/datasets/hassanamin/uci-credit-card?resource=download
- **Size:** 30,000 clients, 23 input features + 1 target (an `ID` column was dropped). Requirement: at least 12 features and 500 instances.
- **Subsample used:** 10,000 randomly sampled rows (random_state=42) to keep SVM training and the Streamlit free tier manageable; split 80/20 stratified into 8,000 train and 2,000 test rows.
- **Target:** `target` - 1 = client defaults next month, 0 = does not default. About 22% of clients default, so the classes are imbalanced.
- **Features:** credit limit (`LIMIT_BAL`), demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`), six months of repayment status (`PAY_0`, `PAY_2`-`PAY_6`), six months of bill amounts (`BILL_AMT1`-`6`) and six months of previous payments (`PAY_AMT1`-`6`).
- **Preprocessing:** no missing values; numeric columns scaled with StandardScaler inside each model pipeline (so the saved models include their preprocessing); target label-encoded.

## c. GitHub Repository 
https://github.com/prachikabra121/Wilp-bits-pilani-M.tech

## Live Streamlit app:
https://wilp-bits-pilani-mtech-ml.streamlit.app/

```
project-folder/
|-- app.py
|-- evaluation.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- data/dataset.csv  #raw data
|-- test_data.csv
|-- model_comparison.csv
|-- .streamlit/config.toml
|-- model/   (logistic_regression, decision_tree, knn, naive_bayes,
|             support_vector_machine .joblib + label_encoder.joblib + meta.json)
```

## d. Models Used  

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

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | With balanced class weights it catches most defaulters (recall 0.642), but it flags many good clients as well, so precision is low (0.371) and accuracy is the lowest of all models (0.685). It also has the lowest MCC (0.289), which suggests a straight-line decision boundary is too simple for this data. |
| Decision Tree | It has the best MCC (0.380) with a good balance between precision (0.504) and recall (0.534). Limiting max_depth to 5 and min_samples_leaf to 20 keeps the tree from overfitting, and it stays easy to explain because each prediction follows a small set of rules. |
| KNN | It has the highest accuracy (0.806) and precision (0.617), but its recall is only 0.291, so it misses most real defaulters. About 78% of clients do not default, so a model that leans towards the majority class gets high accuracy without being useful. KNN has no class weight option, which explains this. |
| Naive Bayes | It finds many defaulters (recall 0.640) but with low precision (0.418), and its accuracy is low (0.728). The features such as BILL_AMT and PAY columns are strongly correlated, which breaks the independence assumption of Naive Bayes and pushes it to predict default too often. |
| Support Vector Machine | It has the best AUC (0.758) and F1 (0.523), and its MCC (0.377) is almost the same as the Decision Tree. The RBF kernel handles non-linear patterns well, but training and probability calculation take the longest of all five models. |
| **Overall winner** | Decision Tree is the best model for this dataset. It has the highest MCC (0.380), balances precision and recall, and is the easiest to interpret. SVM is a very close second and ranks clients best (highest AUC), but it is slower and harder to explain. KNN's high accuracy is misleading because of the class imbalance. |

## How to run
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_models.py        # trains, saves models to model/, writes test_data.csv
streamlit run app.py
```
