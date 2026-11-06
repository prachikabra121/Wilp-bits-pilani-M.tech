# Trains five classification models on the credit card default dataset,
# saves them in the model/ folder and creates test_data.csv for the app.
# Run with:  python train_models.py
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from evaluation import compute_metrics

# ---- settings ----
DATA_PATH = "data/dataset.csv"   # raw dataset downloaded from Kaggle / UCI
TARGET_COL = "target"            # 1 = client defaulted, 0 = did not default
DROP_COLS = ["ID"]               # ID is just a row number, not useful for prediction
MAX_ROWS = 10000                 # use a sample so that SVM trains quickly
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs("model", exist_ok=True)

# ---- load and prepare data ----
df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
df = df.dropna().reset_index(drop=True)

if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

X = df.drop(columns=[TARGET_COL])
label_enc = LabelEncoder()
y = label_enc.fit_transform(df[TARGET_COL])

# stratify keeps the same default ratio in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()


def make_preprocessor():
    # scale numeric columns, one-hot encode text columns (none in this dataset)
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ---- the five models ----
models = {
    "Logistic Regression": LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                            class_weight="balanced", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "Support Vector Machine": SVC(kernel="rbf", C=2.0, gamma="scale",
                                  class_weight="balanced", probability=True,
                                  random_state=RANDOM_STATE),
}

# ---- train, evaluate and save each model ----
rows = []
for name, clf in models.items():
    # pipeline = preprocessing + model, so the app only has to load one file
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    rows.append({"ML Model Name": name, **compute_metrics(y_test, pred, proba)})

    file_name = "model/" + name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, file_name)
    print("saved", file_name)

joblib.dump(label_enc, "model/label_encoder.joblib")
with open("model/meta.json", "w") as f:
    json.dump({"target": TARGET_COL, "features": X.columns.tolist()}, f)

# ---- test data for the Streamlit app (features + target column) ----
test_df = X_test.copy()
test_df[TARGET_COL] = label_enc.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

# ---- comparison table (copied into the README) ----
results = pd.DataFrame(rows).round(4)
results.to_csv("model_comparison.csv", index=False)
print("\n", results.to_markdown(index=False))
