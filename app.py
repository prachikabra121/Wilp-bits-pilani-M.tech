# Streamlit app to evaluate the saved models on an uploaded test file.
# The models are only loaded from the model/ folder, they are never retrained.
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from evaluation import compute_metrics

st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# small bit of styling for the header and the metric boxes
st.markdown("""
<style>
.header-box {background:#1f4e79; padding:22px 28px; border-radius:8px; margin-bottom:18px;}
.header-box h1 {color:#ffffff; margin:0; font-size:30px;}
.header-box p {color:#dbe7f3; margin:6px 0 0 0; font-size:15px;}
div[data-testid="stMetric"] {background:#f1f5f9; border-left:4px solid #1f4e79;
                             padding:10px 14px; border-radius:6px;}
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

METRIC_HELP = {
    "Accuracy": "Share of all predictions that are correct.",
    "AUC": "How well the model ranks defaulters above non-defaulters (0.5 is random).",
    "Precision": "Out of clients predicted as default, how many really defaulted.",
    "Recall": "Out of all real defaulters, how many were found.",
    "F1": "Balance between precision and recall.",
    "MCC": "Balanced score that stays reliable when classes are imbalanced.",
}


@st.cache_resource
def load_files():
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    encoder = joblib.load("model/label_encoder.joblib")
    with open("model/meta.json") as f:
        meta = json.load(f)
    return models, encoder, meta


models, encoder, meta = load_files()
target = meta["target"]

st.markdown("""
<div class="header-box">
  <h1>Credit Card Default Prediction</h1>
  <p>Compare five classification models on your test data</p>
</div>
""", unsafe_allow_html=True)

# ---- sidebar: file upload and model selection ----
with st.sidebar:
    st.header("Options")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.caption(f"The file must have {len(meta['features'])} feature columns "
               f"and the target column '{target}' (1 = default).")
    st.caption("You can use test_data.csv from the GitHub repository.")

if uploaded is None:
    st.info("Upload a test CSV file from the sidebar to see the results.")
    st.stop()

df = pd.read_csv(uploaded)

# check that all columns the models need are present
missing = [c for c in meta["features"] + [target] if c not in df.columns]
if missing:
    st.error(f"These columns are missing in the uploaded file: {missing}")
    st.stop()

X = df[meta["features"]]
try:
    y = encoder.transform(df[target])
except ValueError:
    st.error("The target column has values that the models have not seen.")
    st.stop()

# quick summary of the uploaded file
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", len(meta["features"]))
c3.metric("Default rate", f"{(y == 1).mean():.1%}")

with st.expander("Preview of uploaded data"):
    st.dataframe(df.head(20))


@st.cache_data(show_spinner=False)
def evaluate_all(_models, features, labels):
    out = {}
    for name, pipe in _models.items():
        pred = pipe.predict(features)
        proba = pipe.predict_proba(features)
        out[name] = compute_metrics(labels, pred, proba)
    return pd.DataFrame(out).T


results = evaluate_all(models, X, y)

tab_one, tab_all = st.tabs([choice, "Compare all models"])

# ---- tab 1: details of the selected model ----
with tab_one:
    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, (name, value) in zip(cols, results.loc[choice].items()):
        col.metric(name, f"{value:.3f}", help=METRIC_HELP[name])

    pred = models[choice].predict(X)
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No default", "Default"],
                    yticklabels=["No default", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(y, pred, target_names=["No default", "Default"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

# ---- tab 2: all models side by side ----
with tab_all:
    st.subheader("All models on the uploaded data")
    st.dataframe(results.round(4).style.highlight_max(axis=0, color="#cfe2f3"))
    best = results["MCC"].idxmax()
    st.success(f"Best MCC on this file: {best} ({results.loc[best, 'MCC']:.3f})")
    st.bar_chart(results[["Accuracy", "AUC", "F1", "MCC"]])
