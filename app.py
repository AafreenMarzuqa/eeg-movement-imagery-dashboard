import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EEG Movement Imagery",
    page_icon="🧠",
    layout="wide"
)

DATA = (
    "/content/drive/MyDrive/BCI_Project/results/"
    "iv2b_causal_v2/iv2b_epochs_causal_v2.npz"
)
MODEL_PATH = (
    "/content/drive/MyDrive/BCI_Project/"
    "software_demo_model/eeg_left_right_models.joblib"
)
SUMMARY_PATH = (
    "/content/drive/MyDrive/BCI_Project/results/"
    "iv2b_causal_v2/adaptive_stability_v2/"
    "step38_strict_policy_20261013/strict_policy_summary.csv"
)

LABELS = {0: "LEFT HAND IMAGINED", 1: "RIGHT HAND IMAGINED"}
CHANNELS = ["C3", "Cz", "C4"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_eeg():
    with np.load(DATA) as d:
        return d["X"].astype(np.float64), d["y"].astype(int)

@st.cache_data
def load_evaluation():
    return pd.read_csv(SUMMARY_PATH)

st.title("🧠 EEG Movement Imagery Dashboard")
st.caption("Recorded EEG classification using CSP + Logistic Regression")

if not all(os.path.exists(p) for p in [DATA, MODEL_PATH, SUMMARY_PATH]):
    st.error("Required files not found. Check that Google Drive is mounted in Colab.")
    st.stop()

artifact = load_model()
X, y = load_eeg()
evaluation = load_evaluation()

st.info(
    "This application predicts recorded EEG trials. "
    "It does not capture live brain signals."
)

tab_predict, tab_evaluate = st.tabs(["EEG Prediction", "Research Evaluation"])

with tab_predict:
    st.subheader("Predict imagined movement")

    col1, col2 = st.columns(2)
    with col1:
        trial = st.slider("Select EEG trial", 0, len(y) - 1, 0)
    with col2:
        seconds = st.selectbox(
            "Observation window (seconds)",
            [1.0, 1.5, 2.0, 2.5, 3.0],
            index=4
        )

    if st.button("Predict movement", type="primary"):
        saved = artifact["models"][seconds]
        eeg = X[trial:trial + 1, :, :saved["samples"]]
        features = saved["csp"].transform(eeg)
        clf = saved["classifier"]

        pred = int(clf.predict(features)[0])
        raw = clf.predict_proba(features)[0]
        probs = {
            int(label): float(p)
            for label, p in zip(clf.classes_, raw)
        }

        st.markdown(f"## Prediction: {LABELS[pred]}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model probability score", f"{probs[pred]:.1%}")
        m2.metric("Left-hand probability", f"{probs[0]:.1%}")
        m3.metric("Right-hand probability", f"{probs[1]:.1%}")

        st.write("Recorded dataset label:", LABELS[int(y[trial])])

        prob_df = pd.DataFrame({
            "Movement": ["Left hand", "Right hand"],
            "Probability (%)": [probs[0] * 100, probs[1] * 100]
        })
        st.bar_chart(prob_df.set_index("Movement"))

        times = np.arange(saved["samples"]) / artifact["sampling_rate"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
        for ch, name in enumerate(CHANNELS):
            axes[ch].plot(times, X[trial, ch, :saved["samples"]])
            axes[ch].set_ylabel(name)
            axes[ch].grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle(f"Recorded EEG trial {trial}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_evaluate:
    st.subheader("Nine-subject held-out research evaluation")

    two = evaluation.loc[evaluation["rule"] == "two_window"].iloc[0]
    three = evaluation.loc[evaluation["rule"] == "three_window"].iloc[0]
    fixed_acc = float(evaluation["mean_fixed_accuracy_pct"].iloc[0])
    fixed_bacc = float(evaluation["mean_fixed_balanced_accuracy_pct"].iloc[0])

    report = pd.DataFrame({
        "Method": ["Fixed (3 seconds)", "Two-window adaptive", "Three-window adaptive"],
        "Accuracy (%)": [
            fixed_acc,
            float(two["mean_adaptive_accuracy_pct"]),
            float(three["mean_adaptive_accuracy_pct"])
        ],
        "Balanced accuracy (%)": [
            fixed_bacc,
            float(two["mean_adaptive_balanced_accuracy_pct"]),
            float(three["mean_adaptive_balanced_accuracy_pct"])
        ],
        "Mean observation time (s)": [
            3.0,
            float(two["mean_window_s"]),
            float(three["mean_window_s"])
        ],
        "Early stop (%)": [
            0.0,
            float(two["mean_early_stop_pct"]),
            float(three["mean_early_stop_pct"])
        ]
    })

    st.dataframe(report.round(2), use_container_width=True, hide_index=True)

    st.markdown("### Accuracy comparison")
    st.bar_chart(report.set_index("Method")[["Accuracy (%)"]])

    st.markdown("### Mean observation time")
    st.bar_chart(report.set_index("Method")[["Mean observation time (s)"]])

    st.warning(
        "Adaptive methods reduced observation time, but mean accuracy "
        "was slightly lower than the fixed baseline in this evaluation. "
        "These research results are separate from the all-data-trained demo model."
    )

st.divider()
st.caption("Software-only demonstration • BCI Competition IV 2b recorded EEG")
