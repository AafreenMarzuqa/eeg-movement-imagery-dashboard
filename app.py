
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    APP_DIR, "iv2b_epochs_causal_v2.npz"
)
MODEL_PATH = os.path.join(
    APP_DIR, "eeg_left_right_models.joblib"
)
SUMMARY_PATH = os.path.join(
    APP_DIR, "strict_policy_summary.csv"
)

LABELS = [
    "LEFT HAND IMAGINED",
    "RIGHT HAND IMAGINED"
]

SAMPLE_RATE = 250
WINDOWS = [1.0, 1.5, 2.0, 2.5, 3.0]

st.set_page_config(
    page_title="EEG Movement Imagery Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Load files
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    with np.load(DATA_PATH, allow_pickle=False) as d:
        X = d["X"]
        y = d["y"]
        subjects = d["subjects"] if "subjects" in d.files else None

    return X, y, subjects


@st.cache_data
def load_summary():
    if os.path.exists(SUMMARY_PATH):
        return pd.read_csv(SUMMARY_PATH)
    return pd.DataFrame()


# --------------------------------------------------
# Model helper functions
# --------------------------------------------------
def get_window_model(artifact, window):
    """Find the model saved for the selected observation window."""
    if not isinstance(artifact, dict):
        return artifact

    # Support common dictionary layouts.
    if "models" in artifact and isinstance(artifact["models"], dict):
        artifact = artifact["models"]

    candidates = [
        window,
        str(window),
        f"{window:.1f}",
        int(window) if float(window).is_integer() else None
    ]

    for key in candidates:
        if key is not None and key in artifact:
            return artifact[key]

    # Also handle keys stored as strings such as "1.0s".
    for key, model in artifact.items():
        try:
            if abs(float(str(key).lower().replace("s", "")) - window) < 1e-6:
                return model
        except (ValueError, TypeError):
            continue

    raise KeyError(
        f"No model found for {window}-second window. "
        f"Available model keys: {list(artifact.keys())}"
    )


def predict_trial(model, trial_data, window):
    """Predict one trial using the selected window model."""
    n_samples = int(round(window * SAMPLE_RATE))
    segment = trial_data[:, :n_samples]

    # Most saved CSP pipelines expect (trials, channels, samples).
    input_data = segment[np.newaxis, :, :]

    prediction = int(np.asarray(model.predict(input_data)).ravel()[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(input_data)
        )[0]

    return prediction, probabilities, segment


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🧠 EEG Movement Imagery Dashboard")
st.caption("Recorded EEG classification using CSP + Logistic Regression")

st.info(
    "This application predicts recorded EEG trials. "
    "It does not capture live brain signals."
)

# --------------------------------------------------
# Check files and load resources
# --------------------------------------------------
required_files = [DATA_PATH, MODEL_PATH, SUMMARY_PATH]
missing_files = [p for p in required_files if not os.path.isfile(p)]

if missing_files:
    st.error("Required deployment files are missing:")
    for path in missing_files:
        st.write(f"- `{os.path.basename(path)}`")
    st.stop()

try:
    X, y, subjects = load_dataset()
    artifact = load_model()
    summary = load_summary()
except Exception as e:
    st.error("Could not load the dataset, model, or research summary.")
    st.exception(e)
    st.stop()

if X.ndim != 3:
    st.error(f"Expected EEG data with 3 dimensions; received {X.shape}.")
    st.stop()

if len(X) != len(y):
    st.error("Dataset error: number of EEG trials and labels do not match.")
    st.stop()

if X.shape[1] != 3:
    st.warning(
        f"Expected 3 EEG channels (C3, Cz, C4); found {X.shape[1]}."
    )

if X.shape[2] < SAMPLE_RATE:
    st.error("Each EEG trial is shorter than one second.")
    st.stop()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
prediction_tab, evaluation_tab = st.tabs(
    ["EEG Prediction", "Research Evaluation"]
)

# --------------------------------------------------
# Prediction tab
# --------------------------------------------------
with prediction_tab:
    st.header("Predict imagined movement")

    col_trial, col_window = st.columns([1.2, 1])

    with col_trial:
        trial_index = st.slider(
            "Select EEG trial",
            min_value=0,
            max_value=len(X) - 1,
            value=min(579, len(X) - 1),
            step=1
        )

    available_windows = [
        w for w in WINDOWS
        if int(round(w * SAMPLE_RATE)) <= X.shape[2]
    ]

    with col_window:
        selected_window = st.selectbox(
            "Observation window (seconds)",
            options=available_windows,
            index=len(available_windows) - 1,
            format_func=lambda w: f"{w:.1f}"
        )

    if st.button("Predict movement", type="primary"):
        try:
            selected_model = get_window_model(
                artifact, float(selected_window)
            )

            pred, probs, signal_segment = predict_trial(
                selected_model,
                X[trial_index],
                float(selected_window)
            )

            st.session_state["prediction_result"] = {
                "trial": trial_index,
                "window": float(selected_window),
                "prediction": pred,
                "probabilities": probs,
                "signal": signal_segment
            }
        except Exception as e:
            st.error("Prediction failed. Check the model and input dimensions.")
            st.exception(e)

    result = st.session_state.get("prediction_result")

    if result is not None:
        if (
            result["trial"] != trial_index
            or result["window"] != float(selected_window)
        ):
            st.caption("Click **Predict movement** to update the result.")
        else:
            pred = result["prediction"]
            probs = result["probabilities"]
            signal_segment = result["signal"]

            if pred not in (0, 1):
                st.error(f"Unexpected model class: {pred}")
            else:
                st.markdown(f"## Prediction: {LABELS[pred]}")

                if probs is not None and len(probs) >= 2:
                    classes = list(
                        getattr(
                            get_window_model(artifact, float(selected_window)),
                            "classes_",
                            [0, 1]
                        )
                    )

                    class_prob = {
                        int(c): float(p)
                        for c, p in zip(classes, probs)
                    }
                    left_prob = class_prob.get(0, 0.0)
                    right_prob = class_prob.get(1, 0.0)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Model probability score",
                              f"{max(probs) * 100:.1f}%")
                    c2.metric("Left-hand probability",
                              f"{left_prob * 100:.1f}%")
                    c3.metric("Right-hand probability",
                              f"{right_prob * 100:.1f}%")
                else:
                    st.caption(
                        "This model does not provide probability scores."
                    )

                actual = int(y[trial_index])
                st.write(
                    "Recorded dataset label:",
                    LABELS[actual] if actual in (0, 1) else str(actual)
                )

                if pred == actual:
                    st.success("Prediction matches the recorded dataset label.")
                else:
                    st.warning(
                        "Prediction differs from the recorded dataset label."
                    )

                st.caption(
                    "The displayed probability is the model's output, "
                    "not an independently calibrated confidence estimate."
                )

            st.subheader(f"Recorded EEG trial {trial_index}")
            times = np.arange(signal_segment.shape[1]) / SAMPLE_RATE
            channel_names = ["C3", "Cz", "C4"]

            fig, axes = plt.subplots(
                signal_segment.shape[0], 1,
                figsize=(12, 7), sharex=True
            )
            if signal_segment.shape[0] == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.plot(times, signal_segment[i], linewidth=1)
                ax.set_ylabel(
                    channel_names[i]
                    if i < len(channel_names)
                    else f"Channel {i + 1}"
                )
                ax.grid(True, alpha=0.25)

            axes[-1].set_xlabel("Time (seconds)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Probability chart
            if probs is not None and len(probs) >= 2:
                st.subheader("Class probabilities")
                chart_data = pd.DataFrame({
                    "Imagined movement": [
                        "Left hand", "Right hand"
                    ],
                    "Probability (%)": [
                        left_prob * 100, right_prob * 100
                    ]
                }).set_index("Imagined movement")
                st.bar_chart(chart_data)

# --------------------------------------------------
# Research evaluation tab
# --------------------------------------------------
with evaluation_tab:
    st.header("Research Evaluation")

    st.write(
        "This table reports the saved research evaluation results. "
        "These results are separate from the demo model trained on all "
        "available trials."
    )

    if summary.empty:
        st.warning("Research summary CSV is empty or unavailable.")
    else:
        st.dataframe(summary, use_container_width=True)

        numeric_columns = summary.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:
            st.subheader("Numeric research metrics")
            st.bar_chart(summary[numeric_columns])

    st.info(
        "The research evaluation uses recorded EEG data and offline replay. "
        "It is not a live EEG or real-time latency demonstration."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Software-only demonstration • BCI Competition IV 2b recorded EEG"
)
