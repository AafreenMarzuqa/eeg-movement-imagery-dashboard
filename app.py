
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# EEG Movement Imagery Dashboard
# Software-only demo using recorded BCI Competition IV 2b EEG
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "iv2b_epochs_causal_v2.npz")
MODEL_PATH = os.path.join(APP_DIR, "eeg_left_right_models.joblib")
SUMMARY_PATH = os.path.join(APP_DIR, "strict_policy_summary.csv")

LABELS = {0: "LEFT HAND IMAGINED", 1: "RIGHT HAND IMAGINED"}
CHANNELS = ["C3", "Cz", "C4"]
SAMPLE_RATE = 250
WINDOWS = [1.0, 1.5, 2.0, 2.5, 3.0]

st.set_page_config(
    page_title="EEG Movement Imagery Dashboard",
    page_icon="🧠",
    layout="wide"
)


# --------------------------- Loading ---------------------------

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        if "X" not in data.files or "y" not in data.files:
            raise ValueError(
                "The NPZ file must contain arrays named X and y."
            )
        X = data["X"]
        y = data["y"]
        subjects = data["subjects"] if "subjects" in data.files else None
    return X, y, subjects


@st.cache_data
def load_summary():
    if not os.path.isfile(SUMMARY_PATH):
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_PATH)


# ----------------------- Model discovery -----------------------

def select_window_artifact(artifact, window):
    """Select a model entry for the requested observation window."""
    if not isinstance(artifact, dict):
        return artifact

    # Common nested format: {"models": {window: model, ...}}
    if isinstance(artifact.get("models"), dict):
        nested = artifact["models"]
        selected = select_window_artifact(nested, window)
        if selected is not None:
            return selected

    # Common keys: 1.0, "1.0", "1.0s", "1.0_seconds"
    for key, value in artifact.items():
        try:
            key_text = str(key).lower().replace("_seconds", "").replace("seconds", "")
            key_text = key_text.replace("sec", "").replace("s", "").strip()
            if abs(float(key_text) - float(window)) < 1e-6:
                return value
        except (TypeError, ValueError):
            continue

    # Sometimes models are stored as a list of entries.
    return None


def walk_objects(obj, depth=0, seen=None):
    """Find nested sklearn estimators without looping forever."""
    if seen is None:
        seen = set()

    if obj is None or depth > 8:
        return

    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)

    yield obj

    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk_objects(value, depth + 1, seen)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from walk_objects(value, depth + 1, seen)
        return

    # sklearn Pipeline / estimator steps
    steps = getattr(obj, "steps", None)
    if steps:
        for _, step in steps:
            yield from walk_objects(step, depth + 1, seen)

    # Common saved estimator attributes
    for attr in ("csp", "clf", "classifier", "model", "estimator",
                 "pipeline", "transformer", "feature_extractor"):
        try:
            child = getattr(obj, attr, None)
        except Exception:
            child = None
        if child is not None and child is not obj:
            yield from walk_objects(child, depth + 1, seen)


def get_predictor(model_entry):
    """Return a usable prediction estimator or raise a clear error."""
    candidates = list(walk_objects(model_entry))

    # Prefer a full pipeline that can predict directly.
    for obj in candidates:
        if callable(getattr(obj, "predict", None)):
            if callable(getattr(obj, "transform", None)):
                continue
            return obj

    raise ValueError(
        "No object with a callable predict() method was found in the "
        "selected saved model entry. The artifact's internal structure "
        "needs to be inspected."
    )


def get_transformer(model_entry, predictor):
    """Find a separate CSP/feature transformer if one is saved."""
    for obj in walk_objects(model_entry):
        if obj is predictor:
            continue
        if callable(getattr(obj, "transform", None)):
            return obj
    return None


def predict_trial(artifact, trial_data, window):
    """
    Predict a recorded trial.

    Supports:
    - a saved sklearn Pipeline
    - a dictionary containing a pipeline
    - a dictionary containing separate transformer and classifier objects
    """
    model_entry = select_window_artifact(artifact, window)

    if model_entry is None:
        if callable(getattr(artifact, "predict", None)):
            model_entry = artifact
        else:
            keys = list(artifact.keys()) if isinstance(artifact, dict) else []
            raise ValueError(
                f"No model entry found for {window:.1f} seconds. "
                f"Available top-level keys: {keys}"
            )

    n_samples = int(round(float(window) * SAMPLE_RATE))
    if trial_data.shape[-1] < n_samples:
        raise ValueError(
            f"Trial has {trial_data.shape[-1]} samples, but "
            f"{n_samples} samples are needed for {window:.1f} seconds."
        )

    segment = np.asarray(trial_data[:, :n_samples], dtype=np.float64)
    input_data = segment[np.newaxis, :, :]

    predictor = get_predictor(model_entry)
    transformer = get_transformer(model_entry, predictor)

    # A full pipeline handles features internally.
    if callable(getattr(predictor, "predict", None)):
        try:
            raw_prediction = predictor.predict(input_data)
            probabilities = (
                np.asarray(predictor.predict_proba(input_data))[0]
                if callable(getattr(predictor, "predict_proba", None))
                else None
            )
            prediction = int(np.asarray(raw_prediction).ravel()[0])
            return prediction, probabilities, segment
        except Exception as direct_error:
            # If the predictor expects extracted features, try a separate
            # saved transformer. Otherwise provide the original error.
            if transformer is None:
                raise ValueError(
                    "The saved predictor rejected EEG input with shape "
                    f"{input_data.shape}. Original error: {direct_error}"
                ) from direct_error

    if transformer is None:
        raise ValueError(
            "The saved model has no usable feature transformer. "
            "Expected a pipeline or a separate CSP transformer."
        )

    features = transformer.transform(input_data)
    raw_prediction = predictor.predict(features)
    prediction = int(np.asarray(raw_prediction).ravel()[0])

    probabilities = None
    if callable(getattr(predictor, "predict_proba", None)):
        probabilities = np.asarray(
            predictor.predict_proba(features)
        )[0]

    return prediction, probabilities, segment


# ----------------------------- UI -----------------------------

st.title("🧠 EEG Movement Imagery Dashboard")
st.caption("Recorded EEG classification using CSP + Logistic Regression")
st.info(
    "This application predicts recorded EEG trials. "
    "It does not capture live brain signals."
)

# Validate deployment files
required = [DATA_PATH, MODEL_PATH, SUMMARY_PATH]
missing = [os.path.basename(p) for p in required if not os.path.isfile(p)]

if missing:
    st.error("Required files are missing from the deployed repository:")
    for name in missing:
        st.write(f"- `{name}`")
    st.stop()

try:
    X, y, subjects = load_dataset()
    artifact = load_artifact()
    summary = load_summary()
except Exception as exc:
    st.error("Unable to load the dataset, model, or summary.")
    st.exception(exc)
    st.stop()

if X.ndim != 3:
    st.error(f"Expected X with shape (trials, channels, samples); got {X.shape}.")
    st.stop()

if len(X) != len(y):
    st.error("Dataset issue: X and y have different numbers of trials.")
    st.stop()

if len(X) == 0:
    st.error("The dataset contains no trials.")
    st.stop()

if X.shape[1] != 3:
    st.warning(f"Expected 3 channels; dataset has {X.shape[1]}.")

if X.shape[2] < SAMPLE_RATE:
    st.error("Trials are shorter than one second.")
    st.stop()

available_windows = [
    w for w in WINDOWS
    if int(round(w * SAMPLE_RATE)) <= X.shape[2]
]

if not available_windows:
    st.error("No supported observation window fits this dataset.")
    st.stop()

tab_prediction, tab_research = st.tabs(
    ["EEG Prediction", "Research Evaluation"]
)

# ------------------------ Prediction tab ------------------------

with tab_prediction:
    st.header("Predict imagined movement")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        trial_index = st.slider(
            "Select EEG trial",
            min_value=0,
            max_value=len(X) - 1,
            value=min(579, len(X) - 1),
            step=1
        )

    with col2:
        selected_window = st.selectbox(
            "Observation window (seconds)",
            options=available_windows,
            index=len(available_windows) - 1,
            format_func=lambda value: f"{value:.1f}"
        )

    if st.button("Predict movement", type="primary"):
        try:
            prediction, probabilities, segment = predict_trial(
                artifact,
                X[trial_index],
                float(selected_window)
            )

            st.session_state["eeg_result"] = {
                "trial": trial_index,
                "window": float(selected_window),
                "prediction": prediction,
                "probabilities": probabilities,
                "segment": segment
            }
        except Exception as exc:
            st.error("Prediction could not be completed.")
            st.exception(exc)

    result = st.session_state.get("eeg_result")

    if result is not None:
        same_selection = (
            result["trial"] == trial_index
            and result["window"] == float(selected_window)
        )

        if same_selection:
            prediction = result["prediction"]
            probabilities = result["probabilities"]
            segment = result["segment"]

            if prediction in LABELS:
                st.markdown(f"## Prediction: {LABELS[prediction]}")
            else:
                st.warning(f"Model returned unexpected class: {prediction}")

            actual = int(y[trial_index])
            actual_label = LABELS.get(actual, f"Class {actual}")
            st.write("Recorded dataset label:", actual_label)

            if prediction == actual:
                st.success("Prediction matches the recorded dataset label.")
            else:
                st.warning("Prediction differs from the recorded dataset label.")

            if probabilities is not None:
                probabilities = np.asarray(probabilities).ravel()

                classes = getattr(
                    get_predictor(
                        select_window_artifact(artifact, float(selected_window))
                        if select_window_artifact(artifact, float(selected_window)) is not None
                        else artifact
                    ),
                    "classes_",
                    np.arange(len(probabilities))
                )

                class_probs = {
                    int(c): float(p)
                    for c, p in zip(classes, probabilities)
                }

                left_probability = class_probs.get(0, 0.0)
                right_probability = class_probs.get(1, 0.0)

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Model probability score",
                    f"{max(probabilities) * 100:.1f}%"
                )
                c2.metric(
                    "Left-hand probability",
                    f"{left_probability * 100:.1f}%"
                )
                c3.metric(
                    "Right-hand probability",
                    f"{right_probability * 100:.1f}%"
                )

                st.caption(
                    "Probabilities are model outputs, not independently "
                    "calibrated confidence estimates."
                )

                st.subheader("Class probabilities")
                chart_df = pd.DataFrame({
                    "Imagined movement": ["Left hand", "Right hand"],
                    "Probability (%)": [
                        left_probability * 100,
                        right_probability * 100
                    ]
                }).set_index("Imagined movement")
                st.bar_chart(chart_df)
            else:
                st.caption(
                    "The saved predictor does not provide predict_proba()."
                )

            st.subheader(f"Recorded EEG trial {trial_index}")
            times = np.arange(segment.shape[1]) / SAMPLE_RATE

            fig, axes = plt.subplots(
                segment.shape[0], 1,
                figsize=(12, 7),
                sharex=True
            )

            if segment.shape[0] == 1:
                axes = [axes]

            for index, ax in enumerate(axes):
                ax.plot(times, segment[index], linewidth=1)
                channel = CHANNELS[index] if index < len(CHANNELS) else f"Channel {index + 1}"
                ax.set_ylabel(channel)
                ax.grid(True, alpha=0.25)

            axes[-1].set_xlabel("Time (seconds)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.caption("Click **Predict movement** to update the result.")

# ----------------------- Research tab -----------------------

with tab_research:
    st.header("Research Evaluation")

    st.write(
        "The following table displays the saved offline research results. "
        "These results are separate from the demo model's prediction for "
        "an individual trial."
    )

    if summary.empty:
        st.warning("The research summary CSV is empty.")
    else:
        st.dataframe(summary, use_container_width=True)

        numeric = summary.select_dtypes(include=np.number)
        if not numeric.empty:
            st.subheader("Numeric research metrics")
            st.bar_chart(numeric)

    st.info(
        "This is an offline replay on recorded EEG trials, not a live EEG "
        "or real-time latency demonstration."
    )

st.divider()
st.caption(
    "Software-only demonstration • BCI Competition IV 2b recorded EEG"
)
