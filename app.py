
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# EEG Movement Imagery Dashboard
# Recorded BCI Competition IV 2b EEG — software-only demo
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
    layout="wide",
)


# --------------------------- Load files ---------------------------

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        if "X" not in data.files or "y" not in data.files:
            raise ValueError("NPZ file must contain arrays named X and y.")

        X = data["X"]
        y = data["y"]
        subjects = data["subjects"] if "subjects" in data.files else None

    return X, y, subjects


@st.cache_data
def load_summary():
    if not os.path.isfile(SUMMARY_PATH):
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_PATH)


# --------------------------- Model helpers ---------------------------

def select_window_artifact(artifact, window):
    """Find the saved model corresponding to the selected time window."""
    if not isinstance(artifact, dict):
        return artifact

    if isinstance(artifact.get("models"), dict):
        selected = select_window_artifact(artifact["models"], window)
        if selected is not None:
            return selected

    for key, value in artifact.items():
        try:
            key_text = str(key).lower()
            key_text = key_text.replace("_seconds", "")
            key_text = key_text.replace("seconds", "")
            key_text = key_text.replace("sec", "")
            key_text = key_text.replace("s", "").strip()

            if abs(float(key_text) - float(window)) < 1e-6:
                return value
        except (TypeError, ValueError):
            continue

    return None


def walk_objects(obj, depth=0, seen=None):
    """Inspect nested saved estimators safely."""
    if seen is None:
        seen = set()

    if obj is None or depth > 8:
        return

    obj_id = id(obj)
    if obj_id in seen:
        return

    seen.add(obj_id)
    yield obj

    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk_objects(value, depth + 1, seen)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from walk_objects(value, depth + 1, seen)
        return

    steps = getattr(obj, "steps", None)
    if steps:
        for _, step in steps:
            yield from walk_objects(step, depth + 1, seen)

    attributes = (
        "csp", "clf", "classifier", "model", "estimator",
        "pipeline", "transformer", "feature_extractor",
    )

    for attr in attributes:
        try:
            child = getattr(obj, attr, None)
        except Exception:
            child = None

        if child is not None and child is not obj:
            yield from walk_objects(child, depth + 1, seen)


def get_predictor(model_entry):
    """Find a prediction-capable estimator in the selected model entry."""
    candidates = list(walk_objects(model_entry))

    for obj in candidates:
        if callable(getattr(obj, "predict", None)):
            if callable(getattr(obj, "transform", None)):
                continue
            return obj

    raise ValueError(
        "No usable predict() estimator was found in this saved model entry. "
        "The model artifact structure must be inspected."
    )


def get_transformer(model_entry, predictor):
    """Find a separately saved feature transformer, such as CSP."""
    for obj in walk_objects(model_entry):
        if obj is predictor:
            continue

        if callable(getattr(obj, "transform", None)):
            return obj

    return None


def predict_trial(artifact, trial_data, window):
    """Predict one recorded EEG trial using the saved model."""
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

    n_samples = int(round(window * SAMPLE_RATE))

    if trial_data.shape[-1] < n_samples:
        raise ValueError(
            f"Trial contains {trial_data.shape[-1]} samples, but "
            f"{n_samples} are required for {window:.1f} seconds."
        )

    segment = np.asarray(
        trial_data[:, :n_samples],
        dtype=np.float64,
    )
    input_data = segment[np.newaxis, :, :]

    predictor = get_predictor(model_entry)
    transformer = get_transformer(model_entry, predictor)

    # First attempt: full pipeline accepts raw EEG input.
    try:
        raw_prediction = predictor.predict(input_data)

        probabilities = (
            np.asarray(predictor.predict_proba(input_data))[0]
            if callable(getattr(predictor, "predict_proba", None))
            else None
        )

        prediction = int(np.asarray(raw_prediction).ravel()[0])
        classes = getattr(predictor, "classes_", None)

        return prediction, probabilities, segment, classes

    except Exception as direct_error:
        if transformer is None:
            raise ValueError(
                "The saved predictor rejected the EEG input shape "
                f"{input_data.shape}. Original error: {direct_error}"
            ) from direct_error

    # Second attempt: transform EEG into features before classification.
    features = transformer.transform(input_data)
    raw_prediction = predictor.predict(features)
    prediction = int(np.asarray(raw_prediction).ravel()[0])

    probabilities = None
    if callable(getattr(predictor, "predict_proba", None)):
        probabilities = np.asarray(
            predictor.predict_proba(features)
        )[0]

    classes = getattr(predictor, "classes_", None)

    return prediction, probabilities, segment, classes


# --------------------------- Page heading ---------------------------

st.title("🧠 EEG Movement Imagery Dashboard")
st.caption("Recorded EEG classification using CSP + Logistic Regression")

st.info(
    "This application predicts recorded EEG trials. "
    "It does not capture live brain signals."
)


# --------------------------- Validate files ---------------------------

required_files = [DATA_PATH, MODEL_PATH, SUMMARY_PATH]
missing_files = [
    os.path.basename(path)
    for path in required_files
    if not os.path.isfile(path)
]

if missing_files:
    st.error("Required files are missing from the deployed repository:")
    for filename in missing_files:
        st.write(f"- `{filename}`")
    st.stop()

try:
    X, y, subjects = load_dataset()
    artifact = load_artifact()
    summary = load_summary()

except Exception as exc:
    st.error("Unable to load the dataset, model, or research summary.")
    st.exception(exc)
    st.stop()

if X.ndim != 3:
    st.error(
        f"Expected X shape (trials, channels, samples); received {X.shape}."
    )
    st.stop()

if len(X) != len(y):
    st.error("X and y contain different numbers of trials.")
    st.stop()

if len(X) == 0:
    st.error("The dataset contains no trials.")
    st.stop()

if X.shape[1] != 3:
    st.warning(f"Expected 3 EEG channels; dataset contains {X.shape[1]}.")

if X.shape[2] < SAMPLE_RATE:
    st.error("Trials are shorter than one second.")
    st.stop()

available_windows = [
    window
    for window in WINDOWS
    if int(round(window * SAMPLE_RATE)) <= X.shape[2]
]

if not available_windows:
    st.error("No supported observation window fits this dataset.")
    st.stop()


tab_prediction, tab_research = st.tabs(
    ["EEG Prediction", "Research Evaluation"]
)


# ============================================================
# TAB 1 — EEG PREDICTION
# ============================================================

with tab_prediction:
    st.header("Predict imagined movement")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        trial_index = st.slider(
            "Select EEG trial",
            min_value=0,
            max_value=len(X) - 1,
            value=min(579, len(X) - 1),
            step=1,
        )

    with col2:
        selected_window = st.selectbox(
            "Observation window (seconds)",
            options=available_windows,
            index=len(available_windows) - 1,
            format_func=lambda value: f"{value:.1f}",
        )

    if st.button("Predict movement", type="primary"):
        try:
            prediction, probabilities, segment, model_classes = predict_trial(
                artifact,
                X[trial_index],
                float(selected_window),
            )

            st.session_state["eeg_result"] = {
                "trial": trial_index,
                "window": float(selected_window),
                "prediction": prediction,
                "probabilities": probabilities,
                "classes": model_classes,
                "segment": segment,
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
            model_classes = result.get("classes")
            segment = result["segment"]

            if prediction in LABELS:
                st.markdown(f"## Prediction: {LABELS[prediction]}")
            else:
                st.warning(f"Unexpected model class: {prediction}")

            actual = int(y[trial_index])
            actual_label = LABELS.get(actual, f"Class {actual}")

            st.write("Recorded dataset label:", actual_label)

            if prediction == actual:
                st.success(
                    "Prediction matches the recorded dataset label."
                )
            else:
                st.warning(
                    "Prediction differs from the recorded dataset label."
                )

            if probabilities is not None:
                probabilities = np.asarray(probabilities).ravel()

                if model_classes is None:
                    classes = np.arange(len(probabilities))
                else:
                    classes = np.asarray(model_classes).ravel()

                class_probs = {
                    int(class_id): float(probability)
                    for class_id, probability in zip(
                        classes, probabilities
                    )
                }

                left_probability = class_probs.get(0, 0.0)
                right_probability = class_probs.get(1, 0.0)

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Model probability score",
                    f"{max(probabilities) * 100:.1f}%",
                )
                metric2.metric(
                    "Left-hand probability",
                    f"{left_probability * 100:.1f}%",
                )
                metric3.metric(
                    "Right-hand probability",
                    f"{right_probability * 100:.1f}%",
                )

                st.caption(
                    "Probabilities are model outputs, not independently "
                    "calibrated confidence estimates."
                )

                st.subheader("Class probabilities")

                probability_df = pd.DataFrame(
                    {
                        "Imagined movement": ["Left hand", "Right hand"],
                        "Probability (%)": [
                            left_probability * 100,
                            right_probability * 100,
                        ],
                    }
                ).set_index("Imagined movement")

                st.bar_chart(probability_df)

            else:
                st.caption(
                    "The saved predictor does not provide predict_proba()."
                )

            st.subheader(f"Recorded EEG trial {trial_index}")

            times = np.arange(segment.shape[1]) / SAMPLE_RATE

            fig, axes = plt.subplots(
                segment.shape[0],
                1,
                figsize=(12, 7),
                sharex=True,
            )

            if segment.shape[0] == 1:
                axes = [axes]

            for index, ax in enumerate(axes):
                ax.plot(times, segment[index], linewidth=1)

                channel = (
                    CHANNELS[index]
                    if index < len(CHANNELS)
                    else f"Channel {index + 1}"
                )

                ax.set_ylabel(channel)
                ax.grid(True, alpha=0.25)

            axes[-1].set_xlabel("Time (seconds)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.caption(
                "Click Predict movement to update the selected trial."
            )


# ============================================================
# TAB 2 — RESEARCH EVALUATION
# ============================================================

with tab_research:
    st.header("Research Evaluation")

    st.write(
        "This page displays saved offline research results. These results "
        "are separate from the demo model's prediction for an individual "
        "trial."
    )

    if summary.empty:
        st.warning("The research summary CSV is empty.")

    else:
        # Display the key comparison columns without unrelated metrics.
        display_columns = [
            "rule",
            "subjects",
            "mean_fixed_accuracy_pct",
            "mean_adaptive_accuracy_pct",
            "mean_accuracy_difference_pp",
            "mean_fixed_balanced_accuracy_pct",
            "mean_adaptive_balanced_accuracy_pct",
            "mean_balanced_accuracy_difference_pp",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in summary.columns
        ]

        if available_columns:
            st.subheader("Research results table")

            st.dataframe(
                summary[available_columns].round(4),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "Expected comparison columns were not found. "
                "Displaying the original saved CSV table."
            )
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

        # Chart 1 — Classification accuracy
        accuracy_columns = [
            "mean_fixed_accuracy_pct",
            "mean_adaptive_accuracy_pct",
        ]

        if all(column in summary.columns for column in accuracy_columns):
            st.subheader("1. Classification accuracy")

            accuracy_df = summary[
                ["rule"] + accuracy_columns
            ].copy()

            accuracy_df = accuracy_df.set_index("rule")
            accuracy_df = accuracy_df.rename(
                columns={
                    "mean_fixed_accuracy_pct": "Fixed baseline",
                    "mean_adaptive_accuracy_pct": "Adaptive method",
                }
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))
            accuracy_df.plot(kind="bar", ax=ax, width=0.75)

            ax.set_ylabel("Mean accuracy (%)")
            ax.set_xlabel("Adaptive rule")
            ax.set_ylim(0, 100)
            ax.legend(title="Method")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=0)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Chart 2 — Balanced accuracy
        balanced_columns = [
            "mean_fixed_balanced_accuracy_pct",
            "mean_adaptive_balanced_accuracy_pct",
        ]

        if all(column in summary.columns for column in balanced_columns):
            st.subheader("2. Balanced accuracy")

            balanced_df = summary[
                ["rule"] + balanced_columns
            ].copy()

            balanced_df = balanced_df.set_index("rule")
            balanced_df = balanced_df.rename(
                columns={
                    "mean_fixed_balanced_accuracy_pct": "Fixed baseline",
                    "mean_adaptive_balanced_accuracy_pct": "Adaptive method",
                }
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))
            balanced_df.plot(kind="bar", ax=ax, width=0.75)

            ax.set_ylabel("Mean balanced accuracy (%)")
            ax.set_xlabel("Adaptive rule")
            ax.set_ylim(0, 100)
            ax.legend(title="Method")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=0)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Chart 3 — Differences from the fixed baseline
        difference_columns = [
            "mean_accuracy_difference_pp",
            "mean_balanced_accuracy_difference_pp",
        ]

        available_differences = [
            column
            for column in difference_columns
            if column in summary.columns
        ]

        if available_differences:
            st.subheader(
                "3. Change relative to fixed baseline "
                "(percentage points)"
            )

            difference_df = summary[
                ["rule"] + available_differences
            ].copy()

            difference_df = difference_df.set_index("rule")
            difference_df = difference_df.rename(
                columns={
                    "mean_accuracy_difference_pp":
                        "Accuracy difference (pp)",
                    "mean_balanced_accuracy_difference_pp":
                        "Balanced accuracy difference (pp)",
                }
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))
            difference_df.plot(kind="bar", ax=ax, width=0.75)

            ax.axhline(0, linewidth=1)
            ax.set_ylabel("Difference (percentage points)")
            ax.set_xlabel("Adaptive rule")
            ax.legend(title="Metric")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=0)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Interpret the saved accuracy values without claiming improvement.
        if (
            "mean_fixed_accuracy_pct" in summary.columns
            and "mean_adaptive_accuracy_pct" in summary.columns
        ):
            fixed_mean = summary["mean_fixed_accuracy_pct"].mean()
            adaptive_mean = summary["mean_adaptive_accuracy_pct"].mean()

            if adaptive_mean < fixed_mean:
                st.warning(
                    "Across the displayed rules, the saved adaptive "
                    "accuracy is lower on average than the fixed baseline. "
                    "Report this result as observed; do not describe it "
                    "as an improvement."
                )
            elif adaptive_mean > fixed_mean:
                st.info(
                    "Across the displayed rules, the saved adaptive "
                    "accuracy is higher on average than the fixed baseline. "
                    "Interpret this alongside the evaluation protocol "
                    "and variability."
                )
            else:
                st.info(
                    "Across the displayed rules, mean adaptive accuracy "
                    "and fixed-baseline accuracy are equal."
                )

    st.info(
        "These are saved offline evaluation results on recorded EEG trials. "
        "They are not live EEG measurements or a real-time latency "
        "demonstration."
    )


# --------------------------- Footer ---------------------------

st.divider()
st.caption(
    "Software-only demonstration • BCI Competition IV 2b recorded EEG"
)
