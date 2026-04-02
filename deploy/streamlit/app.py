from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from inference import (
    available_targets,
    build_residual_analysis,
    load_metrics,
    load_report_text,
    predict_future,
    resolve_artifact_paths,
)


def render_residual_plot(residual_data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(residual_data["Index"], residual_data["Residual"], color="#ff6b6b", linewidth=1.5)
    ax.axhline(0, color="#ffffff", linestyle="--", linewidth=1)
    ax.set_title("Residuals Plot")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.set_page_config(
    page_title="Smart STP Predictor",
    page_icon="💧",
    layout="wide",
)

st.title("Smart STP Predictor")
st.caption("Streamlit deployment for wastewater quality forecasting")

with st.sidebar:
    st.header("Prediction Settings")
    targets = available_targets()
    if not targets:
        st.error("No trained model files were found. Add model files in the project model paths.")
        st.stop()

    selected_target = st.selectbox("Target Parameter", options=targets, index=0)
    days = st.slider("Future Days", min_value=1, max_value=30, value=7)
    time_steps = st.slider("Time Window", min_value=7, max_value=90, value=30)

metrics, metrics_path = load_metrics(selected_target)
report_text, report_path = load_report_text(selected_target)
artifact_paths = resolve_artifact_paths(selected_target)

st.subheader("Upload Dataset")
st.write("Upload a CSV with historical STP data. Include the selected target column and numeric feature columns.")

uploaded_file = st.file_uploader("CSV file", type=["csv"])
df = None
residual_data = None
residual_metrics = None
residual_model_path = None
residual_error = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read CSV file: {exc}")
        st.stop()

    st.markdown("#### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    try:
        residual_data, residual_metrics, residual_model_path = build_residual_analysis(
            data=df,
            target_name=selected_target,
            time_steps=time_steps,
        )
    except Exception as exc:
        residual_error = str(exc)

st.subheader("Model Evaluation")
if metrics:
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    rmse_value = float(metrics.get("RMSE", 0.0))
    r2_value = float(metrics.get("R2 Score", metrics.get("R² Score", 0.0)))
    mae_value = float(metrics.get("MAE", 0.0))
    mape_value = float(metrics.get("MAPE (%)", 0.0))
    accuracy_value = max(0.0, 100.0 - mape_value)

    metric_a.metric("RMSE", f"{rmse_value:.3f}")
    metric_b.metric("R² Score", f"{r2_value:.3f}")
    metric_c.metric("MAE", f"{mae_value:.3f}")
    metric_d.metric("Approx. Accuracy", f"{accuracy_value:.2f}%")

    st.caption("Regression models do not have a true classification-style accuracy. The accuracy shown here is derived from 100 - MAPE.")
    if metrics_path:
        st.caption(f"Metrics source: {metrics_path}")
else:
    st.info("No saved metrics were found for this target.")

if df is not None:
    st.markdown("#### Live Residual Analysis")
    if residual_data is not None and residual_metrics is not None:
        live_a, live_b, live_c, live_d = st.columns(4)
        live_a.metric("RMSE", f"{residual_metrics['RMSE']:.3f}")
        live_b.metric("R² Score", f"{residual_metrics['R2 Score']:.3f}")
        live_c.metric("MAE", f"{residual_metrics['MAE']:.3f}")
        live_d.metric("Approx. Accuracy", f"{max(0.0, 100.0 - residual_metrics['MAPE (%)']):.2f}%")

        render_residual_plot(residual_data)
        st.dataframe(residual_data.tail(20), use_container_width=True)
        st.caption(f"Residual analysis model: {residual_model_path}")
    else:
        st.warning(f"Residual plot could not be generated: {residual_error}")

evaluation_tabs = st.tabs(["Training History", "Predictions vs Actual", "Residuals", "Scatter", "Report"])

with evaluation_tabs[0]:
    if artifact_paths["training_history"].exists():
        st.image(str(artifact_paths["training_history"]), use_container_width=True)
    else:
        st.info("Training history plot not available for this target.")

with evaluation_tabs[1]:
    if artifact_paths["predictions"].exists():
        st.image(str(artifact_paths["predictions"]), use_container_width=True)
    else:
        st.info("Predictions-vs-actual plot not available for this target.")

with evaluation_tabs[2]:
    if df is not None and residual_data is not None:
        render_residual_plot(residual_data)
    elif artifact_paths["residuals"].exists():
        st.image(str(artifact_paths["residuals"]), use_container_width=True)
    else:
        st.info("Residual plot not available for this target.")

with evaluation_tabs[3]:
    if artifact_paths["scatter"].exists():
        st.image(str(artifact_paths["scatter"]), use_container_width=True)
    else:
        st.info("Scatter plot not available for this target.")

with evaluation_tabs[4]:
    if report_text:
        st.text_area("Detailed report", value=report_text, height=400)
        if report_path:
            st.caption(f"Report source: {report_path}")
    else:
        st.info("Detailed report not available for this target.")

if df is not None:
    if st.button("Run Prediction", type="primary"):
        try:
            result_df, model_path = predict_future(
                data=df,
                target_name=selected_target,
                days=days,
                time_steps=time_steps,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        value_col = result_df.columns[1]
        st.success("Prediction completed successfully.")
        st.write(f"Model used: {model_path}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Mean", f"{result_df[value_col].mean():.3f}")
        metric_col2.metric("Min", f"{result_df[value_col].min():.3f}")
        metric_col3.metric("Max", f"{result_df[value_col].max():.3f}")

        st.markdown("#### Forecast")
        st.dataframe(result_df, use_container_width=True)
        st.line_chart(result_df.set_index("Day"))

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_bytes,
            file_name="stp_future_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV file to start forecasting.")
