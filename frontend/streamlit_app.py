from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from components.cards import hero, inject_css, metric_card, section_title
from components.charts import anomaly_chart, category_error_chart, feature_importance_chart, forecast_chart
from components.tables import show_table


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CHART_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


st.set_page_config(page_title="RetailPulse AI", page_icon="RP", layout="wide", initial_sidebar_state="expanded")
inject_css()


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def load_metadata() -> dict[str, Any]:
    return api_get("/metadata")


@st.cache_data(ttl=120)
def load_metrics() -> dict[str, Any]:
    return api_get("/metrics")


def render_sidebar(metadata: dict[str, Any]) -> tuple[str, str, str, int]:
    st.sidebar.title("RetailPulse AI")
    st.sidebar.caption("Forecasting & Recommendation Platform")
    st.sidebar.markdown(
        "An end-to-end ML demo for retail demand forecasting, anomaly detection, product recommendations, model explainability, and AI-assisted business insights."
    )
    st.sidebar.divider()
    store_id = st.sidebar.selectbox("Store", metadata["stores"], index=0)
    categories = metadata["categories"]
    category = st.sidebar.selectbox("Category", categories, index=0)
    products = [item for item in metadata["products"] if item["category"] == category]
    labels = {f"{item['product_id']} - {item['product_name']}": item["product_id"] for item in products}
    selected_label = st.sidebar.selectbox("Product / SKU", list(labels.keys()), index=0)
    horizon = st.sidebar.slider("Forecast horizon", min_value=7, max_value=90, value=30, step=1)
    st.sidebar.divider()
    st.sidebar.markdown("Built with Python, FastAPI, Streamlit, LightGBM, scikit-learn, Docker.")
    st.sidebar.caption(f"Model: {metadata.get('model_version', 'loading')}")
    return store_id, category, labels[selected_label], horizon


def load_forecast(store_id: str, product_id: str, horizon: int) -> dict[str, Any]:
    return api_post("/forecast", {"store_id": store_id, "product_id": product_id, "horizon_days": horizon})


def load_anomalies(store_id: str, product_id: str) -> dict[str, Any]:
    return api_post("/anomalies", {"store_id": store_id, "product_id": product_id})


def load_recommendations(product_id: str, top_k: int = 5) -> dict[str, Any]:
    return api_post("/recommend", {"product_id": product_id, "top_k": top_k})


try:
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not reach the RetailPulse API at {API_BASE_URL}. Start the backend and refresh. Details: {exc}")
    st.stop()

store_id, category, product_id, horizon = render_sidebar(metadata)

with st.spinner("Loading model outputs..."):
    forecast_data = load_forecast(store_id, product_id, horizon)
    anomaly_data = load_anomalies(store_id, product_id)
    recommendation_data = load_recommendations(product_id)
    metrics_data = load_metrics()

summary = forecast_data["summary"]
product = forecast_data["product"]
hero(product["product_name"], store_id, category, product_id, metadata.get("model_version", "model-ready"))

tabs = st.tabs(["Overview", "Demand Forecast", "Recommendations", "Anomaly Detection", "Model Performance", "AI Analyst"])

with tabs[0]:
    section_title(f"{product['product_name']} in {store_id}", "Live commercial snapshot with forecast, stockout risk, and action signals.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("30-day demand", f"{summary['forecasted_30_day_demand']:,.0f}", "Forecasted units")
    with c2:
        metric_card("Avg daily", f"{summary['average_daily_demand']:,.1f}", "Expected units/day")
    with c3:
        metric_card("Stockout risk", summary["stockout_risk"], f"{summary['inventory_days_of_cover']} days cover", summary["stockout_risk"])
    with c4:
        metric_card("Inventory", f"{summary['current_inventory']:,}", "Units on hand")
    with c5:
        metric_card("Reorder", f"{summary['recommended_reorder_quantity']:,}", "Suggested units")
    section_title("Demand Pulse", "Historical sales, 30-day forecast, uncertainty range, and detected anomalies in one view.")
    with st.container(border=True):
        st.plotly_chart(
            forecast_chart(forecast_data["historical"], forecast_data["forecast"], anomaly_data["anomalies"]),
            width="stretch",
            key="overview_forecast_chart",
            config=CHART_CONFIG,
            theme=None,
        )
    left, right = st.columns([1.1, 0.9])
    with left:
        section_title("Recommended Bundles")
        show_table(recommendation_data["recommended_bundles"], "No bundle recommendations available.")
    with right:
        section_title("Recent Anomalies")
        show_table(anomaly_data["anomalies"][-6:], "No recent anomalies detected.")

with tabs[1]:
    section_title("Demand Forecast", "Expanded planning view for the selected store and SKU.")
    with st.container(border=True):
        st.plotly_chart(
            forecast_chart(forecast_data["historical"], forecast_data["forecast"], anomaly_data["anomalies"]),
            width="stretch",
            key="demand_forecast_chart",
            config=CHART_CONFIG,
            theme=None,
        )
    section_title("Forecast Table")
    st.dataframe(pd.DataFrame(forecast_data["forecast"]), width="stretch", hide_index=True, height=430)

with tabs[2]:
    section_title("Product Recommendations", "Similarity and bundle ideas generated from demand profile, category, price, and promotion behavior.")
    rec_tabs = st.tabs(["Similar Products", "Frequently Bought Together", "Bundle Candidates"])
    with rec_tabs[0]:
        show_table(recommendation_data["similar_products"], "No similar products found.")
    with rec_tabs[1]:
        show_table(recommendation_data["frequently_bought_together"], "No co-purchase candidates found.")
    with rec_tabs[2]:
        show_table(recommendation_data["recommended_bundles"], "No bundle candidates found.")

with tabs[3]:
    section_title("Anomaly Detection", "Rolling expected demand with spikes, drops, severity, and operational reasons.")
    with st.container(border=True):
        st.plotly_chart(
            anomaly_chart(anomaly_data["series"], anomaly_data["anomalies"]),
            width="stretch",
            key="anomaly_detection_chart",
            config=CHART_CONFIG,
            theme=None,
        )
    section_title("Anomaly Table")
    show_table(anomaly_data["anomalies"], "No anomalies detected for this product and store.")

with tabs[4]:
    section_title("Model Performance", "Validation metrics, recursive forecast backtest, feature importance, and category-level error profile.")
    model_metrics = metrics_data["metrics"]
    recursive_metrics = model_metrics.get("recursive_backtest", {})
    cumulative_30 = recursive_metrics.get("recursive_30_day_cumulative", {})
    daily_30 = recursive_metrics.get("recursive_30_day_daily", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE", model_metrics.get("mae", "n/a"))
    m2.metric("RMSE", model_metrics.get("rmse", "n/a"))
    m3.metric("MAPE", f"{model_metrics.get('mape', 0)}%")
    m4.metric("WAPE Accuracy", f"{model_metrics.get('forecast_accuracy', 0)}%")
    m5.metric("R2 Score", model_metrics.get("r2_score", "n/a"))

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("30-day accuracy", f"{cumulative_30.get('forecast_accuracy', 0)}%")
    r2.metric("30-day WAPE", f"{cumulative_30.get('wmape', 0)}%")
    r3.metric("Portfolio accuracy", f"{cumulative_30.get('total_demand_accuracy', 0)}%")
    r4.metric("Recursive daily WAPE", f"{daily_30.get('wmape', 0)}%")
    r5.metric("Latency", f"{summary['prediction_latency_ms']} ms")

    st.caption(
        "One-step validation measures next-day rows with known lag history. "
        "Recursive metrics replay the deployed multi-day forecast path and are better for inventory planning."
    )
    st.caption(f"Model version: {model_metrics.get('model_version')} | Training date: {model_metrics.get('training_date')} | Rows used: {model_metrics.get('rows_used_for_training'):,}")
    section_title("Feature Importance")
    with st.container(border=True):
        st.plotly_chart(
            feature_importance_chart(metrics_data["feature_importance"]),
            width="stretch",
            key="feature_importance_chart",
            config=CHART_CONFIG,
            theme=None,
        )
    section_title("Error by Category")
    with st.container(border=True):
        st.plotly_chart(
            category_error_chart(metrics_data["category_errors"]),
            width="stretch",
            key="category_error_chart",
            config=CHART_CONFIG,
            theme=None,
        )

with tabs[5]:
    section_title("AI Analyst", "Business-readable interpretation of forecast, anomaly, inventory, and recommendation outputs.")
    st.write("Generate a concise business summary from the selected forecast, anomaly, inventory, and recommendation outputs.")
    if st.button("Generate Insight", type="primary", width="content"):
        with st.spinner("Generating analyst summary..."):
            insight = api_post("/insights", {"store_id": store_id, "product_id": product_id, "horizon_days": horizon})
        st.success(f"Provider: {insight['provider']}")
        st.markdown(f"### Analyst Summary\n{insight['summary']}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Risks")
            for item in insight["risks"]:
                st.write(f"- {item}")
        with right:
            st.markdown("#### Recommended Actions")
            for item in insight["recommended_actions"]:
                st.write(f"- {item}")
    else:
        st.info("Click Generate Insight to create a business-readable recommendation. The app works with or without an LLM API key.")
