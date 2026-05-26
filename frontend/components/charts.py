from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


CHART_LAYOUT = {
    "font": dict(family="Inter, Segoe UI, Arial, sans-serif", color="#111827", size=13),
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "hoverlabel": dict(bgcolor="#111827", font_color="#ffffff", bordercolor="#111827"),
}


def _polish_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e8eef7",
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikecolor="#94a3b8",
        spikethickness=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e8eef7", zeroline=False, rangemode="tozero")
    return fig


def forecast_chart(historical: list[dict], forecast: list[dict], anomalies: list[dict] | None = None) -> go.Figure:
    hist = pd.DataFrame(historical)
    fcst = pd.DataFrame(forecast)
    horizon_days = len(fcst)
    hist["date"] = pd.to_datetime(hist["date"])
    fcst["date"] = pd.to_datetime(fcst["date"])
    visible_start = hist["date"].min()
    visible_end = fcst["date"].max()
    forecast_start = fcst["date"].min()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["units_sold"],
            mode="lines",
            name="Historical sales fill",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, .09)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["units_sold"],
            mode="lines",
            name="Historical daily sales",
            line=dict(color="#1d4ed8", width=2.4),
            hovertemplate="%{x|%b %d, %Y}<br>Sales: %{y:.0f} units<extra></extra>",
        )
    )
    band_x = pd.concat([fcst["date"], fcst["date"].iloc[::-1]], ignore_index=True)
    band_y = pd.concat([fcst["upper_bound"], fcst["lower_bound"].iloc[::-1]], ignore_index=True)
    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=band_y,
            mode="lines",
            name="Estimated range",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(14, 165, 233, .18)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fcst["date"],
            y=fcst["predicted_units"],
            mode="lines+markers",
            name="Daily forecast",
            line=dict(color="#047857", width=4, shape="spline"),
            marker=dict(size=6, color="#10b981", line=dict(color="#064e3b", width=1)),
            hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:.1f} units<extra></extra>",
        )
    )
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        if not anomaly_df.empty:
            anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])
            anomaly_df = anomaly_df[(anomaly_df["date"] >= visible_start) & (anomaly_df["date"] <= hist["date"].max())]
            merged = anomaly_df.merge(hist[["date", "units_sold"]], on="date", how="left")
            if "actual_sales" in merged:
                merged["units_sold"] = merged["units_sold"].fillna(merged["actual_sales"])
            merged = merged.dropna(subset=["units_sold"])
            if not merged.empty:
                marker_sizes = merged["severity"].map({"High": 15, "Medium": 12, "Low": 10}).fillna(10)
                marker_colors = merged["anomaly_type"].map({"Spike": "#ef4444", "Drop": "#f59e0b"}).fillna("#ef4444")
                fig.add_trace(
                    go.Scatter(
                        x=merged["date"],
                        y=merged["units_sold"],
                        mode="markers",
                        name="Anomalies",
                        marker=dict(color=marker_colors, size=marker_sizes, symbol="diamond", line=dict(color="#111827", width=1)),
                        text=merged["reason"],
                        hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:.0f}<br>%{text}<extra></extra>",
                    )
                )
    fig.add_vline(x=forecast_start, line_color="#0f766e", line_width=2, line_dash="dash", opacity=0.65)
    fig.add_annotation(
        x=forecast_start,
        y=1,
        yref="paper",
        text="Forecast starts",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=12, color="#0f766e"),
        bgcolor="rgba(255,255,255,.82)",
        bordercolor="#99f6e4",
        borderwidth=1,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        height=620,
        margin=dict(l=78, r=28, t=58, b=48),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(size=13)),
        title=dict(text=f"Historical Daily Sales and {horizon_days}-Day Daily Forecast", x=0.01, xanchor="left", font=dict(size=20)),
    )
    fig.update_xaxes(range=[visible_start, visible_end], nticks=8, tickformat="%b %Y")
    fig.update_yaxes(title="Units sold")
    return _polish_axes(fig)


def anomaly_chart(series: list[dict], anomalies: list[dict]) -> go.Figure:
    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["units_sold"],
            name="Actual sales",
            line=dict(color="#1d4ed8", width=3),
            hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:.0f} units<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["expected_sales"],
            name="Rolling expected",
            line=dict(color="#475569", width=2.5, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}<br>Expected: %{y:.1f} units<extra></extra>",
        )
    )
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        anomaly_df["date"] = pd.to_datetime(anomaly_df["date"])
        merged = anomaly_df.merge(df[["date", "units_sold"]], on="date", how="left")
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["units_sold"],
                mode="markers",
                name="Detected anomalies",
                marker=dict(color="#ef4444", size=14, symbol="x", line=dict(width=2)),
                hovertemplate="%{x|%b %d, %Y}<br>Anomaly sales: %{y:.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        **CHART_LAYOUT,
        height=590,
        margin=dict(l=78, r=28, t=58, b=48),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        title=dict(text="Anomaly Scan Against Rolling Expected Demand", x=0.01, xanchor="left", font=dict(size=20)),
    )
    fig.update_yaxes(title="Units sold")
    return _polish_axes(fig)


def feature_importance_chart(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows).head(12).sort_values("importance", ascending=True)
    fig = px.bar(df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#bae6fd", "#2563eb", "#0f766e"])
    fig.update_traces(marker_line_color="#ffffff", marker_line_width=1.2, hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>")
    fig.update_layout(
        **CHART_LAYOUT,
        height=560,
        margin=dict(l=120, r=28, t=58, b=48),
        showlegend=False,
        coloraxis_showscale=False,
        title=dict(text="Forecast Model Feature Importance", x=0.01, xanchor="left", font=dict(size=20)),
    )
    fig.update_xaxes(title="Relative importance")
    fig.update_yaxes(title="")
    return _polish_axes(fig)


def category_error_chart(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows).sort_values("mae", ascending=False)
    fig = px.bar(df, x="category", y="mae", color="mape", color_continuous_scale=["#10b981", "#f59e0b", "#dc2626"])
    fig.update_traces(marker_line_color="#ffffff", marker_line_width=1.2, hovertemplate="%{x}<br>MAE: %{y:.2f}<br>MAPE: %{marker.color:.1f}%<extra></extra>")
    fig.update_layout(
        **CHART_LAYOUT,
        height=500,
        margin=dict(l=78, r=28, t=58, b=48),
        coloraxis_colorbar_title="MAPE",
        title=dict(text="Validation Error by Product Category", x=0.01, xanchor="left", font=dict(size=20)),
    )
    fig.update_yaxes(title="MAE")
    fig.update_xaxes(title="")
    return _polish_axes(fig)
