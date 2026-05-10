from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rp-ink: #111827;
            --rp-muted: #5b6475;
            --rp-line: #dbe3ef;
            --rp-blue: #2563eb;
            --rp-teal: #0f766e;
            --rp-gold: #b45309;
            --rp-red: #b91c1c;
            --rp-green: #047857;
        }
        .stApp {
            background:
                linear-gradient(115deg, rgba(37, 99, 235, .08) 0%, transparent 34%),
                linear-gradient(245deg, rgba(15, 118, 110, .08) 0%, transparent 32%),
                linear-gradient(180deg, #f7fbff 0%, #f5f7fb 44%, #eef3f8 100%);
            color: var(--rp-ink);
        }
        .main .block-container {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #0b1220 0%, #111827 52%, #13251f 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        section[data-testid="stSidebar"] * { color: #f9fafb !important; }
        section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stSlider {
            background: rgba(255,255,255,.04);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 8px;
            padding: .45rem .55rem .55rem .55rem;
            margin-bottom: .7rem;
        }
        .rp-hero {
            position: relative;
            overflow: hidden;
            margin: .25rem 0 1.2rem 0;
            padding: 1.35rem 1.45rem;
            border: 1px solid rgba(17, 24, 39, .10);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(17,24,39,.97) 0%, rgba(18,42,65,.96) 42%, rgba(12,84,75,.94) 100%);
            box-shadow: 0 24px 70px rgba(17,24,39,.16);
        }
        .rp-hero::after {
            content: "";
            position: absolute;
            inset: auto 1rem 1rem auto;
            width: 14rem;
            height: .25rem;
            background: linear-gradient(90deg, #38bdf8, #34d399, #f59e0b);
            opacity: .9;
        }
        .rp-title {
            padding: 0;
            font-size: 2.65rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: 0;
            color: #ffffff;
        }
        .rp-subtitle {
            color: rgba(255,255,255,.78);
            font-size: 1rem;
            margin-top: .55rem;
            max-width: 58rem;
        }
        .rp-kicker {
            color: #9debd9;
            font-size: .78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: .45rem;
        }
        .rp-context {
            display: flex;
            flex-wrap: wrap;
            gap: .65rem;
            margin: 1rem 0 .2rem 0;
        }
        .rp-pill {
            border: 1px solid rgba(255,255,255,.18);
            background: rgba(255,255,255,.08);
            color: #ffffff;
            border-radius: 999px;
            padding: .42rem .72rem;
            font-size: .86rem;
            font-weight: 700;
        }
        .rp-card {
            background: #ffffff;
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            padding: 1.05rem 1rem;
            box-shadow: 0 12px 30px rgba(17,24,39,.07);
            min-height: 118px;
            position: relative;
            overflow: hidden;
        }
        .rp-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--rp-blue), var(--rp-teal), var(--rp-gold));
        }
        .rp-label { color: var(--rp-muted); font-size: .78rem; font-weight: 800; text-transform: uppercase; }
        .rp-value { color: var(--rp-ink); font-size: 1.8rem; font-weight: 850; margin-top: .25rem; }
        .rp-help { color: var(--rp-muted); font-size: .84rem; margin-top: .25rem; }
        .risk-High { color: var(--rp-red); }
        .risk-Medium { color: var(--rp-gold); }
        .risk-Low { color: var(--rp-green); }
        .rp-section-title {
            font-size: 1.18rem;
            font-weight: 850;
            color: var(--rp-ink);
            margin: 1.25rem 0 .45rem 0;
        }
        .rp-section-caption {
            color: var(--rp-muted);
            font-size: .92rem;
            margin: -.2rem 0 .75rem 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rp-line) !important;
            box-shadow: 0 16px 42px rgba(17,24,39,.07);
            background: rgba(255,255,255,.82);
        }
        div[data-testid="stPlotlyChart"] {
            border-radius: 8px;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
            background: rgba(255,255,255,.7);
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            padding: .35rem;
            box-shadow: 0 8px 24px rgba(17,24,39,.05);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: .55rem .85rem;
            font-weight: 750;
        }
        .stTabs [aria-selected="true"] {
            background: #111827;
            color: #ffffff;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(17,24,39,.06);
        }
        footer { visibility: hidden; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            padding: .95rem 1rem;
            box-shadow: 0 10px 28px rgba(17,24,39,.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "", risk: str | None = None) -> None:
    risk_class = f"risk-{risk}" if risk else ""
    st.markdown(
        f"""
        <div class="rp-card">
          <div class="rp-label">{label}</div>
          <div class="rp-value {risk_class}">{value}</div>
          <div class="rp-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(product_name: str, store_id: str, category: str, product_id: str, model_version: str) -> None:
    st.markdown(
        f"""
        <div class="rp-hero">
          <div class="rp-kicker">Retail intelligence command center</div>
          <div class="rp-title">RetailPulse AI</div>
          <div class="rp-subtitle">
            Live demand forecasting, anomaly detection, recommendation signals, inventory risk,
            and analyst-ready business summaries for retail teams.
          </div>
          <div class="rp-context">
            <span class="rp-pill">{store_id}</span>
            <span class="rp-pill">{category}</span>
            <span class="rp-pill">{product_id}</span>
            <span class="rp-pill">{product_name}</span>
            <span class="rp-pill">{model_version}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="rp-section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="rp-section-caption">{caption}</div>', unsafe_allow_html=True)
