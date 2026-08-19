"""
Streamlit entry point for the Gold Scalper engine.

Multi-market macro model for **Gold (GC=F / XAU/USD)** using DXY, US
Treasuries (IEF), Silver, S&P 500, EUR/USD, and VIX as correlated inputs.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    push all .py files at the repo root, set entry point `app.py`.

This version uses FLAT IMPORTS so it works whether you uploaded modules
as `engine/*.py` (proper package) OR as `*.py` at the repo root.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Robust imports — try flat layout first, fall back to engine.* package.
# ---------------------------------------------------------------------------
def _try_import():
    try:
        from config import Config, MARKET_LABELS  # noqa: F401
        from data import DataSourceFactory       # noqa: F401
        from scoring import (                    # noqa: F401
            composite_score, forecasts, forecast_classify,
            market_regime, flow_meter,
        )
        from backtest import BacktestConfig, run_backtest  # noqa: F401
        from trades import run_trade_engine, TradeConfig as EngineTradeConfig  # noqa: F401
        from trade_ui import render_trade_dashboard        # noqa: F401
        return {
            "Config": Config, "MARKET_LABELS": MARKET_LABELS,
            "DataSourceFactory": DataSourceFactory,
            "composite_score": composite_score,
            "forecasts": forecasts, "forecast_classify": forecast_classify,
            "market_regime": market_regime, "flow_meter": flow_meter,
            "BacktestConfig": BacktestConfig, "run_backtest": run_backtest,
            "run_trade_engine": run_trade_engine,
            "EngineTradeConfig": EngineTradeConfig,
            "render_trade_dashboard": render_trade_dashboard,
            "layout": "flat",
        }
    except ImportError:
        from engine.config import Config, MARKET_LABELS
        from engine.data import DataSourceFactory
        from engine.scoring import (
            composite_score, forecasts, forecast_classify,
            market_regime, flow_meter,
        )
        from engine.backtest import BacktestConfig, run_backtest
        from engine.trades import run_trade_engine, TradeConfig as EngineTradeConfig
        from engine.trade_ui import render_trade_dashboard
        return {
            "Config": Config, "MARKET_LABELS": MARKET_LABELS,
            "DataSourceFactory": DataSourceFactory,
            "composite_score": composite_score,
            "forecasts": forecasts, "forecast_classify": forecast_classify,
            "market_regime": market_regime, "flow_meter": flow_meter,
            "BacktestConfig": BacktestConfig, "run_backtest": run_backtest,
            "run_trade_engine": run_trade_engine,
            "EngineTradeConfig": EngineTradeConfig,
            "render_trade_dashboard": render_trade_dashboard,
            "layout": "engine",
        }


_IMPORTS = _try_import()
Config                  = _IMPORTS["Config"]
MARKET_LABELS           = _IMPORTS["MARKET_LABELS"]
DataSourceFactory       = _IMPORTS["DataSourceFactory"]
composite_score         = _IMPORTS["composite_score"]
forecasts               = _IMPORTS["forecasts"]
forecast_classify       = _IMPORTS["forecast_classify"]
market_regime           = _IMPORTS["market_regime"]
flow_meter              = _IMPORTS["flow_meter"]
BacktestConfig          = _IMPORTS["BacktestConfig"]
run_backtest            = _IMPORTS["run_backtest"]
run_trade_engine        = _IMPORTS["run_trade_engine"]
EngineTradeConfig       = _IMPORTS["EngineTradeConfig"]
render_trade_dashboard  = _IMPORTS["render_trade_dashboard"]
_LAYOUT                 = _IMPORTS["layout"]


# -----------------------------------------------------------------------------
# Page config & theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Scalper Engine",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .stMetric > div { padding: 0.5rem 0.75rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SIGNAL_COLOURS = {
    "Strong Bullish":  "#006400",
    "Bullish":         "#2E8B57",
    "Neutral":         "#DAA520",
    "Bearish":         "#B22222",
    "Strong Bearish":  "#8B0000",
}

REGIME_COLOURS = {
    "Gold-Friendly":  "#2E8B57",
    "Gold-Hostile":   "#B22222",
    "Transition":     "#DAA520",
}


@st.cache_data(ttl=300, show_spinner="Fetching market data…")
def _fetch(source_name: str, api_key: str, interval: str, lookback_days: int):
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval=interval)
    src = DataSourceFactory.create(cfg)
    return src.fetch_all(lookback_days=lookback_days, interval=interval)


def _score_alignment_check(data):
    missing = [m for m in MARKET_LABELS if m not in data or len(data[m]) == 0]
    return missing


def _gauge(value, title, min_val=0, max_val=100, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": suffix, "font": {"size": 28}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar":  {"color": "#DAA520"},
            "steps": [
                {"range": [min_val, 40],         "color": "#f8d7da"},
                {"range": [40,         60],      "color": "#fff3cd"},
                {"range": [60,         max_val], "color": "#d4edda"},
            ],
        },
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=0))
    return fig


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
config = Config.from_sidebar()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
try:
    from data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG
except ImportError:
    from engine.data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG

INTERVAL_LABELS = {
    "1d":  "Daily (1d) — swing trading",
    "1h":  "Hourly (1h) — intraday",
    "30m": "30-min — intraday",
    "15m": "15-min — intraday / scalping",
    "5m":  "5-min — scalping",
    "1m":  "1-min — scalping (limited history)",
}

with st.spinner(f"Loading {config.interval} market data…"):
    try:
        lb = interval_lookback_days(config.interval)
        data = _fetch(config.data_source, config.twelvedata_api_key, config.interval, lb)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

missing = _score_alignment_check(data)
if missing:
    st.warning(f"Missing data for: {', '.join(missing)}. "
               "Try switching the data source in the sidebar.")
    if "gold" not in data or len(data["gold"]) == 0:
        st.error("Gold data is required.  Cannot continue.")
        st.stop()


# -----------------------------------------------------------------------------
# Compute composite + forecasts + regime + flow
# -----------------------------------------------------------------------------
with st.spinner("Computing composite score…"):
    score_result = composite_score(data, config)
    fcasts = forecasts(data, config)
    f_labels, f_conf = {}, {}
    for name, s in fcasts.items():
        lbl, conf = forecast_classify(s)
        f_labels[name] = lbl
        f_conf[name]   = conf
    regime = market_regime(score_result.per_market)
    flow   = flow_meter(
        composite   = score_result.composite,
        vix_close   = data["vix"]["Close"]   if "vix"   in data else pd.Series(15.0,  index=score_result.composite.index),
        dxy_close   = data["dxy"]["Close"]   if "dxy"   in data else pd.Series(100.0, index=score_result.composite.index),
        silver_score= score_result.per_market.get("silver", pd.Series(0.0, index=score_result.composite.index)),
    )


# -----------------------------------------------------------------------------
# Helpers (humanize bar-count to wall-clock)
# -----------------------------------------------------------------------------
_blabel = INTERVAL_CONFIG[config.interval]["bar_label"]

def _humanize(bars: int) -> str:
    if config.interval == "1d":
        return f"~{bars} days"
    if config.interval == "1h":
        return f"~{bars/24:.1f} days" if bars >= 24 else f"~{bars} hours"
    hours = bars * {"30m": 0.5, "15m": 0.25, "5m": 5/60, "1m": 1/60}.get(config.interval, 1.0)
    if hours < 1:
        return f"~{int(hours*60)} min"
    if hours < 24:
        return f"~{hours:.1f} hours"
    return f"~{hours/24:.1f} days"


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🥇 Gold Scalper — Multi-Market Macro Engine")
st.caption(
    f"Interval: **{INTERVAL_LABELS.get(config.interval, config.interval)}** · "
    f"Data: **{config.data_source}** · "
    f"Last bar: **{data['gold'].index[-1].strftime('%Y-%m-%d %H:%M')}** · "
    f"Markets loaded: **{len(data)}/{len(MARKET_LABELS)}**"
)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_live, tab_backtest, tab_trades, tab_about = st.tabs(
    ["📊  Live", "🧪  Backtest", "🎯  Trades", "ℹ️  About"]
)


# =============================================================================
# LIVE TAB
# =============================================================================
with tab_live:
    latest = score_result.composite.index[-1]
    last_score   = float(score_result.composite.loc[latest])
    last_label   = str(score_result.label.loc[latest])
    last_conf    = float(score_result.confidence.loc[latest])
    last_color   = SIGNAL_COLOURS[last_label]

    # ----- Macro context (composite + flow, compact) -----------------------
    last_flow = float(flow.loc[latest])
    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        st.markdown(
            f"""
            <div style="background:{last_color}1a;border-left:5px solid {last_color};
                        padding:10px 14px;border-radius:6px;display:flex;align-items:center;gap:14px">
              <div>
                <div style="font-size:0.7rem;color:#666;letter-spacing:0.04em">MACRO (7-MARKET COMPOSITE)</div>
                <div style="font-size:1.15rem;font-weight:700;color:{last_color};line-height:1.1">
                  {last_label}
                </div>
              </div>
              <div style="font-size:0.85rem;color:#888">score {last_score:+.1f} · conf {last_conf:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "ℹ️ Composite is the macro read.  Action signals come from the "
            "**three forecast horizons below** — pick the one that matches "
            "your holding period."
        )
    with col_b:
        st.metric("Institutional Flow", f"{last_flow:.0f}/100",
                  delta=f"{last_flow - 50:+.0f} vs neutral")
    with col_c:
        st.metric("Composite score", f"{last_score:+.1f}")
    with col_d:
        st.metric("Last bar",
                  latest.strftime('%Y-%m-%d %H:%M'),
                  help=f"Interval: {INTERVAL_LABELS.get(config.interval, config.interval)}")

    # ----- Per-market breakdown table --------------------------------------
    st.subheader("Market breakdown")
    contribs = score_result.contributions
    weights  = config.normalized_weights()
    rows = []
    for mkt, label in MARKET_LABELS.items():
        if mkt not in score_result.per_market:
            continue
        s = float(score_result.per_market[mkt].loc[latest])
        w = weights[mkt] * 100
        c = float(contribs[mkt].loc[latest])
        direction = "▼" if c < 0 else "▲"
        rows.append({
            "Market":      label,
            "Asset Score": f"{s:+.1f}",
            "Weight":      f"{w:.1f}%",
            "Contribution": f"{c:+.1f}",
            "Direction":   direction,
        })
    df_mkts = pd.DataFrame(rows).set_index("Market")

    def _color_contrib(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 20:   return "background-color: #2E8B5730"
        if v < -20:  return "background-color: #B2222230"
        return "background-color: #DAA52030"

    st.dataframe(
        df_mkts.style.map(_color_contrib, subset=["Contribution"]),
        use_container_width=True, height=290,
    )

    # ----- Forecasts + regime + flow ---------------------------------------
    st.subheader("Regime & flow")
    r1, r2 = st.columns([1, 1])
    with r1:
        reg_now = regime.loc[latest]
        st.markdown(
            f"""<div style="background:{REGIME_COLOURS.get(reg_now, '#DAA520')}1a;border-left:5px solid {REGIME_COLOURS.get(reg_now, '#DAA520')};
                        padding:10px 12px;border-radius:6px">
                  <div style="font-size:0.78rem;color:#666">Market Regime</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{REGIME_COLOURS.get(reg_now, '#DAA520')}">{reg_now}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "Gold-Friendly: DXY<−20 ∧ VIX<−10 ∧ SP500>10 · "
            "Gold-Hostile: DXY>20 ∧ VIX>10 ∧ SP500<−10 · otherwise Transition"
        )
    with r2:
        st.plotly_chart(
            _gauge(float(flow.loc[latest]), "Institutional Flow Meter"),
            use_container_width=True,
        )

    # ----- Composite history ------------------------------------------------
    st.subheader("Composite history")
    window = st.slider("Lookback (days)", 30, 730, 180, key="live_lb")
    hist = score_result.composite.tail(window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Composite",
                             line=dict(color="#DAA520", width=2)))
    for level, color, name in [
        ( 70, "#006400", "Strong Bull"),
        ( 40, "#2E8B57", "Bull"),
        (-40, "#B22222", "Bear"),
        (-70, "#8B0000", "Strong Bear"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.5,
                      annotation_text=name, annotation_position="right")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Composite Score", xaxis_title=None,
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ----- Live entry / SL / TP setup ---------------------------------------
    try:
        from indicators import atr as _atr, ema as _ema
    except ImportError:
        try:
            from engine.indicators import atr as _atr, ema as _ema
        except ImportError:
            _atr = _ema = None

    if _atr is not None and "gold" in data and len(data["gold"]) > 30:
        gold_now = data["gold"]
        atr_series = _atr(gold_now["High"], gold_now["Low"], gold_now["Close"], 14)
        price_now = float(gold_now["Close"].iloc[-1])
        atr_now   = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

        # ----- Clickable forecast cards (primary action surface) -----------
        st.markdown("##### 🎯 Forecast horizons — pick one for the entry setup")

        if "active_horizon" not in st.session_state:
            st.session_state.active_horizon = "medium"
        if not st.session_state.get("_horizon_user_picked", False):
            f_abs = {h: abs(float(fcasts[h].loc[latest])) for h in ("short", "medium", "long")}
            st.session_state.active_horizon = max(f_abs, key=f_abs.get)

        horizon_meta = {
            "short":  {"ef": config.forecasts["short"]["ema_fast"],
                       "es": config.forecasts["short"]["ema_slow"],
                       "duration": _humanize(config.forecasts["short"]["ema_slow"])},
            "medium": {"ef": config.forecasts["medium"]["ema_fast"],
                       "es": config.forecasts["medium"]["ema_slow"],
                       "duration": _humanize(config.forecasts["medium"]["ema_slow"])},
            "long":   {"ef": config.forecasts["long"]["ema_fast"],
                       "es": config.forecasts["long"]["ema_slow"],
                       "duration": _humanize(config.forecasts["long"]["ema_slow"])},
        }

        card_cols = st.columns(3)
        for col, hname in zip(card_cols, ("short", "medium", "long")):
            meta = horizon_meta[hname]
            lbl  = str(f_labels[hname].loc[latest])
            conf = float(f_conf[hname].loc[latest])
            sc   = float(fcasts[hname].loc[latest])
            c    = SIGNAL_COLOURS.get(lbl, "#888")
            is_active = (st.session_state.active_horizon == hname)
            ring = "outline: 3px solid #DAA520;" if is_active else "outline: 1px solid #ddd;"

            with col:
                st.markdown(
                    f"""
                    <div style="background:{c}1a;border-left:5px solid {c};
                                border-radius:8px;padding:14px;{ring}">
                      <div style="font-size:0.72rem;color:#666;letter-spacing:0.04em">
                        {hname.upper()} · EMA {meta['ef']}/{meta['es']} · ~{meta['duration']}
                      </div>
                      <div style="font-size:1.5rem;font-weight:700;color:{c};line-height:1.1;margin:4px 0">
                        {lbl}
                      </div>
                      <div style="font-size:0.82rem;color:#555">
                        Score <b style="color:{c}">{sc:+.1f}</b> · Conf <b>{conf:.0f}%</b>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                btn_label = "✓ Active" if is_active else f"Use {hname} for entry"
                if st.button(btn_label, key=f"pick_{hname}", use_container_width=True,
                             disabled=is_active):
                    st.session_state.active_horizon = hname
                    st.session_state._horizon_user_picked = True
                    st.rerun()

        horizon = st.session_state.active_horizon
        st.markdown("")

        # ----- Entry / SL / TP for the active horizon -----------------------
        f_label_now = str(f_labels[horizon].loc[latest])
        f_conf_now  = float(f_conf[horizon].loc[latest])
        f_score_now = float(fcasts[horizon].loc[latest])
        f_color_now = SIGNAL_COLOURS.get(f_label_now, "#888")
        hz_cfg = config.forecasts[horizon]
        f_ef, f_es = hz_cfg["ema_fast"], hz_cfg["ema_slow"]

        if f_label_now == "Bullish":
            side = 1
        elif f_label_now == "Bearish":
            side = -1
        else:
            side = 0

        # Header strip
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(
                f"""<div style="background:{f_color_now}1a;border-left:5px solid {f_color_now};
                            padding:8px 12px;border-radius:6px">
                      <div style="font-size:0.7rem;color:#666">FORECAST ({horizon.upper()})</div>
                      <div style="font-size:1.05rem;font-weight:700;color:{f_color_now}">{f_label_now}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with h2:
            st.metric("Forecast score", f"{f_score_now:+.1f}")
        with h3:
            st.metric("Forecast confidence", f"{f_conf_now:.1f}%")
        with h4:
            st.metric("EMA pair", f"{f_ef}/{f_es}",
                      help="Fast/slow EMA pair driving this forecast.")

        st.subheader(f"🎯 Live entry setup — {horizon} horizon")

        if side == 0 or atr_now <= 0:
            st.info(
                f"ℹ️ No entry on the **{horizon}** forecast right now "
                f"(label = **{f_label_now}**). Wait for a Bullish / Bearish read."
            )
        else:
            # Horizon-tuned SL/TP multipliers
            _horizon_params = {
                "short":  {"k_sl": 1.5, "k_tp": 2.5, "pb": 0.3},
                "medium": {"k_sl": 1.5, "k_tp": 3.0, "pb": 0.5},
                "long":   {"k_sl": 2.0, "k_tp": 5.0, "pb": 0.8},
            }
            params = _horizon_params[horizon]
            k_sl, k_tp, pb_frac = params["k_sl"], params["k_tp"], params["pb"]

            e_fast = float(_ema(gold_now["Close"], f_ef).iloc[-1])
            e_slow = float(_ema(gold_now["Close"], f_es).iloc[-1])
            structural_pullback = e_slow

            entry_limit = price_now - side * pb_frac * atr_now
            if side == 1 and entry_limit < structural_pullback:
                entry_limit = max(entry_limit, structural_pullback)
            if side == -1 and entry_limit > structural_pullback:
                entry_limit = min(entry_limit, structural_pullback)

            stop_dist   = k_sl * atr_now
            target_dist = k_tp * atr_now
            stop_price  = entry_limit - side * stop_dist
            target_price= entry_limit + side * target_dist
            rr          = target_dist / stop_dist

            equity       = 10_000.0
            risk_dollars = equity * 0.01
            size         = risk_dollars / stop_dist

            t_target = _humanize(int(f_es))

            side_label  = "LONG 🟢" if side == 1 else "SHORT 🔴"
            stop_color  = "#B22222"
            tgt_color   = "#006400"
            entry_color = "#1f77b4"

            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            ec1.metric("Action",        side_label, delta=f"{horizon} forecast")
            ec2.metric("Entry (limit)", f"{entry_limit:,.2f}",
                       delta=f"{abs(price_now - entry_limit):.1f} pts from close")
            ec3.metric("Stop",          f"{stop_price:,.2f}",
                       delta=f"{stop_dist:.1f} pts risk", delta_color="inverse")
            ec4.metric("Target",        f"{target_price:,.2f}",
                       delta=f"+{target_dist:.1f} pts · ~{t_target}")
            ec5.metric("Size (1% risk)", f"{size:.3f}",
                       help=f"Contracts at 1% of $10k account. Risk = ${risk_dollars:.0f} per trade.")

            st.caption(
                f"**{horizon.upper()}** forecast: **{f_label_now}** "
                f"(score {f_score_now:+.1f}, conf {f_conf_now:.1f}%) · "
                f"EMA bias: fast={e_fast:,.0f} vs slow={e_slow:,.0f} · "
                f"Entry **{entry_limit:,.2f}** · "
                f"Stop **{stop_price:,.2f}** ({stop_dist:.1f} pts) · "
                f"Target **{target_price:,.2f}** ({target_dist:.1f} pts) · "
                f"R:R **{rr:.2f}**"
            )

            # Price chart with overlays
            lookback = min(60, len(gold_now))
            chart_df = gold_now.tail(lookback)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"], high=chart_df["High"],
                low=chart_df["Low"],   close=chart_df["Close"],
                name="Gold", increasing_line_color="#DAA520",
                decreasing_line_color="#8B0000",
            ))
            ema_slow_full = _ema(gold_now["Close"], f_es).tail(lookback)
            fig.add_trace(go.Scatter(
                x=ema_slow_full.index, y=ema_slow_full.values,
                name=f"EMA {f_es}", line=dict(color="#FFA500", width=1, dash="dot"),
            ))
            for level, color, name in [
                (entry_limit,  entry_color, f"Entry {entry_limit:,.0f}"),
                (stop_price,   stop_color,  f"Stop {stop_price:,.0f}"),
                (target_price, tgt_color,   f"Target {target_price:,.0f}"),
            ]:
                fig.add_hline(y=level, line_color=color, line_width=2,
                              line_dash="dash", annotation_text=name,
                              annotation_position="right",
                              annotation_font_color=color)
            fig.update_layout(
                height=400, template="plotly_white",
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Gold", xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Live entry setup unavailable (ATR not yet warmed up or imports failed).")

    st.markdown("")

    # ----- Mini price charts ------------------------------------------------
    st.subheader("Underlying price action")
    chart_cols = st.columns(2)
    for i, (mkt, label) in enumerate(MARKET_LABELS.items()):
        if mkt not in data:
            continue
        close = data[mkt]["Close"].tail(120)
        with chart_cols[i % 2]:
            fig = go.Figure(go.Scatter(x=close.index, y=close.values, name=label,
                                       line=dict(color="#DAA520", width=1.5)))
            fig.update_layout(height=160, margin=dict(l=10, r=10, t=20, b=0),
                              title=label, template="plotly_white",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BACKTEST TAB
# =============================================================================
with tab_backtest:
    st.subheader("🧪 Historical backtest")
    st.caption(
        f"Strategy: long Gold when the model says **Bullish / Strong Bullish**, "
        f"cash otherwise. Returns are computed on **{config.interval} bars**. "
        f"No look-ahead — the signal at bar T only uses bars up to T.  "
        f"⚠️ Intraday backtests use limited history "
        f"({interval_lookback_days(config.interval)} days max)."
    )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        start = st.date_input("Start",  value=datetime(2018, 1, 1),
                              min_value=datetime(2000, 1, 1),
                              max_value=datetime.now())
    with bc2:
        end   = st.date_input("End",    value=datetime.now(),
                              min_value=datetime(2000, 1, 1),
                              max_value=datetime.now() + timedelta(days=1))
    with bc3:
        _max_hold = {"1d": 250, "1h": 500, "30m": 500, "15m": 500, "5m": 1000, "1m": 1000}.get(config.interval, 500)
        _default_hold = {"1d": 5, "1h": 12, "30m": 24, "15m": 32, "5m": 96, "1m": 240}.get(config.interval, 5)
        hold  = st.number_input(
            f"Holding period ({_blabel}s)",
            1, _max_hold, _default_hold,
            help=(
                "Forward return window in bars.  "
                "5 daily = 1 trading week.  "
                "32 15-min = 1 trading day."
            ),
        )

    bt_cfg = BacktestConfig(
        start=start.isoformat(),
        end=end.isoformat(),
        holding_period=int(hold),
    )

    with st.spinner("Running backtest…"):
        try:
            result = run_backtest(data, config, bt_cfg)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    st.markdown("##### Performance vs buy & hold")
    m = result.metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Strategy CAGR",       f"{m['Strategy CAGR %']:.2f}%",
               delta=f"{m['Strategy CAGR %'] - m['Benchmark CAGR %']:.2f}% vs BH")
    mc2.metric("Strategy Sharpe",      f"{m['Strategy Sharpe']:.2f}",
               delta=f"{m['Strategy Sharpe'] - m['Benchmark Sharpe']:.2f}")
    mc3.metric("Strategy Max DD",      f"{m['Strategy Max DD %']:.2f}%",
               delta=f"{m['Strategy Max DD %'] - m['Benchmark Max DD %']:.2f}% vs BH",
               delta_color="inverse")
    mc4.metric("Strategy Total Ret",   f"{m['Strategy Total Ret %']:.2f}%",
               delta=f"{m['Strategy Total Ret %'] - m['Benchmark Total Ret %']:.2f}% vs BH")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Strategy (long-flat)", line=dict(color="#DAA520", width=2)))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                             name="Buy & Hold Gold",      line=dict(color="#888", width=1.5, dash="dot")))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=0),
                      yaxis_title="Equity ($)", xaxis_title=None,
                      template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Per-signal statistics (forward return, %)")
    st.dataframe(result.summary, use_container_width=True)

    st.markdown("##### Last 30 signals")
    st.dataframe(
        result.signals.tail(30).iloc[::-1]
            .assign(composite=lambda d: d["composite"].round(1),
                    fwd_return=lambda d: (d["fwd_return"] * 100).round(2))
            .rename(columns={"fwd_return": "fwd_ret_%"}),
        use_container_width=True, height=380,
    )

    st.markdown("##### Score distribution & forward return")
    fig = px.scatter(
        result.signals.dropna(subset=["fwd_return"]).reset_index(),
        x="composite", y="fwd_return",
        color="signal",
        color_discrete_map=SIGNAL_COLOURS,
        labels={"composite": "Composite Score", "fwd_return": "Forward Return"},
        opacity=0.6,
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TRADES TAB
# =============================================================================
with tab_trades:
    st.subheader("🎯 Trade Engine — R-based execution")
    st.caption(
        "Pullback entries · ATR stops · structural stop blend · automatic "
        "breakeven at +1R · risk-based sizing · both directions."
    )

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        risk_pct = st.number_input(
            "Risk per trade (% equity)", 0.1, 5.0, 1.0, step=0.1,
            help="1% is the institutional default.", key="trade_risk",
        ) / 100.0
    with tc2:
        k_sl = st.number_input("Stop (× ATR)", 0.5, 5.0, 1.5, step=0.1, key="trade_k_sl")
    with tc3:
        k_tp = st.number_input("Target (× ATR)", 1.0, 10.0, 3.0, step=0.1, key="trade_k_tp")
    with tc4:
        min_rr = st.number_input("Min R:R filter", 1.0, 5.0, 1.5, step=0.1, key="trade_min_rr")

    tc5, tc6, tc7, tc8 = st.columns(4)
    with tc5:
        use_be = st.checkbox("Breakeven at +1R", value=True, key="trade_be")
    with tc6:
        swing_lb = st.number_input("Swing lookback (bars)", 0, 50, 10, step=1, key="trade_swing")
    with tc7:
        pb_frac = st.number_input("Pullback depth (× ATR)", 0.0, 2.0, 0.5, step=0.1, key="trade_pb")
    with tc8:
        entry_win = st.number_input("Entry window (bars)", 1, 20, 3, step=1, key="trade_ew")

    trade_cfg = EngineTradeConfig(
        risk_per_trade=float(risk_pct),
        k_sl=float(k_sl),
        k_tp=float(k_tp),
        min_rr=float(min_rr),
        use_breakeven_be=bool(use_be),
        swing_lookback=int(swing_lb),
        pullback_atr_frac=float(pb_frac),
        entry_window=int(entry_win),
    )

    with st.spinner("Running trade engine…"):
        try:
            trade_log = run_trade_engine(
                target=data["gold"],
                composite=score_result.composite,
                labels=score_result.label,
                confidence=score_result.confidence,
                cfg=trade_cfg,
                initial_equity=10_000.0,
            )
            render_trade_dashboard(trade_log, data["gold"], config)
        except Exception as e:
            st.error(f"Trade engine failed: {e}")
            st.exception(e)


# =============================================================================
# ABOUT TAB
# =============================================================================
with tab_about:
    st.markdown(
        """
        ### What this is
        A multi-market, weighted macro engine for forecasting the next
        directional bias of **Gold (XAU/USD)**.

        ### Market set
        | Market | Why |
        |---|---|
        | **DXY** (25%) | #1 driver of gold (negative correlation) |
        | **IEF / US Treasuries** (20%) | Yield / risk-off proxy |
        | **Silver** (15%) | Sister precious metal confirmation |
        | **S&P 500** (15%) | Equities risk-on/off |
        | **EUR/USD** (10%) | Dollar strength cross-check |
        | **VIX** (10%) | Volatility / fear gauge |
        | **Gold** (5%) | Own-momentum confirmation |

        ### Data sources
        - **yfinance** (default) — free, no key, covers all 7 markets.
        - **Twelve Data** — faster, needs a free API key (twelvedata.com).
          Swap with one dropdown in the sidebar.

        ### Deploy
        1. Push all `.py` files to the **root** of your GitHub repo.
        2. Go to **share.streamlit.io** and connect the repo.
        3. Add `app.py` as the entry point.

        ### Limits
        - Free yfinance tier is delayed 15 min and occasionally rate-limited.
        - Twelve Data free tier = 800 requests/day, 8/min.
        - Backtest assumes no commissions and no slippage.
        """
    )

    st.markdown("#### Engine modules (flat layout)")
    st.code(
        """
your-repo/
├── app.py           # Streamlit UI (Live + Backtest + Trades)
├── config.py        # weights, periods, data source, sidebar builder
├── data.py          # yfinance + Twelve Data abstraction
├── indicators.py    # EMA, RSI, ROC, ATR (Pine-equivalent)
├── scoring.py       # 7-market composite + forecasts + regime
├── backtest.py      # long-flat backtest engine
├── trades.py        # R-based trade engine
├── trade_ui.py      # trade dashboard helpers
└── requirements.txt
        """,
        language="text",
    )
