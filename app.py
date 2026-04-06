# -*- coding: utf-8 -*-
"""
ICT Analyzer — Streamlit 웹 대시보드
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 분석 로직 (ict_analyzer.py 에서 가져옴)
# ─────────────────────────────────────────────────────────────────────────────
import yfinance as yf


def get_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def detect_order_blocks(df, move_threshold=0.015, lookforward=5):
    obs = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    for i in range(1, len(df) - lookforward):
        future = c[i + 1 : i + 1 + lookforward]
        if c[i] < o[i] and (future.max() - c[i]) / c[i] >= move_threshold:
            obs.append(dict(type="bullish", index=i, date=df.index[i],
                            top=float(max(o[i], c[i])), bottom=float(min(o[i], c[i])),
                            high=float(h[i]), low=float(l[i])))
        elif c[i] > o[i] and (c[i] - future.min()) / c[i] >= move_threshold:
            obs.append(dict(type="bearish", index=i, date=df.index[i],
                            top=float(max(o[i], c[i])), bottom=float(min(o[i], c[i])),
                            high=float(h[i]), low=float(l[i])))
    return obs


def detect_fvg(df, min_gap_pct=0.001):
    fvgs = []
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    for i in range(1, len(df) - 1):
        if l[i + 1] > h[i - 1] and (l[i + 1] - h[i - 1]) / c[i] >= min_gap_pct:
            fvgs.append(dict(type="bullish", index=i, date=df.index[i - 1],
                             top=float(l[i + 1]), bottom=float(h[i - 1]),
                             mid=float((l[i + 1] + h[i - 1]) / 2)))
        elif h[i + 1] < l[i - 1] and (l[i - 1] - h[i + 1]) / c[i] >= min_gap_pct:
            fvgs.append(dict(type="bearish", index=i, date=df.index[i - 1],
                             top=float(l[i - 1]), bottom=float(h[i + 1]),
                             mid=float((l[i - 1] + h[i + 1]) / 2)))
    return fvgs


def detect_liquidity(df, window=5):
    h, l = df["High"].values, df["Low"].values
    swing_highs, swing_lows = [], []
    for i in range(window, len(df) - window):
        if h[i] == h[i - window : i + window + 1].max() and h[i] > h[i - 1] and h[i] > h[i + 1]:
            swing_highs.append(dict(index=i, date=df.index[i], price=float(h[i])))
        if l[i] == l[i - window : i + window + 1].min() and l[i] < l[i - 1] and l[i] < l[i + 1]:
            swing_lows.append(dict(index=i, date=df.index[i], price=float(l[i])))
    return swing_highs, swing_lows


def calculate_trade_setup(df, obs, fvgs, swing_highs, swing_lows, lookback=20):
    current_price = float(df["Close"].iloc[-1])
    n = len(df)
    setups = []

    lo, hi = current_price * 0.90, current_price * 1.10

    bull_obs  = [x for x in obs  if x["type"] == "bullish" and x["index"] > n - lookback and lo < x["top"]    < current_price]
    bear_obs  = [x for x in obs  if x["type"] == "bearish" and x["index"] > n - lookback and current_price < x["bottom"] < hi]
    bull_fvgs = [x for x in fvgs if x["type"] == "bullish" and x["index"] > n - lookback and lo < x["top"]    < current_price]
    bear_fvgs = [x for x in fvgs if x["type"] == "bearish" and x["index"] > n - lookback and current_price < x["bottom"] < hi]

    highs_above = sorted([h for h in swing_highs if h["price"] > current_price], key=lambda x: x["price"])
    lows_below  = sorted([l for l in swing_lows  if l["price"] < current_price], key=lambda x: x["price"], reverse=True)

    # LONG
    entry_src = None
    if bull_obs:
        best = max(bull_obs, key=lambda x: x["top"])
        entry, sl, entry_src = best["top"], best["low"] * 0.999, f"Bullish OB ({best['date'].date()})"
    elif bull_fvgs:
        best = max(bull_fvgs, key=lambda x: x["top"])
        entry, sl, entry_src = best["mid"], best["bottom"] * 0.998, f"Bullish FVG ({best['date'].date()})"
    if entry_src:
        tp = highs_above[0]["price"] if highs_above else current_price * 1.05
        risk = entry - sl
        setups.append(dict(direction="LONG", entry=round(entry, 4), stop_loss=round(sl, 4),
                           take_profit=round(tp, 4), rr_ratio=round((tp - entry) / risk, 2) if risk > 0 else 0,
                           source=entry_src))

    # SHORT
    entry_src = None
    if bear_obs:
        best = min(bear_obs, key=lambda x: x["bottom"])
        entry, sl, entry_src = best["bottom"], best["high"] * 1.001, f"Bearish OB ({best['date'].date()})"
    elif bear_fvgs:
        best = min(bear_fvgs, key=lambda x: x["bottom"])
        entry, sl, entry_src = best["mid"], best["top"] * 1.002, f"Bearish FVG ({best['date'].date()})"
    if entry_src:
        tp = lows_below[0]["price"] if lows_below else current_price * 0.95
        risk = sl - entry
        setups.append(dict(direction="SHORT", entry=round(entry, 4), stop_loss=round(sl, 4),
                           take_profit=round(tp, 4), rr_ratio=round((entry - tp) / risk, 2) if risk > 0 else 0,
                           source=entry_src))
    return setups


# ─────────────────────────────────────────────────────────────────────────────
# 차트 빌더
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_BARS = 80

def build_chart(df, obs, fvgs, swing_highs, swing_lows, trade_setups, ticker):
    n = len(df)
    cutoff   = max(0, n - DISPLAY_BARS)
    df_view  = df.iloc[cutoff:]
    last_dt  = df_view.index[-1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # ── 캔들 ──
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view["Open"], high=df_view["High"],
        low=df_view["Low"],   close=df_view["Close"],
        name="Price",
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── 거래량 ──
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df_view["Close"], df_view["Open"])]
    fig.add_trace(go.Bar(
        x=df_view.index, y=df_view["Volume"],
        marker_color=vol_colors, opacity=0.55, name="Volume",
    ), row=2, col=1)

    # ── Order Blocks ──
    for ob in obs:
        if ob["index"] < cutoff:
            continue
        bull = ob["type"] == "bullish"
        fill   = "rgba(30,144,255,0.15)"   if bull else "rgba(255,69,58,0.15)"
        border = "rgba(30,144,255,0.80)"   if bull else "rgba(255,69,58,0.80)"
        label  = "Bull OB"                 if bull else "Bear OB"
        aly    = ob["bottom"]              if bull else ob["top"]

        fig.add_shape(type="rect",
            x0=ob["date"], x1=last_dt,
            y0=ob["bottom"], y1=ob["top"],
            fillcolor=fill, line=dict(color=border, width=1),
            row=1, col=1)
        fig.add_annotation(x=ob["date"], y=aly, text=f"  {label}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color=border), row=1, col=1)

    # ── FVG ──
    for fvg in fvgs:
        if fvg["index"] < cutoff:
            continue
        bull = fvg["type"] == "bullish"
        fill   = "rgba(0,230,118,0.12)"  if bull else "rgba(255,183,0,0.12)"
        border = "rgba(0,200,100,0.70)"  if bull else "rgba(200,140,0,0.70)"
        label  = "Bull FVG"              if bull else "Bear FVG"

        fig.add_shape(type="rect",
            x0=fvg["date"], x1=last_dt,
            y0=fvg["bottom"], y1=fvg["top"],
            fillcolor=fill, line=dict(color=border, width=1, dash="dot"),
            row=1, col=1)
        fig.add_annotation(x=fvg["date"], y=fvg["top"], text=f"  {label}",
            showarrow=False, xanchor="left",
            font=dict(size=8, color=border), row=1, col=1)

    # ── Liquidity ──
    vis_h = [x for x in swing_highs if x["index"] >= cutoff]
    vis_l = [x for x in swing_lows  if x["index"] >= cutoff]
    if vis_h:
        fig.add_trace(go.Scatter(
            x=[x["date"] for x in vis_h], y=[x["price"] for x in vis_h],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=11, color="#ff4d4d",
                        line=dict(color="#ff0000", width=1)),
            text=["  BSL" for _ in vis_h], textposition="middle right",
            textfont=dict(size=9, color="#ff4d4d"), name="BSL (Buy Side Liq.)",
        ), row=1, col=1)
    if vis_l:
        fig.add_trace(go.Scatter(
            x=[x["date"] for x in vis_l], y=[x["price"] for x in vis_l],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=11, color="#4da6ff",
                        line=dict(color="#0080ff", width=1)),
            text=["  SSL" for _ in vis_l], textposition="middle right",
            textfont=dict(size=9, color="#4da6ff"), name="SSL (Sell Side Liq.)",
        ), row=1, col=1)

    # ── 매매 라인 ──
    for setup in trade_setups:
        ec = "#00e676" if setup["direction"] == "LONG" else "#ff6b6b"
        fig.add_hline(y=setup["entry"],
            line=dict(color=ec, width=2),
            annotation_text=f"  Entry ({setup['direction']}): {setup['entry']}",
            annotation_font=dict(color=ec, size=11),
            annotation_position="top right", row=1, col=1)
        fig.add_hline(y=setup["stop_loss"],
            line=dict(color="#ff3333", width=1.5, dash="dash"),
            annotation_text=f"  SL: {setup['stop_loss']}",
            annotation_font=dict(color="#ff3333", size=10),
            annotation_position="bottom right", row=1, col=1)
        fig.add_hline(y=setup["take_profit"],
            line=dict(color="#00bcd4", width=1.5, dash="dash"),
            annotation_text=f"  TP: {setup['take_profit']}",
            annotation_font=dict(color="#00bcd4", size=10),
            annotation_position="top right", row=1, col=1)

    # ── 현재가 ──
    cp = float(df["Close"].iloc[-1])
    fig.add_hline(y=cp,
        line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dot"),
        annotation_text=f"  현재가: {cp:,.4f}",
        annotation_font=dict(color="rgba(255,255,255,0.7)", size=10),
        annotation_position="top left", row=1, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  —  ICT Analysis",
                   font=dict(size=18, color="#e0e0e0")),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(13,17,23,0.8)",
                    bordercolor="#333", borderwidth=1, font=dict(size=10)),
        margin=dict(l=50, r=130, t=60, b=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 페이지
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="주식 분석기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 글로벌 CSS ──
st.markdown("""
<style>
  /* Streamlit UI 요소 숨기기 */
  #MainMenu { visibility: hidden; }
  header[data-testid="stHeader"] { display: none; }
  footer { visibility: hidden; }
  .stDeployButton { display: none; }

  /* 배경 */
  .stApp { background-color: #0d1117; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* 헤더 */
  .ict-header {
    text-align: center;
    padding: 28px 0 8px;
  }
  .ict-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #00e676, #00bcd4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .ict-header p {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 6px;
  }

  /* 메트릭 카드 */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-card .label {
    color: #8b949e;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .metric-card .value {
    color: #e6edf3;
    font-size: 1.55rem;
    font-weight: 700;
  }
  .metric-card .sub {
    color: #8b949e;
    font-size: 0.78rem;
    margin-top: 4px;
  }

  /* 시나리오 카드 */
  .scenario-long {
    background: linear-gradient(135deg, #0d2b1a, #0d1117);
    border: 1px solid #00e676;
    border-radius: 12px;
    padding: 20px 24px;
  }
  .scenario-short {
    background: linear-gradient(135deg, #2b0d0d, #0d1117);
    border: 1px solid #ff6b6b;
    border-radius: 12px;
    padding: 20px 24px;
  }
  .scenario-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 14px;
  }
  .scenario-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.92rem;
  }
  .scenario-row:last-child { border-bottom: none; }
  .scenario-row .s-label { color: #8b949e; }
  .scenario-row .s-value { color: #e6edf3; font-weight: 600; font-family: monospace; }
  .entry-val  { color: #00e676 !important; }
  .sl-val     { color: #ff4d4d !important; }
  .tp-val     { color: #00bcd4 !important; }
  .rr-val     { color: #ffd600 !important; }

  /* 입력창 스타일 */
  .stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 1rem !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #00e676 !important;
    box-shadow: 0 0 0 2px rgba(0,230,118,0.15) !important;
  }

  /* 버튼 */
  .stButton > button {
    background: linear-gradient(90deg, #00897b, #00acc1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    height: 46px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* selectbox */
  .stSelectbox > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
  }

  /* 섹션 구분선 */
  .section-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 24px 0;
  }

  /* 태그 뱃지 */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
  }
  .badge-ob  { background:#1e3a5f; color:#4da6ff; }
  .badge-fvg { background:#1e3d2a; color:#00e676; }
  .badge-liq { background:#3d2a1e; color:#ffb74d; }

  /* 없음 메시지 */
  .no-setup {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: #8b949e;
    font-size: 0.92rem;
  }

  /* 레이아웃 */
  .block-container { padding-top: 0 !important; max-width: 1400px !important; }
</style>
""", unsafe_allow_html=True)


# ── 헤더 ──
st.markdown("""
<div class="ict-header">
  <h1>📈 주식 분석기</h1>
  <p>매입가 · 손절가 · 목표가 자동 분석</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── 입력 영역 ──
col_t, col_p, col_b = st.columns([3, 2, 1.2])

with col_t:
    ticker_input = st.text_input(
        "티커 심볼",
        placeholder="예: AAPL · NVDA · SPY · 005930.KS",
        label_visibility="collapsed",
    )

with col_p:
    period_map = {
        "1개월": "1mo", "3개월": "3mo",
        "6개월 (기본)": "6mo", "1년": "1y", "2년": "2y",
    }
    period_label = st.selectbox(
        "기간", list(period_map.keys()),
        index=2, label_visibility="collapsed",
    )

with col_b:
    analyze_btn = st.button("🔍 분석")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ── 분석 실행 ──
if analyze_btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    period = period_map[period_label]

    with st.spinner(f"**{ticker}** 데이터 수집 및 분석 중…"):
        try:
            df = get_data(ticker, period)
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            st.stop()

    if df.empty or len(df) < 20:
        st.error(f"**{ticker}** 의 데이터를 찾을 수 없습니다. 티커를 다시 확인해주세요.")
        st.stop()

    obs                     = detect_order_blocks(df)
    fvgs                    = detect_fvg(df)
    swing_highs, swing_lows = detect_liquidity(df)
    trade_setups            = calculate_trade_setup(df, obs, fvgs, swing_highs, swing_lows)

    current_price  = float(df["Close"].iloc[-1])
    prev_price     = float(df["Close"].iloc[-2])
    price_chg      = current_price - prev_price
    price_chg_pct  = price_chg / prev_price * 100
    chg_color      = "#00e676" if price_chg >= 0 else "#ef5350"
    chg_sign       = "+" if price_chg >= 0 else ""

    n = len(df)
    r_obs  = [x for x in obs  if x["index"] > n - DISPLAY_BARS]
    r_fvgs = [x for x in fvgs if x["index"] > n - DISPLAY_BARS]

    # ── 상단 메트릭 ──
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("현재가", f"{current_price:,.2f}",
         f'<span style="color:{chg_color}">{chg_sign}{price_chg:.2f} ({chg_sign}{price_chg_pct:.2f}%)</span>'),
        ("Order Block", f"{len(r_obs)}",
         f"최근 {DISPLAY_BARS}봉 기준"),
        ("Fair Value Gap", f"{len(r_fvgs)}",
         f"최근 {DISPLAY_BARS}봉 기준"),
        ("스윙 고점 (BSL)", f"{len(swing_highs)}", "Buy Side Liquidity"),
        ("스윙 저점 (SSL)", f"{len(swing_lows)}",  "Sell Side Liquidity"),
    ]
    for col, (label, val, sub) in zip([m1, m2, m3, m4, m5], metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 매매 시나리오 ──
    st.markdown("### 매매 시나리오")

    if trade_setups:
        s_cols = st.columns(len(trade_setups))
        for col, s in zip(s_cols, trade_setups):
            if s["direction"] == "LONG":
                card_cls = "scenario-long"
                icon, title_color = "▲", "#00e676"
            else:
                card_cls = "scenario-short"
                icon, title_color = "▼", "#ff6b6b"

            risk = abs(s["entry"] - s["stop_loss"])
            reward = abs(s["take_profit"] - s["entry"])
            rr_bar_width = min(int(reward / (risk + reward) * 100), 100) if (risk + reward) > 0 else 50

            col.markdown(f"""
            <div class="{card_cls}">
              <div class="scenario-title" style="color:{title_color}">
                {icon} {s['direction']} &nbsp;
                <span style="font-size:0.78rem; color:#8b949e; font-weight:400">{s['source']}</span>
              </div>
              <div class="scenario-row">
                <span class="s-label">매입가 (Entry)</span>
                <span class="s-value entry-val">{s['entry']:,.4f}</span>
              </div>
              <div class="scenario-row">
                <span class="s-label">손절가 (SL)</span>
                <span class="s-value sl-val">{s['stop_loss']:,.4f}</span>
              </div>
              <div class="scenario-row">
                <span class="s-label">목표가 (TP)</span>
                <span class="s-value tp-val">{s['take_profit']:,.4f}</span>
              </div>
              <div class="scenario-row">
                <span class="s-label">Risk / Reward</span>
                <span class="s-value rr-val">1 : {s['rr_ratio']}</span>
              </div>
              <div style="margin-top:12px;">
                <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#8b949e; margin-bottom:4px;">
                  <span>Risk</span><span>Reward</span>
                </div>
                <div style="background:#21262d; border-radius:4px; height:6px; overflow:hidden;">
                  <div style="background:#ff4d4d; width:{100 - rr_bar_width}%; height:100%; float:left; border-radius:4px 0 0 4px;"></div>
                  <div style="background:#00bcd4; width:{rr_bar_width}%; height:100%; float:left; border-radius:0 4px 4px 0;"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-setup">
          현재 뚜렷한 매매 시나리오가 없습니다.<br>
          <span style="font-size:0.82rem">최근 OB / FVG 구간이 현재가에서 너무 멀거나 존재하지 않습니다.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── 차트 ──
    st.markdown("### 차트")
    fig = build_chart(df, obs, fvgs, swing_highs, swing_lows, trade_setups, ticker)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    # ── 범례 설명 ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("##### 차트 범례")
    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    legends = [
        ("🟦", "Bullish OB", "매수 오더블록 — 하방 지지 구간"),
        ("🟥", "Bearish OB", "매도 오더블록 — 상방 저항 구간"),
        ("🟩", "Bullish FVG", "상승 갭 구간 — 미채워진 공백"),
        ("🟧", "Bearish FVG", "하락 갭 구간 — 미채워진 공백"),
        ("🔺🔻", "BSL / SSL", "유동성 풀 — 스윙 고/저점"),
    ]
    for col, (ico, name, desc) in zip([lc1, lc2, lc3, lc4, lc5], legends):
        col.markdown(f"""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:8px;
                    padding:12px 14px; font-size:0.82rem;">
          <div style="font-weight:600; color:#e6edf3; margin-bottom:4px;">{ico} {name}</div>
          <div style="color:#8b949e;">{desc}</div>
        </div>""", unsafe_allow_html=True)

elif analyze_btn and not ticker_input.strip():
    st.warning("티커를 입력해주세요.")

else:
    # 초기 상태 안내
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color:#8b949e;">
      <div style="font-size:3rem; margin-bottom:16px;">📊</div>
      <div style="font-size:1.1rem; color:#e6edf3; margin-bottom:8px;">티커를 입력하고 분석 버튼을 눌러주세요</div>
      <div style="font-size:0.88rem;">
        미국 주식: <code style="background:#161b22; padding:2px 8px; border-radius:4px; color:#00e676;">AAPL</code>
        <code style="background:#161b22; padding:2px 8px; border-radius:4px; color:#00e676;">NVDA</code>
        <code style="background:#161b22; padding:2px 8px; border-radius:4px; color:#00e676;">SPY</code>
        &nbsp;&nbsp;
        국내 주식: <code style="background:#161b22; padding:2px 8px; border-radius:4px; color:#00bcd4;">005930.KS</code>
        <code style="background:#161b22; padding:2px 8px; border-radius:4px; color:#00bcd4;">000660.KS</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
