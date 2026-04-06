# -*- coding: utf-8 -*-
"""
ICT (Inner Circle Trader) 매매법 기반 주가 분석기
- Order Block 탐지
- FVG (Fair Value Gap) 탐지
- Liquidity 영역 (스윙 고/저점)
- 매입가 / 손절가 / 목표가 자동 계산
- Plotly 시각화
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────
# 1. 데이터 수집
# ──────────────────────────────────────────────────────────────

def get_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    # yfinance 최신 버전 MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ──────────────────────────────────────────────────────────────
# 2. Order Block 탐지
# ──────────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame,
                        move_threshold: float = 0.015,
                        lookforward: int = 5) -> list[dict]:
    """
    Bullish OB : 강한 상승 직전의 마지막 음봉 (매수 주문이 쌓인 구간)
    Bearish OB : 강한 하락 직전의 마지막 양봉 (매도 주문이 쌓인 구간)
    move_threshold : OB로 인정할 최소 가격 이동 비율 (기본 1.5%)
    """
    obs: list[dict] = []
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    for i in range(1, n - lookforward):
        future = c[i + 1 : i + 1 + lookforward]

        # ── Bullish OB : 음봉 ──
        if c[i] < o[i]:
            move_up = (future.max() - c[i]) / c[i]
            if move_up >= move_threshold:
                obs.append(dict(
                    type="bullish",
                    index=i,
                    date=df.index[i],
                    top=float(max(o[i], c[i])),
                    bottom=float(min(o[i], c[i])),
                    high=float(h[i]),
                    low=float(l[i]),
                ))

        # ── Bearish OB : 양봉 ──
        elif c[i] > o[i]:
            move_dn = (c[i] - future.min()) / c[i]
            if move_dn >= move_threshold:
                obs.append(dict(
                    type="bearish",
                    index=i,
                    date=df.index[i],
                    top=float(max(o[i], c[i])),
                    bottom=float(min(o[i], c[i])),
                    high=float(h[i]),
                    low=float(l[i]),
                ))

    return obs


# ──────────────────────────────────────────────────────────────
# 3. FVG 탐지
# ──────────────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.001) -> list[dict]:
    """
    Fair Value Gap : 3개 캔들 패턴에서 발생하는 가격 공백 구간
    Bullish FVG : candle[i-1].high < candle[i+1].low  → 하방 지지 갭
    Bearish FVG : candle[i-1].low  > candle[i+1].high → 상방 저항 갭
    """
    fvgs: list[dict] = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    for i in range(1, len(df) - 1):
        # Bullish FVG
        if l[i + 1] > h[i - 1]:
            gap_pct = (l[i + 1] - h[i - 1]) / c[i]
            if gap_pct >= min_gap_pct:
                fvgs.append(dict(
                    type="bullish",
                    index=i,
                    date=df.index[i - 1],
                    top=float(l[i + 1]),
                    bottom=float(h[i - 1]),
                    mid=float((l[i + 1] + h[i - 1]) / 2),
                ))

        # Bearish FVG
        elif h[i + 1] < l[i - 1]:
            gap_pct = (l[i - 1] - h[i + 1]) / c[i]
            if gap_pct >= min_gap_pct:
                fvgs.append(dict(
                    type="bearish",
                    index=i,
                    date=df.index[i - 1],
                    top=float(l[i - 1]),
                    bottom=float(h[i + 1]),
                    mid=float((l[i - 1] + h[i + 1]) / 2),
                ))

    return fvgs


# ──────────────────────────────────────────────────────────────
# 4. Liquidity 탐지 (스윙 고/저점)
# ──────────────────────────────────────────────────────────────

def detect_liquidity(df: pd.DataFrame, window: int = 5) -> tuple[list[dict], list[dict]]:
    """
    스윙 고점 : 주변 window 캔들 중 가장 높은 고점 → 위쪽 유동성 풀
    스윙 저점 : 주변 window 캔들 중 가장 낮은 저점 → 아래쪽 유동성 풀
    """
    highs_arr = df["High"].values
    lows_arr  = df["Low"].values
    swing_highs: list[dict] = []
    swing_lows:  list[dict] = []

    for i in range(window, len(df) - window):
        local_h = highs_arr[i - window : i + window + 1]
        local_l = lows_arr [i - window : i + window + 1]

        if highs_arr[i] == local_h.max() and highs_arr[i] > highs_arr[i - 1] and highs_arr[i] > highs_arr[i + 1]:
            swing_highs.append(dict(index=i, date=df.index[i], price=float(highs_arr[i])))

        if lows_arr[i] == local_l.min() and lows_arr[i] < lows_arr[i - 1] and lows_arr[i] < lows_arr[i + 1]:
            swing_lows.append(dict(index=i, date=df.index[i], price=float(lows_arr[i])))

    return swing_highs, swing_lows


# ──────────────────────────────────────────────────────────────
# 5. 매매 시나리오 계산
# ──────────────────────────────────────────────────────────────

def calculate_trade_setup(df: pd.DataFrame,
                          obs: list[dict],
                          fvgs: list[dict],
                          swing_highs: list[dict],
                          swing_lows: list[dict],
                          lookback: int = 20) -> list[dict]:
    """
    최근 lookback 캔들 내 OB / FVG 기반으로 매수·매도 시나리오 구성
    """
    current_price = float(df["Close"].iloc[-1])
    n = len(df)
    setups: list[dict] = []

    # 현재가 기준 ±10% 이내 + 최근 lookback 봉 이내
    lo, hi = current_price * 0.90, current_price * 1.10

    bull_obs  = [x for x in obs  if x["type"] == "bullish" and x["index"] > n - lookback and lo < x["top"]    < current_price]
    bear_obs  = [x for x in obs  if x["type"] == "bearish" and x["index"] > n - lookback and current_price < x["bottom"] < hi]
    bull_fvgs = [x for x in fvgs if x["type"] == "bullish" and x["index"] > n - lookback and lo < x["top"]    < current_price]
    bear_fvgs = [x for x in fvgs if x["type"] == "bearish" and x["index"] > n - lookback and current_price < x["bottom"] < hi]

    highs_above = sorted([h for h in swing_highs if h["price"] > current_price], key=lambda x: x["price"])
    lows_below  = sorted([l for l in swing_lows  if l["price"] < current_price], key=lambda x: x["price"], reverse=True)

    # ── LONG 시나리오 ──
    entry_src = None
    if bull_obs:
        best = max(bull_obs, key=lambda x: x["top"])
        entry   = best["top"]
        sl      = best["low"] * 0.999
        entry_src = f"Bullish OB ({best['date'].date()})"
    elif bull_fvgs:
        best = max(bull_fvgs, key=lambda x: x["top"])
        entry   = best["mid"]
        sl      = best["bottom"] * 0.998
        entry_src = f"Bullish FVG ({best['date'].date()})"

    if entry_src:
        tp   = highs_above[0]["price"] if highs_above else current_price * 1.05
        risk = entry - sl
        rr   = round((tp - entry) / risk, 2) if risk > 0 else 0
        setups.append(dict(direction="LONG", entry=round(entry, 4),
                           stop_loss=round(sl, 4), take_profit=round(tp, 4),
                           rr_ratio=rr, source=entry_src))

    # ── SHORT 시나리오 ──
    entry_src = None
    if bear_obs:
        best = min(bear_obs, key=lambda x: x["bottom"])
        entry   = best["bottom"]
        sl      = best["high"] * 1.001
        entry_src = f"Bearish OB ({best['date'].date()})"
    elif bear_fvgs:
        best = min(bear_fvgs, key=lambda x: x["bottom"])
        entry   = best["mid"]
        sl      = best["top"] * 1.002
        entry_src = f"Bearish FVG ({best['date'].date()})"

    if entry_src:
        tp   = lows_below[0]["price"] if lows_below else current_price * 0.95
        risk = sl - entry
        rr   = round((entry - tp) / risk, 2) if risk > 0 else 0
        setups.append(dict(direction="SHORT", entry=round(entry, 4),
                           stop_loss=round(sl, 4), take_profit=round(tp, 4),
                           rr_ratio=rr, source=entry_src))

    return setups


# ──────────────────────────────────────────────────────────────
# 6. Plotly 차트
# ──────────────────────────────────────────────────────────────

DISPLAY_BARS = 80   # 차트에 표시할 최근 봉 수

def plot_chart(df: pd.DataFrame,
               obs: list[dict],
               fvgs: list[dict],
               swing_highs: list[dict],
               swing_lows: list[dict],
               trade_setups: list[dict],
               ticker: str) -> None:

    n = len(df)
    cutoff = max(0, n - DISPLAY_BARS)
    df_view = df.iloc[cutoff:]
    last_date = df_view.index[-1]
    first_date = df_view.index[0]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # ── 캔들 차트 ──
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view["Open"], high=df_view["High"],
        low=df_view["Low"],   close=df_view["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── 거래량 ──
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df_view["Close"], df_view["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df_view.index, y=df_view["Volume"],
        marker_color=vol_colors, opacity=0.6, name="Volume",
    ), row=2, col=1)

    # ── Order Blocks ──
    for ob in obs:
        if ob["index"] < cutoff:
            continue
        if ob["type"] == "bullish":
            fill   = "rgba(30,144,255,0.18)"
            border = "rgba(30,144,255,0.85)"
            label  = "Bull OB"
            ly     = ob["bottom"]
        else:
            fill   = "rgba(255,69,58,0.18)"
            border = "rgba(255,69,58,0.85)"
            label  = "Bear OB"
            ly     = ob["top"]

        fig.add_shape(type="rect",
            x0=ob["date"], x1=last_date,
            y0=ob["bottom"], y1=ob["top"],
            fillcolor=fill,
            line=dict(color=border, width=1),
            row=1, col=1)

        fig.add_annotation(
            x=ob["date"], y=ly, text=f"  {label}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color=border),
            row=1, col=1)

    # ── FVG ──
    for fvg in fvgs:
        if fvg["index"] < cutoff:
            continue
        if fvg["type"] == "bullish":
            fill   = "rgba(0,230,118,0.15)"
            border = "rgba(0,200,100,0.7)"
            label  = "Bull FVG"
        else:
            fill   = "rgba(255,183,0,0.15)"
            border = "rgba(200,140,0,0.7)"
            label  = "Bear FVG"

        fig.add_shape(type="rect",
            x0=fvg["date"], x1=last_date,
            y0=fvg["bottom"], y1=fvg["top"],
            fillcolor=fill,
            line=dict(color=border, width=1, dash="dot"),
            row=1, col=1)

        fig.add_annotation(
            x=fvg["date"], y=fvg["top"], text=f"  {label}",
            showarrow=False, xanchor="left",
            font=dict(size=8, color=border),
            row=1, col=1)

    # ── Liquidity: 스윙 고점 ──
    vis_highs = [h for h in swing_highs if h["index"] >= cutoff]
    if vis_highs:
        fig.add_trace(go.Scatter(
            x=[h["date"] for h in vis_highs],
            y=[h["price"] for h in vis_highs],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=11, color="#ff4d4d",
                        line=dict(color="#ff0000", width=1)),
            text=["  BSL" for _ in vis_highs],  # Buy Side Liquidity
            textposition="middle right",
            textfont=dict(size=9, color="#ff4d4d"),
            name="Liquidity High (BSL)",
        ), row=1, col=1)

    # ── Liquidity: 스윙 저점 ──
    vis_lows = [l for l in swing_lows if l["index"] >= cutoff]
    if vis_lows:
        fig.add_trace(go.Scatter(
            x=[l["date"] for l in vis_lows],
            y=[l["price"] for l in vis_lows],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=11, color="#4da6ff",
                        line=dict(color="#0080ff", width=1)),
            text=["  SSL" for _ in vis_lows],  # Sell Side Liquidity
            textposition="middle right",
            textfont=dict(size=9, color="#4da6ff"),
            name="Liquidity Low (SSL)",
        ), row=1, col=1)

    # ── 매매 라인 ──
    for setup in trade_setups:
        if setup["direction"] == "LONG":
            entry_color = "#00e676"
        else:
            entry_color = "#ff6b6b"

        # Entry
        fig.add_hline(y=setup["entry"],
            line=dict(color=entry_color, width=2, dash="solid"),
            annotation_text=f"  Entry ({setup['direction']}): {setup['entry']}",
            annotation_font=dict(color=entry_color, size=11),
            annotation_position="top right",
            row=1, col=1)

        # Stop Loss
        fig.add_hline(y=setup["stop_loss"],
            line=dict(color="#ff3333", width=1.5, dash="dash"),
            annotation_text=f"  SL: {setup['stop_loss']}",
            annotation_font=dict(color="#ff3333", size=10),
            annotation_position="bottom right",
            row=1, col=1)

        # Take Profit
        fig.add_hline(y=setup["take_profit"],
            line=dict(color="#00bcd4", width=1.5, dash="dash"),
            annotation_text=f"  TP: {setup['take_profit']}",
            annotation_font=dict(color="#00bcd4", size=10),
            annotation_position="top right",
            row=1, col=1)

    # ── 현재가 ──
    current_price = float(df["Close"].iloc[-1])
    fig.add_hline(y=current_price,
        line=dict(color="#ffffff", width=1, dash="dot"),
        annotation_text=f"  현재가: {current_price:.4f}",
        annotation_font=dict(color="#ffffff", size=10),
        annotation_position="top left",
        row=1, col=1)

    # ── 레이아웃 ──
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> — ICT 분석  (Order Block · FVG · Liquidity)",
            font=dict(size=17, color="#ffffff"),
        ),
        template="plotly_dark",
        height=820,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(20,20,30,0.7)",
            bordercolor="#555", borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=120, t=60, b=40),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                     zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                     zeroline=False)

    fig.show()


# ──────────────────────────────────────────────────────────────
# 7. 메인 분석 루틴
# ──────────────────────────────────────────────────────────────

def analyze(ticker: str) -> None:
    ticker = ticker.strip().upper()
    print(f"\n[ {ticker} ] 데이터 수집 중 …")

    df = get_data(ticker)
    if df.empty or len(df) < 20:
        print(f"  데이터를 가져올 수 없습니다: {ticker}")
        print("  ▶ 미국 주식: AAPL / 국내 주식: 005930.KS / ETF: SPY")
        return

    current_price = float(df["Close"].iloc[-1])
    period_start  = df.index[0].date()
    period_end    = df.index[-1].date()

    print(f"  기간: {period_start} ~ {period_end}  ({len(df)} 봉)")
    print(f"  현재가: {current_price:,.4f}")

    obs           = detect_order_blocks(df)
    fvgs          = detect_fvg(df)
    swing_highs, swing_lows = detect_liquidity(df)
    trade_setups  = calculate_trade_setup(df, obs, fvgs, swing_highs, swing_lows)

    # 최근 통계
    r_obs  = [x for x in obs  if x["index"] > len(df) - DISPLAY_BARS]
    r_fvgs = [x for x in fvgs if x["index"] > len(df) - DISPLAY_BARS]

    print()
    print("─" * 55)
    print(f"  Order Block  : 전체 {len(obs):>4}개  │  최근 {DISPLAY_BARS}봉 내 {len(r_obs)}개")
    print(f"  FVG          : 전체 {len(fvgs):>4}개  │  최근 {DISPLAY_BARS}봉 내 {len(r_fvgs)}개")
    print(f"  스윙 고점    : {len(swing_highs)}개  │  스윙 저점 : {len(swing_lows)}개")
    print("─" * 55)

    if trade_setups:
        print()
        print("  ★ 매매 시나리오")
        for s in trade_setups:
            arrow = "▲" if s["direction"] == "LONG" else "▼"
            print()
            print(f"  {arrow} {s['direction']}  |  근거: {s['source']}")
            print(f"     매입가   : {s['entry']:>12,.4f}")
            print(f"     손절가   : {s['stop_loss']:>12,.4f}")
            print(f"     목표가   : {s['take_profit']:>12,.4f}")
            print(f"     Risk/Reward : 1 : {s['rr_ratio']}")
    else:
        print()
        print("  현재 뚜렷한 매매 시나리오 없음")
        print("  (최근 OB/FVG 구간이 현재가에서 너무 멀거나 존재하지 않음)")

    print()
    print("─" * 55)
    print("  차트를 여는 중 …")
    print()

    plot_chart(df, obs, fvgs, swing_highs, swing_lows, trade_setups, ticker)


# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Windows 콘솔 UTF-8 출력 설정
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("=" * 55)
    print("  ICT 분석기  (Order Block · FVG · Liquidity)")
    print("=" * 55)
    print()
    print("  예시 티커:")
    print("    미국 주식 : AAPL  TSLA  NVDA  SPY  QQQ")
    print("    국내 주식 : 005930.KS  000660.KS  035420.KS")
    print()

    while True:
        raw = input("티커를 입력하세요 (종료: q): ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            print("종료합니다.")
            break
        analyze(raw)
        print()
