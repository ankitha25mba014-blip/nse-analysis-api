"""
NSE Stock Analysis API  (v2.0)
================================================================================
FastAPI service for Indian (NSE) stocks. Everything uses FREE data from yfinance.

EXISTING ENDPOINTS (unchanged)
  GET /                      health check
  GET /analyze/{symbol}      technicals + fundamentals
  GET /chart/{symbol}        OHLC + indicator series
  GET /watchlist?symbols=    multi-stock snapshot

NEW ENDPOINTS (v2.0)
  GET /risk/{symbol}         volatility, Sharpe, max drawdown, beta vs Nifty, VaR, trailing returns
  GET /levels/{symbol}       pivot points, support/resistance, ATR, Stochastic, OBV, Fibonacci
  GET /signals/{symbol}      rule-based bullish/bearish score with reasons
  GET /compare?symbols=      rebased performance + correlation matrix (2-5 stocks)
  GET /market                index snapshot + top gainers/losers + market breadth
  GET /backtest/{symbol}     moving-average crossover strategy vs buy & hold
  GET /forecast/{symbol}     Monte Carlo price simulation (probabilistic, NOT a prediction)

REQUIREMENTS:  pip install fastapi uvicorn yfinance pandas numpy
RUN LOCALLY:   uvicorn nse_analysis_api:app --reload --port 8000
RENDER START:  uvicorn nse_analysis_api:app --host 0.0.0.0 --port $PORT

Educational project only - nothing here is investment advice.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NSE Stock Analysis API", version="2.0")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INDICATOR FUNCTIONS (original) ────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth (fixes fixed-position matching bug on irregular quarters)."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Fundamentals calculated from income statement + balance sheet (avoids the blocked stock.info endpoint)."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass

    return fundamentals


def get_technical_snapshot(hist):
    """Core technical calculation, shared by /analyze and /watchlist."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ── NEW: SHARED HELPERS ───────────────────────────────────────────────────

_CACHE = {}
CACHE_TTL_SECONDS = 300  # 5 min cache keeps the free Render tier fast and avoids Yahoo rate limits


def norm_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol.startswith("^") or symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def clean_index(df):
    """Make the index timezone-naive daily dates so different stocks/indices align cleanly."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def get_history(ticker: str, period: str = "2y"):
    """Cached price history. Raises 404 if the symbol has no data."""
    key = ("hist", ticker, period)
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is None or hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")
    hist = clean_index(hist)
    _CACHE[key] = (now, hist)
    return hist


def max_drawdown_pct(series):
    running_max = series.cummax()
    return float(((series / running_max) - 1).min() * 100)


def trailing_return(close, trading_days):
    if len(close) <= trading_days:
        return None
    return safe_round((close.iloc[-1] / close.iloc[-1 - trading_days] - 1) * 100)


# ── NEW: EXTRA INDICATORS ─────────────────────────────────────────────────

def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(d_period).mean()
    return k, d


def calculate_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ── EXISTING ENDPOINTS ────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API v2.0 is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """OHLC candlestick data + overlay indicator series for charting."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """Lightweight multi-stock snapshot. /watchlist?symbols=RELIANCE,TCS,INFY (max 10)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", auto_adjust=True)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 1: RISK METRICS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/risk/{symbol}")
def risk_metrics(symbol: str, risk_free_rate: float = 7.0):
    """Volatility, Sharpe ratio, max drawdown, beta vs Nifty 50, 95% VaR, trailing returns.
    risk_free_rate is in % per year (default 7, roughly Indian government bond yield)."""
    ticker = norm_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        close = hist["Close"]
        returns = close.pct_change().dropna()
        if len(returns) < 60:
            raise HTTPException(status_code=400, detail="Not enough price history for risk metrics.")

        ann_vol = returns.std() * np.sqrt(252) * 100
        years = len(returns) / 252
        cagr = ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100
        sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol else None

        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) * 100 if len(downside) > 1 else None
        sortino = (cagr - risk_free_rate) / downside_vol if downside_vol else None

        var_95 = float(np.percentile(returns, 5) * 100)
        mdd = max_drawdown_pct(close)

        beta = None
        correlation = None
        try:
            nifty = get_history("^NSEI", "2y")["Close"].pct_change().dropna()
            joined = pd.concat([returns, nifty], axis=1, join="inner").dropna()
            joined.columns = ["stock", "nifty"]
            if len(joined) > 60 and joined["nifty"].var() > 0:
                beta = joined["stock"].cov(joined["nifty"]) / joined["nifty"].var()
                correlation = joined["stock"].corr(joined["nifty"])
        except Exception:
            pass  # beta is a nice-to-have

        if ann_vol < 20:
            risk_level = "Low"
        elif ann_vol < 35:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            "symbol": ticker,
            "period": "2y",
            "annualised_return_pct": safe_round(cagr),
            "annualised_volatility_pct": safe_round(ann_vol),
            "sharpe_ratio": safe_round(sharpe),
            "sortino_ratio": safe_round(sortino),
            "max_drawdown_pct": safe_round(mdd),
            "value_at_risk_95_daily_pct": safe_round(var_95),
            "beta_vs_nifty50": safe_round(beta),
            "correlation_with_nifty50": safe_round(correlation),
            "risk_level": risk_level,
            "risk_free_rate_used_pct": risk_free_rate,
            "trailing_returns_pct": {
                "1_month": trailing_return(close, 21),
                "3_months": trailing_return(close, 63),
                "6_months": trailing_return(close, 126),
                "1_year": trailing_return(close, 252),
            },
            "note": "VaR 95% means: on 95% of days the loss was smaller than this historically.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing risk for {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 2: SUPPORT / RESISTANCE + EXTRA OSCILLATORS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/levels/{symbol}")
def price_levels(symbol: str):
    """Pivot points, swing support/resistance, Fibonacci levels, ATR, Stochastic, OBV trend."""
    ticker = norm_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        high, low, close, volume = hist["High"], hist["Low"], hist["Close"], hist["Volume"]

        h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        pivot = (h + l + c) / 3
        pivots = {
            "pivot": safe_round(pivot),
            "r1": safe_round(2 * pivot - l),
            "r2": safe_round(pivot + (h - l)),
            "s1": safe_round(2 * pivot - h),
            "s2": safe_round(pivot - (h - l)),
        }

        recent = hist.tail(60)
        swing = {
            "support_60d": safe_round(recent["Low"].min()),
            "resistance_60d": safe_round(recent["High"].max()),
        }

        yr = hist.tail(252)
        hi52, lo52 = float(yr["High"].max()), float(yr["Low"].min())
        rng = hi52 - lo52
        fib = {
            "high_52w": safe_round(hi52),
            "low_52w": safe_round(lo52),
            "fib_23_6": safe_round(hi52 - 0.236 * rng),
            "fib_38_2": safe_round(hi52 - 0.382 * rng),
            "fib_50": safe_round(hi52 - 0.5 * rng),
            "fib_61_8": safe_round(hi52 - 0.618 * rng),
        }

        atr = calculate_atr(high, low, close)
        atr_val = float(atr.iloc[-1])
        k, d = calculate_stochastic(high, low, close)
        k_val, d_val = float(k.iloc[-1]), float(d.iloc[-1])
        if k_val > 80:
            stoch_state = "overbought"
        elif k_val < 20:
            stoch_state = "oversold"
        else:
            stoch_state = "neutral"

        obv = calculate_obv(close, volume)
        obv_ma = obv.rolling(20).mean()
        obv_trend = None
        if not pd.isna(obv_ma.iloc[-1]):
            obv_trend = "accumulation (rising)" if obv.iloc[-1] > obv_ma.iloc[-1] else "distribution (falling)"

        avg_vol_20 = float(volume.tail(20).mean())
        vol_ratio = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 else None

        return {
            "symbol": ticker,
            "current_price": safe_round(c),
            "pivot_points": pivots,
            "swing_levels": swing,
            "fibonacci": fib,
            "atr_14": safe_round(atr_val),
            "atr_pct_of_price": safe_round(atr_val / c * 100),
            "suggested_stop_loss_2atr": safe_round(c - 2 * atr_val),
            "stochastic": {"percent_k": safe_round(k_val), "percent_d": safe_round(d_val), "state": stoch_state},
            "volume": {
                "obv_trend": obv_trend,
                "last_volume": int(volume.iloc[-1]),
                "avg_volume_20d": int(avg_vol_20),
                "volume_vs_avg_ratio": safe_round(vol_ratio),
            },
            "note": "Pivot points are computed from the most recent trading day's high/low/close.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing levels for {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 3: SIGNAL SCORE
# ══════════════════════════════════════════════════════════════════════════

@app.get("/signals/{symbol}")
def signal_score(symbol: str):
    """Rule-based technical score from -6 (very bearish) to +6 (very bullish), with plain-English reasons."""
    ticker = norm_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        high, low, close = hist["High"], hist["Low"], hist["Close"]

        rsi = calculate_rsi(close).iloc[-1]
        _, _, macd_hist = calculate_macd(close)
        macd_h = macd_hist.iloc[-1]
        ma50, ma200 = calculate_moving_averages(close)
        ma50_v, ma200_v = ma50.iloc[-1], ma200.iloc[-1]
        k, _ = calculate_stochastic(high, low, close)
        k_v = k.iloc[-1]
        pb, _ = calculate_bollinger_bands(close)
        pb_v = pb.iloc[-1]
        adx = calculate_adx(high, low, close).iloc[-1]
        price = close.iloc[-1]

        score = 0
        reasons = []

        def add(points, text):
            nonlocal score
            score += points
            reasons.append({"points": points, "reason": text})

        if not pd.isna(rsi):
            if rsi < 30:
                add(1, f"RSI {rsi:.0f} is oversold (possible bounce)")
            elif rsi > 70:
                add(-1, f"RSI {rsi:.0f} is overbought (possible pullback)")
            else:
                add(0, f"RSI {rsi:.0f} is in the neutral zone")
        if not pd.isna(macd_h):
            add(1 if macd_h > 0 else -1,
                "MACD histogram is positive (momentum up)" if macd_h > 0 else "MACD histogram is negative (momentum down)")
        if not pd.isna(ma50_v):
            add(1 if price > ma50_v else -1,
                "Price is above the 50-day average" if price > ma50_v else "Price is below the 50-day average")
        if not pd.isna(ma50_v) and not pd.isna(ma200_v):
            add(1 if ma50_v > ma200_v else -1,
                "50-day average is above 200-day (long-term uptrend)" if ma50_v > ma200_v
                else "50-day average is below 200-day (long-term downtrend)")
        if not pd.isna(k_v):
            if k_v < 20:
                add(1, f"Stochastic %K {k_v:.0f} is oversold")
            elif k_v > 80:
                add(-1, f"Stochastic %K {k_v:.0f} is overbought")
            else:
                add(0, f"Stochastic %K {k_v:.0f} is neutral")
        if not pd.isna(pb_v):
            if pb_v < 0.05:
                add(1, "Price is hugging the lower Bollinger Band")
            elif pb_v > 0.95:
                add(-1, "Price is hugging the upper Bollinger Band")
            else:
                add(0, "Price is inside the Bollinger Bands")

        if score >= 3:
            label = "Bullish"
        elif score >= 1:
            label = "Mildly Bullish"
        elif score == 0:
            label = "Neutral"
        elif score >= -2:
            label = "Mildly Bearish"
        else:
            label = "Bearish"

        trend_strength = None
        if not pd.isna(adx):
            trend_strength = "strong trend" if adx > 25 else ("weak / no trend" if adx < 20 else "developing trend")

        return {
            "symbol": ticker,
            "score": score,
            "score_range": [-6, 6],
            "label": label,
            "trend_strength": trend_strength,
            "adx_14": safe_round(adx),
            "reasons": reasons,
            "disclaimer": "Educational rule-based score, not investment advice.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing signals for {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 4: COMPARE STOCKS + CORRELATION
# ══════════════════════════════════════════════════════════════════════════

@app.get("/compare")
def compare_stocks(symbols: str, months: int = 12):
    """Compare 2-5 stocks: rebased performance (start = 100), return/volatility table, correlation matrix.
    Example: /compare?symbols=TCS,INFY,WIPRO&months=12"""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()][:5]
    if len(symbol_list) < 2:
        raise HTTPException(status_code=400, detail="Provide 2 to 5 symbols, e.g. ?symbols=TCS,INFY")
    months = min(max(months, 1), 24)

    try:
        closes = {}
        errors = {}
        for s in symbol_list:
            t = norm_ticker(s)
            try:
                closes[t] = get_history(t, "2y")["Close"]
            except HTTPException:
                errors[t] = "No data found"
        if len(closes) < 2:
            raise HTTPException(status_code=404, detail=f"Need at least 2 valid symbols. Problems: {errors}")

        df = pd.DataFrame(closes).dropna().tail(months * 21)
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Not enough overlapping history to compare.")

        rebased = df / df.iloc[0] * 100
        step = max(1, len(rebased) // 120)  # cap payload size
        sampled = rebased.iloc[::step]
        if sampled.index[-1] != rebased.index[-1]:
            sampled = pd.concat([sampled, rebased.tail(1)])

        returns = df.pct_change().dropna()
        corr = returns.corr().round(2).replace({np.nan: None})

        summary = []
        for t in df.columns:
            total = (df[t].iloc[-1] / df[t].iloc[0] - 1) * 100
            summary.append({
                "symbol": t,
                "start_price": safe_round(df[t].iloc[0]),
                "end_price": safe_round(df[t].iloc[-1]),
                "total_return_pct": safe_round(total),
                "annualised_volatility_pct": safe_round(returns[t].std() * np.sqrt(252) * 100),
                "max_drawdown_pct": safe_round(max_drawdown_pct(df[t])),
            })
        summary.sort(key=lambda x: x["total_return_pct"] if x["total_return_pct"] is not None else -999, reverse=True)

        return {
            "months": months,
            "symbols": list(df.columns),
            "dates": [d.strftime("%Y-%m-%d") for d in sampled.index],
            "rebased_series": {t: [safe_round(v) for v in sampled[t]] for t in sampled.columns},
            "summary": summary,
            "correlation_matrix": corr.to_dict(),
            "skipped": errors,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing stocks: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 5: MARKET OVERVIEW (indices, gainers/losers, breadth)
# ══════════════════════════════════════════════════════════════════════════

INDICES = {
    "^NSEI": "Nifty 50",
    "^BSESN": "Sensex",
    "^NSEBANK": "Bank Nifty",
    "^CNXIT": "Nifty IT",
}

MOVER_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "TATAMOTORS", "TATASTEEL", "ONGC",
    "NTPC", "POWERGRID", "BAJFINANCE", "HCLTECH", "M&M", "ADANIENT", "COALINDIA", "TECHM",
]


@app.get("/market")
def market_overview():
    """Index snapshot + top 5 gainers/losers among 30 large caps + advance/decline breadth."""
    key = ("market",)
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]

    try:
        tickers = list(INDICES.keys()) + [f"{s}.NS" for s in MOVER_UNIVERSE]
        data = yf.download(tickers, period="7d", progress=False, auto_adjust=True, threads=True)
        if data is None or data.empty:
            raise HTTPException(status_code=502, detail="Market data source returned nothing. Try again shortly.")
        closes = data["Close"].ffill()
        if len(closes) < 2:
            raise HTTPException(status_code=502, detail="Not enough recent data from the market data source.")

        last, prev = closes.iloc[-1], closes.iloc[-2]
        change = (last / prev - 1) * 100

        indices = []
        for t, name in INDICES.items():
            if t in closes.columns and not pd.isna(last[t]):
                indices.append({
                    "symbol": t, "name": name,
                    "price": safe_round(last[t]),
                    "change_pct": safe_round(change[t]),
                })

        movers = []
        for s in MOVER_UNIVERSE:
            t = f"{s}.NS"
            if t in closes.columns and not pd.isna(last[t]) and not pd.isna(change[t]):
                movers.append({"symbol": t, "price": safe_round(last[t]), "change_pct": safe_round(change[t])})
        movers.sort(key=lambda x: x["change_pct"], reverse=True)

        advancing = sum(1 for m in movers if m["change_pct"] > 0)
        declining = sum(1 for m in movers if m["change_pct"] < 0)

        result = {
            "as_of": closes.index[-1].strftime("%Y-%m-%d"),
            "indices": indices,
            "top_gainers": movers[:5],
            "top_losers": list(reversed(movers[-5:])),
            "breadth": {
                "advancing": advancing,
                "declining": declining,
                "unchanged": len(movers) - advancing - declining,
                "universe_size": len(movers),
            },
            "note": "Gainers/losers are drawn from 30 large-cap NSE stocks, not the whole market.",
        }
        _CACHE[key] = (now, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building market overview: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 6: BACKTEST (moving-average crossover vs buy & hold)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/backtest/{symbol}")
def backtest(symbol: str, fast: int = 20, slow: int = 50, years: int = 3, cost_pct: float = 0.1):
    """Backtest a long-only moving-average crossover: hold the stock while fast MA > slow MA, else stay in cash.
    cost_pct is the transaction cost per trade in % (default 0.1). Signals act on the NEXT day (no look-ahead)."""
    ticker = norm_ticker(symbol)
    fast = max(2, min(fast, 100))
    slow = max(fast + 1, min(slow, 250))
    years = min(max(years, 1), 5)

    try:
        hist = get_history(ticker, "5y")
        close = hist["Close"].tail(years * 252 + slow)
        if len(close) < slow + 30:
            raise HTTPException(status_code=400, detail="Not enough history for these settings. Lower the slow window.")

        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        valid = slow_ma.notna()
        close, fast_ma, slow_ma = close[valid], fast_ma[valid], slow_ma[valid]

        position = (fast_ma > slow_ma).astype(int).shift(1).fillna(0)
        daily_ret = close.pct_change().fillna(0)
        trade_cost = position.diff().abs().fillna(0) * (cost_pct / 100)
        strat_ret = position * daily_ret - trade_cost

        equity = (1 + strat_ret).cumprod() * 100
        buy_hold = (1 + daily_ret).cumprod() * 100

        # individual trades
        trades = []
        in_trade = False
        entry_date = entry_idx = None
        pos_values = position.values
        idx = position.index
        for i in range(len(pos_values)):
            if pos_values[i] == 1 and not in_trade:
                in_trade, entry_idx, entry_date = True, i, idx[i]
            elif pos_values[i] == 0 and in_trade:
                seg = daily_ret.iloc[entry_idx:i]
                trades.append({"entry": entry_date.strftime("%Y-%m-%d"), "exit": idx[i].strftime("%Y-%m-%d"),
                               "return_pct": safe_round(((1 + seg).prod() - 1) * 100)})
                in_trade = False
        if in_trade:
            seg = daily_ret.iloc[entry_idx:]
            trades.append({"entry": entry_date.strftime("%Y-%m-%d"), "exit": "open",
                           "return_pct": safe_round(((1 + seg).prod() - 1) * 100)})

        wins = [t for t in trades if t["return_pct"] is not None and t["return_pct"] > 0]
        yrs = len(close) / 252

        def stats(curve):
            total = curve.iloc[-1] / 100 - 1
            return {
                "total_return_pct": safe_round(total * 100),
                "cagr_pct": safe_round(((1 + total) ** (1 / yrs) - 1) * 100) if yrs > 0 else None,
                "max_drawdown_pct": safe_round(max_drawdown_pct(curve)),
            }

        step = max(1, len(equity) // 150)
        curve_idx = list(range(0, len(equity), step))
        if curve_idx[-1] != len(equity) - 1:
            curve_idx.append(len(equity) - 1)

        s_stats, b_stats = stats(equity), stats(buy_hold)
        verdict = (
            "The strategy beat buy & hold over this period."
            if s_stats["total_return_pct"] > b_stats["total_return_pct"]
            else "Buy & hold beat the strategy over this period."
        )

        return {
            "symbol": ticker,
            "settings": {"fast_ma": fast, "slow_ma": slow, "years": years, "cost_per_trade_pct": cost_pct},
            "strategy": s_stats,
            "buy_and_hold": b_stats,
            "trades_count": len(trades),
            "win_rate_pct": safe_round(len(wins) / len(trades) * 100) if trades else None,
            "currently_in_position": bool(pos_values[-1] == 1),
            "verdict": verdict,
            "trades": trades[-15:],
            "equity_curve": {
                "dates": [equity.index[i].strftime("%Y-%m-%d") for i in curve_idx],
                "strategy": [safe_round(equity.iloc[i]) for i in curve_idx],
                "buy_and_hold": [safe_round(buy_hold.iloc[i]) for i in curve_idx],
            },
            "disclaimer": "Past performance does not predict future results.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 7: MONTE CARLO FORECAST
# ══════════════════════════════════════════════════════════════════════════

@app.get("/forecast/{symbol}")
def monte_carlo_forecast(symbol: str, days: int = 30, simulations: int = 1000):
    """Monte Carlo simulation (geometric Brownian motion) using the last year of returns.
    Shows a RANGE of possible outcomes, not a prediction."""
    ticker = norm_ticker(symbol)
    days = min(max(days, 5), 90)
    simulations = min(max(simulations, 200), 3000)

    try:
        hist = get_history(ticker, "2y")
        close = hist["Close"].tail(253)
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 60:
            raise HTTPException(status_code=400, detail="Not enough history for a forecast.")

        mu, sigma = float(log_ret.mean()), float(log_ret.std())
        last_price = float(close.iloc[-1])

        rng = np.random.default_rng(42)
        shocks = rng.normal(0, 1, size=(simulations, days))
        daily = (mu - 0.5 * sigma ** 2) + sigma * shocks
        paths = last_price * np.exp(np.cumsum(daily, axis=1))

        pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        final = paths[:, -1]

        # business-day dates for the x-axis
        future_dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=days)

        return {
            "symbol": ticker,
            "current_price": safe_round(last_price),
            "days": days,
            "simulations": simulations,
            "assumptions": {
                "daily_drift_pct": safe_round(mu * 100, 4),
                "daily_volatility_pct": safe_round(sigma * 100, 3),
                "history_used": "last 1 year of daily returns",
            },
            "final_price_range": {
                "worst_case_5th_pct": safe_round(pcts[0][-1]),
                "pessimistic_25th_pct": safe_round(pcts[1][-1]),
                "median": safe_round(pcts[2][-1]),
                "optimistic_75th_pct": safe_round(pcts[3][-1]),
                "best_case_95th_pct": safe_round(pcts[4][-1]),
            },
            "probability_above_current_pct": safe_round(float((final > last_price).mean() * 100)),
            "probability_down_10pct_or_more_pct": safe_round(float((final < last_price * 0.9).mean() * 100)),
            "probability_up_10pct_or_more_pct": safe_round(float((final > last_price * 1.1).mean() * 100)),
            "fan_chart": {
                "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "p5": [safe_round(v) for v in pcts[0]],
                "p25": [safe_round(v) for v in pcts[1]],
                "p50": [safe_round(v) for v in pcts[2]],
                "p75": [safe_round(v) for v in pcts[3]],
                "p95": [safe_round(v) for v in pcts[4]],
            },
            "disclaimer": "A statistical simulation of possible outcomes, not a price prediction or advice.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forecasting {ticker}: {str(e)}")
