import math
from datetime import datetime, time as clock_time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_cookies_manager_ext import EncryptedCookieManager
from supabase import create_client


st.set_page_config(
    page_title="Folsom Trade Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


try:
    cookie_password = st.secrets["COOKIE_PASSWORD"]
except Exception:
    st.error(
        "COOKIE_PASSWORD is missing from Streamlit secrets. "
        "Add it before using persistent sign-in."
    )
    st.stop()


auth_cookies = EncryptedCookieManager(
    prefix="folsom-trade-assistant/",
    password=cookie_password,
)

if not auth_cookies.ready():
    st.stop()


st.title("📈 Folsom Trade Assistant")

st.caption(
    "A beginner-friendly dashboard for finding and explaining higher-quality "
    "trade candidates with minimal input. Active Trades is an optional organizer "
    "for positions you actually enter."
)

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "analysis_entry_anchors" not in st.session_state:
    st.session_state.analysis_entry_anchors = {}


def save_auth_tokens(access_token, refresh_token):
    """Save Supabase tokens in Session State and encrypted browser cookies."""
    st.session_state.supabase_access_token = access_token
    st.session_state.supabase_refresh_token = refresh_token

    auth_cookies["supabase_access_token"] = access_token
    auth_cookies["supabase_refresh_token"] = refresh_token
    auth_cookies.save()


def remember_auth_response(response):
    """Remember an authenticated Supabase response."""
    if response.user:
        st.session_state.supabase_user_id = str(response.user.id)
        st.session_state.supabase_user_email = (
            response.user.email or "Signed-in user"
        )

    if response.session:
        save_auth_tokens(
            response.session.access_token,
            response.session.refresh_token,
        )


def clear_auth_state():
    """Remove authentication from Session State and browser cookies."""
    for key in [
        "supabase_client",
        "supabase_user_id",
        "supabase_user_email",
        "supabase_access_token",
        "supabase_refresh_token",
    ]:
        st.session_state.pop(key, None)

    for cookie_name in [
        "supabase_access_token",
        "supabase_refresh_token",
    ]:
        if cookie_name in auth_cookies:
            del auth_cookies[cookie_name]

    auth_cookies.save()


def get_supabase_client():
    """Create a client and restore a browser login when available."""
    if "supabase_client" in st.session_state:
        return st.session_state.supabase_client, None

    try:
        project_url = st.secrets["SUPABASE_URL"]
        publishable_key = st.secrets["SUPABASE_KEY"]
    except Exception:
        return None, (
            "Supabase credentials are missing. Add SUPABASE_URL and "
            "SUPABASE_KEY to Streamlit secrets."
        )

    try:
        client = create_client(project_url, publishable_key)

        access_token = (
            st.session_state.get("supabase_access_token")
            or auth_cookies.get("supabase_access_token")
        )
        refresh_token = (
            st.session_state.get("supabase_refresh_token")
            or auth_cookies.get("supabase_refresh_token")
        )

        if access_token and refresh_token:
            try:
                restored = client.auth.set_session(
                    access_token,
                    refresh_token,
                )
                remember_auth_response(restored)
            except Exception:
                clear_auth_state()

        st.session_state.supabase_client = client
        return client, None

    except Exception as error:
        return None, f"Supabase connection failed: {error}"


def load_cloud_trades(client, user_id):
    """Load this user's active trades from Supabase."""
    response = (
        client.table("trades")
        .select(
            "id,ticker,direction,entry_price,stop_price,"
            "target_price,quantity,status,notes,created_at"
        )
        .eq("user_id", user_id)
        .eq("status", "ACTIVE")
        .order("created_at", desc=False)
        .execute()
    )

    return response.data or []


def load_pending_orders(client, user_id):
    """Load paper limit orders that have not filled yet."""
    response = (
        client.table("trades")
        .select(
            "id,ticker,direction,entry_price,stop_price,"
            "target_price,quantity,status,notes,created_at"
        )
        .eq("user_id", user_id)
        .eq("status", "PENDING")
        .order("created_at", desc=False)
        .execute()
    )

    return response.data or []


def load_closed_trades(client, user_id, limit=20):
    """Load recent completed trades for the authenticated user."""
    response = (
        client.table("trades")
        .select(
            "id,ticker,direction,entry_price,stop_price,"
            "target_price,quantity,exit_price,status,notes,"
            "created_at,closed_at"
        )
        .eq("user_id", user_id)
        .eq("status", "CLOSED")
        .order("closed_at", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data or []


def add_cloud_trade(client, user_id, trade):
    """Insert one active trade for the authenticated user."""
    payload = {
        "user_id": user_id,
        "ticker": trade["ticker"],
        "direction": trade["direction"],
        "entry_price": trade["entry"],
        "stop_price": trade["stop"],
        "target_price": trade["target"],
        "quantity": trade["quantity"],
        "status": trade.get("status", "ACTIVE"),
    }

    if trade.get("notes"):
        payload["notes"] = trade["notes"]

    response = client.table("trades").insert(payload).execute()
    return response.data


def add_pending_paper_order(client, user_id, order):
    """Save a recommended entry as a pending paper limit order."""
    return add_cloud_trade(
        client,
        user_id,
        {
            **order,
            "status": "PENDING",
            "notes": "PAPER LIMIT ORDER | Waiting for recommended entry",
        },
    )


def entry_is_waiting(direction, entry_price, current_price):
    """Return True when the proposed limit entry has not traded yet."""
    if current_price is None:
        return False

    entry_price = float(entry_price)
    current_price = float(current_price)
    tolerance = max(0.01, entry_price * 0.001)

    if str(direction).upper() == "LONG":
        return current_price > entry_price + tolerance
    return current_price < entry_price - tolerance


def activate_pending_order(client, order_id):
    """Convert a filled pending paper order into an active paper trade."""
    detected_at = datetime.now(timezone.utc).isoformat()
    response = (
        client.table("trades")
        .update(
            {
                "status": "ACTIVE",
                "notes": (
                    "PAPER TRADE | Pending limit entry was touched; "
                    f"fill detected {detected_at}"
                ),
            }
        )
        .eq("id", order_id)
        .execute()
    )
    return response.data


def cancel_pending_order(client, order_id):
    """Cancel a pending paper order without creating an active position."""
    response = (
        client.table("trades")
        .update(
            {
                "status": "CANCELLED",
                "notes": "PAPER LIMIT ORDER | Cancelled before fill",
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", order_id)
        .execute()
    )
    return response.data


def close_cloud_trade(client, trade_id, exit_price, notes=""):
    """Close an active trade and preserve it in trade history."""
    response = (
        client.table("trades")
        .update(
            {
                "status": "CLOSED",
                "exit_price": exit_price,
                "notes": notes or None,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", trade_id)
        .execute()
    )

    return response.data


@st.cache_data(ttl=30, show_spinner=False)
def get_latest_quote(ticker):
    """Return the latest Yahoo quote, prior close, and quote timestamp."""
    symbol = ticker.strip().upper()
    stock = yf.Ticker(symbol)

    latest_price = None
    previous_close = None
    quote_timestamp = None

    try:
        fast_info = stock.fast_info
        fast_latest = fast_info.get("last_price")
        fast_previous = fast_info.get("previous_close")

        if fast_latest is not None and float(fast_latest) > 0:
            latest_price = float(fast_latest)

        if fast_previous is not None and float(fast_previous) > 0:
            previous_close = float(fast_previous)
    except Exception:
        pass

    intraday = pd.DataFrame()

    for period, interval in [("1d", "1m"), ("5d", "5m")]:
        try:
            intraday = stock.history(
                period=period,
                interval=interval,
                prepost=True,
                auto_adjust=False,
            )
            intraday = intraday.dropna(subset=["Close"])
        except Exception:
            intraday = pd.DataFrame()

        if not intraday.empty:
            break

    if not intraday.empty:
        latest_price = float(intraday["Close"].iloc[-1])
        quote_timestamp = pd.Timestamp(intraday.index[-1])

    if latest_price is None or previous_close is None:
        try:
            daily = stock.history(
                period="5d",
                interval="1d",
                auto_adjust=False,
            ).dropna(subset=["Close"])

            if latest_price is None and not daily.empty:
                latest_price = float(daily["Close"].iloc[-1])

            if previous_close is None:
                if len(daily) >= 2:
                    previous_close = float(daily["Close"].iloc[-2])
                elif not daily.empty:
                    previous_close = float(daily["Close"].iloc[-1])
        except Exception:
            pass

    if latest_price is None:
        return {
            "price": None,
            "previous_close": previous_close,
            "change": None,
            "change_percent": None,
            "updated_at": None,
        }

    if quote_timestamp is not None:
        try:
            if quote_timestamp.tzinfo is None:
                quote_timestamp = quote_timestamp.tz_localize(
                    "America/New_York"
                )
            else:
                quote_timestamp = quote_timestamp.tz_convert(
                    "America/New_York"
                )

            updated_at = quote_timestamp.strftime(
                "%b %d, %Y %I:%M:%S %p ET"
            )
        except Exception:
            updated_at = None
    else:
        updated_at = datetime.now(
            ZoneInfo("America/New_York")
        ).strftime("%b %d, %Y %I:%M:%S %p ET")

    if previous_close and previous_close > 0:
        change = latest_price - previous_close
        change_percent = (change / previous_close) * 100
    else:
        change = None
        change_percent = None

    return {
        "price": float(latest_price),
        "previous_close": previous_close,
        "change": change,
        "change_percent": change_percent,
        "updated_at": updated_at,
    }


@st.cache_data(ttl=30, show_spinner=False)
def get_latest_trade_price(ticker):
    """Return the latest available price used by Active Trades."""
    return get_latest_quote(ticker).get("price")


@st.cache_data(ttl=30, show_spinner=False)
def check_pending_limit_fill(ticker, direction, entry_price, created_at):
    """Check whether a pending limit entry traded since the order was saved."""
    symbol = ticker.strip().upper()
    entry_price = float(entry_price)
    direction = direction.upper()
    created = pd.to_datetime(created_at, utc=True, errors="coerce")
    current_price = None

    try:
        current_price = get_latest_trade_price(symbol)
    except Exception:
        current_price = None

    touched = False
    observed_low = None
    observed_high = None

    for period, interval in [("7d", "1m"), ("60d", "5m")]:
        try:
            bars = yf.Ticker(symbol).history(
                period=period,
                interval=interval,
                prepost=True,
                auto_adjust=False,
            )
            bars = bars.dropna(subset=["High", "Low"])
        except Exception:
            bars = pd.DataFrame()

        if bars.empty:
            continue

        index = pd.DatetimeIndex(bars.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        bars = bars.copy()
        bars.index = index

        if not pd.isna(created):
            bars = bars.loc[bars.index >= created]
        if bars.empty:
            continue

        observed_low = float(bars["Low"].min())
        observed_high = float(bars["High"].max())
        if direction == "LONG":
            touched = observed_low <= entry_price
        else:
            touched = observed_high >= entry_price
        break

    if not touched and current_price is not None:
        if direction == "LONG":
            touched = float(current_price) <= entry_price
        else:
            touched = float(current_price) >= entry_price

    return {
        "filled": bool(touched),
        "current_price": float(current_price) if current_price is not None else None,
        "observed_low": observed_low,
        "observed_high": observed_high,
    }


def sync_pending_orders(client, pending_orders):
    """Activate pending paper orders whose entry price has been touched."""
    filled = []
    checks = {}

    for order in pending_orders:
        try:
            check = check_pending_limit_fill(
                order["ticker"],
                order["direction"],
                order["entry_price"],
                order.get("created_at"),
            )
            checks[order["id"]] = check
            if check["filled"]:
                activate_pending_order(client, order["id"])
                filled.append(order["ticker"])
        except Exception as error:
            checks[order["id"]] = {
                "filled": False,
                "current_price": None,
                "error": str(error),
            }

    return filled, checks


def calculate_live_trade_metrics(trade, current_price):
    """Calculate P/L, level distances, and an easy-to-read status."""
    entry_price = float(trade["entry_price"])
    stop_price = float(trade["stop_price"])
    target_price = float(trade["target_price"])
    quantity = int(trade.get("quantity") or 1)
    direction = trade["direction"]

    if direction == "LONG":
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
        pnl_per_share = current_price - entry_price
        stop_distance = current_price - stop_price
        target_distance = target_price - current_price
        stop_hit = current_price <= stop_price
        target_hit = current_price >= target_price
    else:
        risk_per_share = stop_price - entry_price
        reward_per_share = entry_price - target_price
        pnl_per_share = entry_price - current_price
        stop_distance = stop_price - current_price
        target_distance = current_price - target_price
        stop_hit = current_price >= stop_price
        target_hit = current_price <= target_price

    total_pnl = pnl_per_share * quantity
    pnl_percent = (
        pnl_per_share / entry_price
        if entry_price > 0
        else 0.0
    )

    if target_hit:
        status = "TARGET HIT"
        status_kind = "success"
    elif stop_hit:
        status = "STOP LEVEL HIT"
        status_kind = "error"
    elif (
        reward_per_share > 0
        and target_distance <= reward_per_share * 0.25
    ):
        status = "NEAR TARGET"
        status_kind = "success"
    elif (
        risk_per_share > 0
        and stop_distance <= risk_per_share * 0.25
    ):
        status = "STOP AT RISK"
        status_kind = "warning"
    else:
        status = "HOLD / MONITOR"
        status_kind = "info"

    return {
        "quantity": quantity,
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "pnl_per_share": pnl_per_share,
        "total_pnl": total_pnl,
        "pnl_percent": pnl_percent,
        "stop_distance": max(0.0, stop_distance),
        "target_distance": max(0.0, target_distance),
        "status": status,
        "status_kind": status_kind,
    }


def build_suggested_trade_plan(
    ticker,
    direction,
    fallback_price,
    atr_14,
    reward_to_risk=2.0,
    entry_price_override=None,
):
    """Build editable ATR-based entry, stop, and target placeholders."""
    latest_price = (
        entry_price_override
        if entry_price_override is not None
        else get_latest_trade_price(ticker)
    )
    entry_price = float(latest_price or fallback_price)

    atr_value = float(atr_14) if atr_14 and atr_14 > 0 else 0.0
    minimum_risk = max(entry_price * 0.005, 0.01)
    risk_per_share = max(atr_value * 1.25, minimum_risk)

    if direction == "LONG":
        stop_price = max(0.01, entry_price - risk_per_share)
        actual_risk = entry_price - stop_price
        target_price = entry_price + (actual_risk * reward_to_risk)
    else:
        stop_price = entry_price + risk_per_share
        actual_risk = stop_price - entry_price
        target_price = max(
            0.01,
            entry_price - (actual_risk * reward_to_risk),
        )

    return {
        "entry": round(entry_price, 2),
        "stop": round(stop_price, 2),
        "target": round(target_price, 2),
        "risk_per_share": round(actual_risk, 2),
        "reward_per_share": round(actual_risk * reward_to_risk, 2),
        "reward_to_risk": reward_to_risk,
    }


def calculate_closed_trade_result(trade):
    """Return realized P/L for one completed trade."""
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    quantity = int(trade.get("quantity") or 1)

    if trade["direction"] == "LONG":
        pnl_per_share = exit_price - entry_price
    else:
        pnl_per_share = entry_price - exit_price

    return pnl_per_share * quantity



def extract_number(value):
    """Return a float from either a raw number or Yahoo's raw/fmt dictionary."""
    if isinstance(value, dict):
        value = value.get("raw", value.get("fmt"))

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_url(value):
    """Return a URL from either a string or a Yahoo URL dictionary."""
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return value.get("url")

    return None


def format_news_date(value):
    """Convert Yahoo timestamps or ISO date strings into readable text."""
    if value is None:
        return ""

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(
                value,
                tz=timezone.utc,
            ).strftime("%b %d, %Y %I:%M %p UTC")
        except (ValueError, OSError):
            return ""

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
            return parsed.strftime("%b %d, %Y %I:%M %p UTC")
        except ValueError:
            return value

    return ""


def parse_news_item(item):
    """Handle both older and newer yfinance news response formats."""
    content = item.get("content", item)

    title = (
        content.get("title")
        or item.get("title")
        or "Untitled article"
    )

    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "Unknown source")
    else:
        publisher = (
            item.get("publisher")
            or provider
            or "Unknown source"
        )

    url = (
        extract_url(content.get("clickThroughUrl"))
        or extract_url(content.get("canonicalUrl"))
        or extract_url(content.get("previewUrl"))
        or extract_url(item.get("link"))
    )

    published_value = (
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
    )

    summary = (
        content.get("summary")
        or content.get("description")
        or ""
    )

    return {
        "title": title,
        "publisher": publisher,
        "url": url,
        "published": format_news_date(published_value),
        "summary": summary,
    }


def build_screener_table(screen_name, count=10):
    """Load and clean one predefined yfinance screener."""
    response = yf.screen(screen_name, count=count)
    quotes = response.get("quotes", [])

    rows = []

    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol:
            continue

        price = extract_number(
            quote.get("regularMarketPrice")
            or quote.get("intradayprice")
        )

        change_percent = extract_number(
            quote.get("regularMarketChangePercent")
            or quote.get("percentchange")
        )

        volume = extract_number(
            quote.get("regularMarketVolume")
            or quote.get("dayvolume")
            or quote.get("eodvolume")
        )

        rows.append(
            {
                "Ticker": symbol,
                "Company": (
                    quote.get("shortName")
                    or quote.get("longName")
                    or ""
                ),
                "Price": price,
                "Change %": change_percent,
                "Volume": volume,
            }
        )

    table = pd.DataFrame(rows)

    if not table.empty:
        table["Price"] = table["Price"].map(
            lambda value: (
                f"${value:,.2f}"
                if pd.notna(value)
                else "—"
            )
        )

        table["Change %"] = table["Change %"].map(
            lambda value: (
                f"{value:+.2f}%"
                if pd.notna(value)
                else "—"
            )
        )

        table["Volume"] = table["Volume"].map(
            lambda value: (
                f"{int(value):,}"
                if pd.notna(value)
                else "—"
            )
        )

    return table


def parse_watchlist(text, primary_ticker):
    """Clean a comma-separated watchlist and keep no more than eight symbols."""
    symbols = []

    for item in text.split(","):
        symbol = item.strip().upper()

        if symbol and symbol not in symbols:
            symbols.append(symbol)

    if primary_ticker and primary_ticker not in symbols:
        symbols.insert(0, primary_ticker)

    return symbols[:8]


def build_watchlist_data(symbols):
    """Create a watchlist table and normalized comparison data."""
    rows = []
    normalized_prices = pd.DataFrame()
    failed_symbols = []

    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(
                period="3mo",
                auto_adjust=False,
            )

            data = data.dropna(subset=["Close"])

            if len(data) < 2:
                failed_symbols.append(symbol)
                continue

            latest_price = float(data["Close"].iloc[-1])
            previous_price = float(data["Close"].iloc[-2])
            starting_price = float(data["Close"].iloc[0])

            daily_change_percent = (
                (latest_price - previous_price)
                / previous_price
            ) * 100

            three_month_return = (
                (latest_price - starting_price)
                / starting_price
            ) * 100

            average_20 = float(
                data["Close"].rolling(20).mean().iloc[-1]
            )

            if math.isnan(average_20):
                trend = "Not enough data"
            elif latest_price > average_20:
                trend = "Above 20-day average"
            else:
                trend = "Below 20-day average"

            rows.append(
                {
                    "Ticker": symbol,
                    "Price": latest_price,
                    "Daily change": daily_change_percent,
                    "3-month return": three_month_return,
                    "Volume": int(data["Volume"].iloc[-1]),
                    "Short-term trend": trend,
                }
            )

            normalized_prices[symbol] = (
                data["Close"] / starting_price
            ) * 100

        except Exception:
            failed_symbols.append(symbol)

    table = pd.DataFrame(rows)

    if not table.empty:
        display_table = table.copy()

        display_table["Price"] = display_table["Price"].map(
            lambda value: f"${value:,.2f}"
        )

        display_table["Daily change"] = (
            display_table["Daily change"].map(
                lambda value: f"{value:+.2f}%"
            )
        )

        display_table["3-month return"] = (
            display_table["3-month return"].map(
                lambda value: f"{value:+.2f}%"
            )
        )

        display_table["Volume"] = display_table["Volume"].map(
            lambda value: f"{value:,}"
        )

    else:
        display_table = table

    return display_table, normalized_prices, failed_symbols


def calculate_trade_setup(
    price,
    previous_price,
    ma20,
    ma50,
    rsi,
    macd,
    signal,
    histogram,
    previous_histogram,
    upper_band,
    lower_band,
    latest_volume,
    average_volume_20,
):
    """
    Return a simple technical direction and setup-quality score.

    The direction score ranges from -7 to +7:
    negative values lean short, positive values lean long.

    Setup quality ranges from 0 to 100 and measures how strongly
    the indicators agree. It is not a probability of profit.
    """
    direction_score = 0
    evidence = []
    risk_flags = []

    # Price versus moving averages
    if price > ma20 and price > ma50:
        direction_score += 2
        evidence.append(
            "Price is above both the 20-day and 50-day averages."
        )

    elif price < ma20 and price < ma50:
        direction_score -= 2
        evidence.append(
            "Price is below both the 20-day and 50-day averages."
        )

    else:
        risk_flags.append(
            "Price is between the moving averages, so trend direction is mixed."
        )

    # Moving-average relationship
    if ma20 > ma50:
        direction_score += 1
        evidence.append(
            "The 20-day average is above the 50-day average."
        )
    else:
        direction_score -= 1
        evidence.append(
            "The 20-day average is below the 50-day average."
        )

    # MACD direction
    if macd > signal:
        direction_score += 2
        evidence.append(
            "MACD is above its signal line."
        )
    else:
        direction_score -= 2
        evidence.append(
            "MACD is below its signal line."
        )

    # MACD histogram acceleration
    if histogram > 0 and histogram >= previous_histogram:
        direction_score += 1
        evidence.append(
            "The positive MACD histogram is holding or strengthening."
        )

    elif histogram < 0 and histogram <= previous_histogram:
        direction_score -= 1
        evidence.append(
            "The negative MACD histogram is holding or weakening further."
        )

    else:
        risk_flags.append(
            "MACD histogram momentum is not clearly strengthening."
        )

    # RSI directional zone
    if 55 <= rsi < 70:
        direction_score += 1
        evidence.append(
            "RSI is in a positive momentum zone without being above 70."
        )

    elif 30 < rsi <= 45:
        direction_score -= 1
        evidence.append(
            "RSI is in a weak momentum zone."
        )

    elif rsi >= 70:
        risk_flags.append(
            "RSI is above 70, so the move may be stretched upward."
        )

    elif rsi <= 30:
        risk_flags.append(
            "RSI is below 30, so the move may be stretched downward."
        )

    # Bollinger stretch warnings
    if price > upper_band:
        risk_flags.append(
            "Price is above the upper Bollinger Band."
        )

    elif price < lower_band:
        risk_flags.append(
            "Price is below the lower Bollinger Band."
        )

    # Convert direction score into a plain-English bias
    if direction_score >= 4:
        bias = "LONG BIAS"
        bias_sign = 1

    elif direction_score <= -4:
        bias = "SHORT BIAS"
        bias_sign = -1

    else:
        bias = "WAIT"
        bias_sign = 0

    # Score how strongly the signals agree with the detected bias
    setup_quality = 0

    if bias_sign == 1:
        if price > ma20 and price > ma50:
            setup_quality += 25

        if ma20 > ma50:
            setup_quality += 15

        if macd > signal:
            setup_quality += 25

        if histogram > 0 and histogram >= previous_histogram:
            setup_quality += 10

        if 50 < rsi < 70:
            setup_quality += 15
        elif 45 <= rsi <= 75:
            setup_quality += 8

        if latest_volume > average_volume_20:
            if price > previous_price:
                setup_quality += 10
            else:
                setup_quality += 4

    elif bias_sign == -1:
        if price < ma20 and price < ma50:
            setup_quality += 25

        if ma20 < ma50:
            setup_quality += 15

        if macd < signal:
            setup_quality += 25

        if histogram < 0 and histogram <= previous_histogram:
            setup_quality += 10

        if 30 < rsi < 50:
            setup_quality += 15
        elif 25 <= rsi <= 55:
            setup_quality += 8

        if latest_volume > average_volume_20:
            if price < previous_price:
                setup_quality += 10
            else:
                setup_quality += 4

    else:
        setup_quality = round(
            abs(direction_score) / 7 * 45
        )

    # Penalize stretched or conflicting setups
    if rsi >= 75 or rsi <= 25:
        setup_quality -= 10

    if price > upper_band or price < lower_band:
        setup_quality -= 10

    if bias_sign == 1 and price < previous_price:
        setup_quality -= 5

    if bias_sign == -1 and price > previous_price:
        setup_quality -= 5

    setup_quality = max(
        0,
        min(100, int(round(setup_quality))),
    )

    # Decide whether the setup is actionable enough to investigate
    if bias == "WAIT":
        trade_status = "NO CLEAR TRADE"
        status_text = (
            "The indicators do not agree strongly enough on direction."
        )

    elif setup_quality >= 75:
        trade_status = "POTENTIAL TRADE SETUP"
        status_text = (
            "Several indicators agree. Check news, entry price, "
            "risk, and market conditions before acting."
        )

    elif setup_quality >= 60:
        trade_status = "WATCH FOR CONFIRMATION"
        status_text = (
            "There is a directional lean, but the setup needs "
            "more confirmation."
        )

    else:
        trade_status = "NO CLEAR TRADE"
        status_text = (
            "The directional lean is not supported strongly enough "
            "for a clean setup."
        )

    return {
        "bias": bias,
        "direction_score": direction_score,
        "setup_quality": setup_quality,
        "trade_status": trade_status,
        "status_text": status_text,
        "evidence": evidence,
        "risk_flags": risk_flags,
    }



def add_backtest_indicators(data):
    """Calculate technical indicators using only information available at each close."""
    result = data.copy()

    result["MA20"] = result["Close"].rolling(20).mean()
    result["MA50"] = result["Close"].rolling(50).mean()

    standard_deviation = result["Close"].rolling(20).std()
    result["Upper Band"] = result["MA20"] + (2 * standard_deviation)
    result["Lower Band"] = result["MA20"] - (2 * standard_deviation)

    movement = result["Close"].diff()
    average_gain = movement.clip(lower=0).rolling(14).mean()
    average_loss = (-movement.clip(upper=0)).rolling(14).mean()
    relative_strength = average_gain / average_loss
    result["RSI"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_gain == 0) & (average_loss == 0), "RSI"] = 50.0
    result.loc[(average_gain > 0) & (average_loss == 0), "RSI"] = 100.0
    result.loc[(average_gain == 0) & (average_loss > 0), "RSI"] = 0.0

    ema_12 = result["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = result["Close"].ewm(span=26, adjust=False).mean()
    result["MACD"] = ema_12 - ema_26
    result["Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["Histogram"] = result["MACD"] - result["Signal"]
    result["Average Volume 20"] = result["Volume"].rolling(20).mean()

    previous_close = result["Close"].shift(1)
    true_range = pd.concat(
        [
            result["High"] - result["Low"],
            (result["High"] - previous_close).abs(),
            (result["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR14"] = true_range.rolling(14).mean()

    return result

def remove_unfinished_daily_bar(data):
    """
    Exclude today's daily candle while the regular U.S. session is open.

    This prevents a partly formed daily candle from being treated as a
    completed historical signal.
    """
    if data.empty:
        return data

    try:
        now_eastern = datetime.now(
            ZoneInfo("America/New_York")
        )

        last_timestamp = pd.Timestamp(data.index[-1])

        if last_timestamp.tzinfo is not None:
            last_date = (
                last_timestamp
                .tz_convert("America/New_York")
                .date()
            )
        else:
            last_date = last_timestamp.date()

        market_is_not_finished = (
            now_eastern.weekday() < 5
            and now_eastern.time() < clock_time(16, 15)
        )

        if (
            last_date == now_eastern.date()
            and market_is_not_finished
        ):
            return data.iloc[:-1].copy()

    except Exception:
        return data

    return data



AIRLINE_TICKERS = {
    "AAL", "ALK", "DAL", "JBLU", "LUV", "UAL", "CPA",
}

ENERGY_TICKERS = {
    "APA", "COP", "CVX", "DVN", "EOG", "FANG", "HES", "MPC",
    "OXY", "PSX", "XOM", "SLB", "HAL", "BKR",
}

SEMICONDUCTOR_TICKERS = {
    "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "AMAT",
    "LRCX", "KLAC", "ASML", "TSM", "ARM", "MRVL", "ON", "MCHP",
    "ADI", "NXPI", "MPWR", "SMCI",
}

BANK_TICKERS = {
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC",
    "SCHW", "BK", "STT", "COF",
}

GOLD_MINER_TICKERS = {
    "NEM", "GOLD", "AEM", "KGC", "AU", "GFI", "WPM", "FNV",
}

GROWTH_TECH_TICKERS = {
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "NFLX",
    "CRM", "ORCL", "ADBE", "NOW", "PLTR", "SNOW", "SHOP",
}

INDUSTRIAL_TICKERS = {
    "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "NOC", "UPS",
    "FDX", "UNP", "CSX",
}

SECTOR_ETF_BY_SECTOR = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

MACRO_SYMBOLS = {
    "Oil": "CL=F",
    "Gold": "GC=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "Dollar": "DX-Y.NYB",
    "US10Y": "^TNX",
    "JETS": "JETS",
    "SMH": "SMH",
    "XLF": "XLF",
    "KRE": "KRE",
    "XLE": "XLE",
    "GDX": "GDX",
    "XLK": "XLK",
    "XLI": "XLI",
    "XLV": "XLV",
    "XLY": "XLY",
    "XLP": "XLP",
    "XLC": "XLC",
    "XLB": "XLB",
    "XLRE": "XLRE",
    "XLU": "XLU",
}


def normalize_market_dates(index):
    """Convert market timestamps into timezone-naive Eastern calendar dates."""
    normalized = pd.DatetimeIndex(pd.to_datetime(index))

    if normalized.tz is not None:
        normalized = (
            normalized
            .tz_convert("America/New_York")
            .tz_localize(None)
        )

    return normalized.normalize()


@st.cache_data(ttl=86400, show_spinner=False)
def get_ticker_company_profile(ticker):
    """Detect sector, industry, and a practical macro relationship profile."""
    symbol = ticker.strip().upper()
    sector = "Unknown"
    industry = "Unknown"

    try:
        info = yf.Ticker(symbol).get_info()
        sector = str(info.get("sector") or "Unknown")
        industry = str(info.get("industry") or "Unknown")
    except Exception:
        pass

    combined = f"{sector} {industry}".lower()

    if symbol in AIRLINE_TICKERS or "airline" in combined:
        profile = "Airline"
    elif symbol in SEMICONDUCTOR_TICKERS or "semiconductor" in combined:
        profile = "Semiconductor"
    elif symbol in GOLD_MINER_TICKERS or (
        "gold" in combined and ("mining" in combined or "miner" in combined)
    ):
        profile = "Gold miner"
    elif symbol in ENERGY_TICKERS or sector == "Energy":
        profile = "Energy"
    elif symbol in BANK_TICKERS or "bank" in combined:
        profile = "Bank"
    elif symbol in INDUSTRIAL_TICKERS or sector == "Industrials":
        profile = "Industrial"
    elif symbol in GROWTH_TECH_TICKERS or sector == "Technology":
        profile = "Growth / technology"
    else:
        profile = "General market"

    sector_etf = SECTOR_ETF_BY_SECTOR.get(sector)

    return {
        "ticker": symbol,
        "sector": sector,
        "industry": industry,
        "profile": profile,
        "sector_etf": sector_etf,
    }


def resolve_ticker_macro_profile(ticker, selected_profile):
    """Return detected company context, allowing a manual profile override."""
    classification = get_ticker_company_profile(ticker).copy()
    if selected_profile != "Auto by ticker":
        classification["profile"] = selected_profile
    return classification


def _factor(
    key,
    label,
    mode,
    bullish_when,
    weight,
    relevance,
    threshold,
    reason,
):
    return {
        "key": key,
        "label": label,
        "mode": mode,
        "bullish_when": bullish_when,
        "weight": int(weight),
        "relevance": relevance,
        "threshold": float(threshold),
        "reason": reason,
    }


def build_ticker_factor_plan(classification, oil_threshold=0.02):
    """Build a ticker-aware factor list while still observing broad context."""
    profile = classification["profile"]
    sector_etf = classification.get("sector_etf")

    factors = [
        _factor(
            "SPY", "Broad market trend", "trend", "up", 2, "High", 0,
            "Most stocks trade better when the broad market trend agrees.",
        ),
        _factor(
            "VIX", "Market volatility", "return", "down", 1, "Moderate", 0.10,
            "Falling volatility generally supports risk-taking; sharp rises add stress.",
        ),
    ]

    if profile == "Airline":
        factors += [
            _factor("JETS", "Airline sector", "trend", "up", 3, "Very high", 0,
                    "Airline-sector strength is a direct peer confirmation."),
            _factor("Oil", "Oil / fuel cost", "return", "down", 3, "Very high", oil_threshold,
                    "Lower fuel costs can support airline margins; sharp oil rises can pressure them."),
            _factor("Dollar", "U.S. dollar", "return", "down", 1, "Moderate", 0.015,
                    "Large dollar moves can affect international demand and costs."),
        ]
    elif profile == "Semiconductor":
        factors += [
            _factor("SMH", "Semiconductor sector", "trend", "up", 3, "Very high", 0,
                    "Chip-sector strength is more relevant than unrelated commodities."),
            _factor("QQQ", "Nasdaq growth trend", "trend", "up", 2, "High", 0,
                    "Semiconductors are strongly tied to growth-stock risk appetite."),
            _factor("US10Y", "10-year Treasury yield", "change", "down", 2, "High", 1.0,
                    "Rapidly rising yields can pressure high-duration growth valuations."),
            _factor("Dollar", "U.S. dollar", "return", "down", 1, "Moderate", 0.015,
                    "A stronger dollar can weigh on multinational revenue translation."),
        ]
    elif profile == "Bank":
        factors += [
            _factor("XLF", "Financial sector", "trend", "up", 3, "Very high", 0,
                    "Financial-sector confirmation is a direct peer signal."),
            _factor("KRE", "Regional bank trend", "trend", "up", 2, "High", 0,
                    "Regional-bank strength helps reveal banking-system risk appetite."),
            _factor("US10Y", "10-year Treasury yield", "change", "up", 2, "High", 1.0,
                    "Rate changes can affect lending margins, although the relationship is not always linear."),
        ]
    elif profile == "Energy":
        factors += [
            _factor("XLE", "Energy sector", "trend", "up", 3, "Very high", 0,
                    "Energy-sector strength is the closest peer confirmation."),
            _factor("Oil", "Oil price", "return", "up", 3, "Very high", oil_threshold,
                    "Higher oil generally supports producers; sharp declines can pressure them."),
            _factor("Natural Gas", "Natural gas", "return", "up", 1, "Moderate", 0.04,
                    "Natural-gas sensitivity matters for many diversified energy companies."),
            _factor("Dollar", "U.S. dollar", "return", "down", 1, "Moderate", 0.015,
                    "A stronger dollar can pressure dollar-priced commodities."),
        ]
    elif profile == "Gold miner":
        factors += [
            _factor("GDX", "Gold-miner sector", "trend", "up", 3, "Very high", 0,
                    "Gold-miner peer strength is a direct confirmation."),
            _factor("Gold", "Gold price", "return", "up", 3, "Very high", 0.02,
                    "Gold prices are a primary revenue driver for miners."),
            _factor("Dollar", "U.S. dollar", "return", "down", 2, "High", 0.015,
                    "Gold often faces pressure from a sharply stronger dollar."),
            _factor("US10Y", "10-year Treasury yield", "change", "down", 1, "Moderate", 1.0,
                    "Rising yields can compete with non-yielding gold."),
        ]
    elif profile == "Industrial":
        factors += [
            _factor("XLI", "Industrial sector", "trend", "up", 3, "Very high", 0,
                    "Industrial-sector strength is a direct peer confirmation."),
            _factor("Copper", "Copper / growth demand", "return", "up", 1, "Moderate", 0.02,
                    "Copper can reflect global industrial demand and economic expectations."),
            _factor("Dollar", "U.S. dollar", "return", "down", 1, "Moderate", 0.015,
                    "A strong dollar can pressure multinational industrial revenue."),
        ]
    elif profile == "Growth / technology":
        factors += [
            _factor("QQQ", "Nasdaq growth trend", "trend", "up", 3, "Very high", 0,
                    "Growth-stock risk appetite is a major technology driver."),
            _factor("XLK", "Technology sector", "trend", "up", 2, "High", 0,
                    "Technology-sector strength provides direct peer confirmation."),
            _factor("US10Y", "10-year Treasury yield", "change", "down", 2, "High", 1.0,
                    "Rapidly rising yields can pressure growth valuations."),
            _factor("Dollar", "U.S. dollar", "return", "down", 1, "Moderate", 0.015,
                    "A stronger dollar can weigh on multinational earnings translation."),
        ]
    elif sector_etf and sector_etf != "SPY":
        factors.append(
            _factor(
                sector_etf,
                f"{classification.get('sector', 'Sector')} trend",
                "trend", "up", 3, "Very high", 0,
                "The company's sector trend is its most relevant peer comparison.",
            )
        )

    # Keep the broad picture visible without forcing unrelated factors into the score.
    observed_only = [
        _factor("QQQ", "Nasdaq trend", "trend", "up", 0, "Low", 0,
                "Observed for context but not used in this profile's score."),
        _factor("US10Y", "10-year Treasury yield", "change", "down", 0, "Low", 1.0,
                "Observed for context but not used in this profile's score."),
        _factor("Dollar", "U.S. dollar", "return", "down", 0, "Low", 0.015,
                "Observed for context but not used in this profile's score."),
        _factor("Oil", "Oil price", "return", "up", 0, "Low", oil_threshold,
                "Observed for context but not used in this profile's score."),
        _factor("Gold", "Gold price", "return", "up", 0, "Low", 0.02,
                "Observed for context but not used in this profile's score."),
    ]

    by_key = {factor["key"]: factor for factor in factors}
    for factor in observed_only:
        by_key.setdefault(factor["key"], factor)

    return list(by_key.values())


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_history(start_date, requested_labels=()):
    """Download only the macro/sector histories required by the factor plan."""
    labels = tuple(requested_labels) or tuple(MACRO_SYMBOLS)
    series = {}

    for label in labels:
        symbol = MACRO_SYMBOLS.get(label)
        if not symbol:
            continue
        try:
            history = yf.Ticker(symbol).history(
                start=start_date,
                auto_adjust=True,
            )
        except Exception:
            history = pd.DataFrame()

        if history.empty or "Close" not in history:
            continue

        close = history["Close"].dropna().astype(float)
        close.index = normalize_market_dates(close.index)
        close = close[~close.index.duplicated(keep="last")]
        series[label] = close

    if not series:
        return pd.DataFrame()

    return pd.concat(series, axis=1).sort_index().ffill()


def build_macro_context(macro_history, lookback_days=5):
    """Create comparable returns, trends, and yield-change features."""
    if macro_history.empty:
        return pd.DataFrame()

    context = macro_history.copy().sort_index()

    for label in macro_history.columns:
        if label == "US10Y":
            context["US10Y Change"] = context[label].diff(lookback_days)
        else:
            context[f"{label} Return"] = context[label].pct_change(lookback_days)

        context[f"{label} MA50"] = context[label].rolling(50).mean()
        context[f"{label} Above MA50"] = context[label] > context[f"{label} MA50"]

    return context


def factor_required_column(factor):
    if factor["mode"] == "trend":
        return f"{factor['key']} Above MA50"
    if factor["mode"] == "change":
        return f"{factor['key']} Change"
    return f"{factor['key']} Return"


def evaluate_macro_factor(row, factor):
    """Evaluate one factor as bullish, bearish, neutral, or observed-only."""
    column = factor_required_column(factor)
    value = row.get(column)
    weight = int(factor["weight"])

    if pd.isna(value):
        return 0, "Unavailable", "—", f"{factor['label']} was unavailable."

    if factor["mode"] == "trend":
        direction = "up" if bool(value) else "down"
        reading = "Above MA50" if bool(value) else "Below MA50"
        crossed = True
    else:
        numeric_value = float(value)
        reading = f"{numeric_value:+.1%}" if factor["mode"] == "return" else f"{numeric_value:+.2f}"
        if abs(numeric_value) < factor["threshold"]:
            direction = "neutral"
            crossed = False
        else:
            direction = "up" if numeric_value > 0 else "down"
            crossed = True

    if weight == 0:
        return 0, "Not used", reading, f"{factor['label']} is observed but excluded from this ticker's score."

    if not crossed or direction == "neutral":
        return 0, "Neutral", reading, f"{factor['label']} did not make a meaningful move."

    supportive = direction == factor["bullish_when"]
    score = weight if supportive else -weight
    effect = "Supportive" if supportive else "Negative"
    evidence = f"{factor['label']} was {effect.lower()} ({reading})."
    return score, effect, reading, evidence


def calculate_macro_context_score(row, factor_plan):
    """Return a bullish-to-bearish score using only relevant ticker factors."""
    total_score = 0
    evidence = []

    for factor in factor_plan:
        score, effect, reading, note = evaluate_macro_factor(row, factor)
        total_score += score
        if factor["weight"] > 0 and effect != "Neutral":
            evidence.append(note)

    if not evidence:
        evidence.append("No relevant macro factor produced a strong directional signal.")

    return total_score, evidence


def build_current_factor_table(macro_context, factor_plan):
    """Create a readable current factor table and current weighted score."""
    columns = ["Factor", "Relevance", "Current effect", "Reading", "Weight", "Why it matters"]
    if macro_context.empty:
        return pd.DataFrame(columns=columns), 0

    latest = macro_context.dropna(how="all").iloc[-1]
    rows = []
    total_score = 0

    for factor in factor_plan:
        score, effect, reading, _ = evaluate_macro_factor(latest, factor)
        total_score += score
        rows.append(
            {
                "Factor": factor["label"],
                "Relevance": factor["relevance"],
                "Current effect": effect,
                "Reading": reading,
                "Weight": factor["weight"] if factor["weight"] else "Ignored",
                "Why it matters": factor["reason"],
            }
        )

    relevance_order = {"Very high": 0, "High": 1, "Moderate": 2, "Low": 3}
    table = pd.DataFrame(rows)
    table["_order"] = table["Relevance"].map(relevance_order).fillna(9)
    table = table.sort_values(["_order", "Factor"]).drop(columns="_order")
    return table, total_score


def attach_macro_context(prepared, macro_context):
    """Join macro features to stock sessions using each session's date."""
    result = prepared.copy()
    result["_Macro Date"] = normalize_market_dates(result.index)

    context = macro_context.copy()
    context.index = normalize_market_dates(context.index)
    context = context[
        ~context.index.duplicated(keep="last")
    ]

    return result.join(
        context,
        on="_Macro Date",
        how="left",
    )



def run_macro_strategy_backtest(
    data,
    macro_context,
    test_start_date,
    holding_days,
    minimum_quality,
    cost_bps_per_side,
    factor_plan,
    minimum_macro_score,
    stop_atr_multiple=1.25,
    reward_to_risk=2.0,
):
    """Run technical signals only when ticker-relevant macro context agrees."""
    prepared = remove_unfinished_daily_bar(data)
    prepared = prepared.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    prepared = add_backtest_indicators(prepared)
    prepared = attach_macro_context(prepared, macro_context)

    required_columns = [
        "Open", "High", "Low", "Close", "MA20", "MA50", "RSI", "MACD",
        "Signal", "Histogram", "Upper Band", "Lower Band",
        "Average Volume 20", "ATR14",
    ]
    required_columns += [
        factor_required_column(factor)
        for factor in factor_plan
        if factor["weight"] > 0
    ]
    prepared = prepared.dropna(subset=list(dict.fromkeys(required_columns)))

    trades = []
    index_position = 1

    while index_position < len(prepared) - 1:
        signal_row = prepared.iloc[index_position]
        previous_row = prepared.iloc[index_position - 1]
        signal_date = pd.Timestamp(prepared.index[index_position])

        if signal_date.date() < test_start_date:
            index_position += 1
            continue

        setup = calculate_trade_setup(
            price=float(signal_row["Close"]),
            previous_price=float(previous_row["Close"]),
            ma20=float(signal_row["MA20"]),
            ma50=float(signal_row["MA50"]),
            rsi=float(signal_row["RSI"]),
            macd=float(signal_row["MACD"]),
            signal=float(signal_row["Signal"]),
            histogram=float(signal_row["Histogram"]),
            previous_histogram=float(previous_row["Histogram"]),
            upper_band=float(signal_row["Upper Band"]),
            lower_band=float(signal_row["Lower Band"]),
            latest_volume=int(signal_row["Volume"]),
            average_volume_20=float(signal_row["Average Volume 20"]),
        )

        if setup["bias"] not in ("LONG BIAS", "SHORT BIAS"):
            index_position += 1
            continue
        if setup["setup_quality"] < minimum_quality:
            index_position += 1
            continue

        direction = "LONG" if setup["bias"] == "LONG BIAS" else "SHORT"
        macro_score, macro_evidence = calculate_macro_context_score(signal_row, factor_plan)
        macro_confirms = (
            macro_score >= minimum_macro_score
            if direction == "LONG"
            else macro_score <= -minimum_macro_score
        )
        if not macro_confirms:
            index_position += 1
            continue

        entry_position = index_position + 1
        if entry_position >= len(prepared):
            break
        entry_price = float(prepared["Open"].iloc[entry_position])
        if entry_price <= 0:
            index_position += 1
            continue

        simulated = simulate_atr_trade_exit(
            prepared=prepared,
            entry_position=entry_position,
            maximum_holding_days=holding_days,
            direction=direction,
            entry_price=entry_price,
            atr_at_signal=float(signal_row["ATR14"]),
            stop_atr_multiple=stop_atr_multiple,
            reward_to_risk=reward_to_risk,
        )
        exit_position = simulated["exit_position"]
        exit_price = simulated["exit_price"]
        gross_return = (
            (exit_price / entry_price) - 1
            if direction == "LONG"
            else (entry_price / exit_price) - 1
        )
        net_return = gross_return - (2 * cost_bps_per_side / 10000)

        trades.append(
            {
                "Signal Date": signal_date,
                "Entry Date": pd.Timestamp(prepared.index[entry_position]),
                "Exit Date": pd.Timestamp(prepared.index[exit_position]),
                "Direction": direction,
                "Setup Quality": int(setup["setup_quality"]),
                "Direction Score": int(setup["direction_score"]),
                "Macro Score": int(macro_score),
                "Macro Evidence": " ".join(macro_evidence),
                "Entry Price": entry_price,
                "Stop Price": simulated["stop_price"],
                "Target Price": simulated["target_price"],
                "Exit Price": exit_price,
                "Exit Reason": simulated["exit_reason"],
                "Gross Return": gross_return,
                "Net Return": net_return,
                "Winner": net_return > 0,
                "Holding Sessions": int(exit_position - entry_position + 1),
            }
        )
        index_position = exit_position + 1

    return pd.DataFrame(trades), prepared

def run_oil_shock_study(
    data,
    macro_context,
    test_start_date,
    holding_days,
    cost_bps_per_side,
    oil_threshold,
    study_mode,
):
    """
    Test the direct hypothesis of buying after oil drops or shorting after spikes.
    """
    prepared = remove_unfinished_daily_bar(data)

    prepared = prepared.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    prepared = attach_macro_context(
        prepared,
        macro_context,
    ).dropna(subset=["Oil Return"])

    trades = []
    index_position = 0

    while index_position < len(prepared) - holding_days:
        signal_row = prepared.iloc[index_position]
        signal_date = pd.Timestamp(
            prepared.index[index_position]
        )

        if signal_date.date() < test_start_date:
            index_position += 1
            continue

        oil_move = float(signal_row["Oil Return"])

        if study_mode == "Long stock after oil drop":
            qualifies = oil_move <= -oil_threshold
            direction = "LONG"
        elif study_mode == "Long stock after oil spike":
            qualifies = oil_move >= oil_threshold
            direction = "LONG"
        elif study_mode == "Short stock after oil drop":
            qualifies = oil_move <= -oil_threshold
            direction = "SHORT"
        else:
            qualifies = oil_move >= oil_threshold
            direction = "SHORT"

        if not qualifies:
            index_position += 1
            continue

        entry_position = index_position + 1
        exit_position = entry_position + holding_days - 1

        if exit_position >= len(prepared):
            break

        entry_price = float(
            prepared["Open"].iloc[entry_position]
        )
        exit_price = float(
            prepared["Close"].iloc[exit_position]
        )

        if entry_price <= 0 or exit_price <= 0:
            index_position += 1
            continue

        if direction == "LONG":
            gross_return = (exit_price / entry_price) - 1
        else:
            gross_return = (entry_price / exit_price) - 1

        net_return = gross_return - (
            2 * cost_bps_per_side / 10000
        )

        trades.append(
            {
                "Signal Date": signal_date,
                "Entry Date": pd.Timestamp(
                    prepared.index[entry_position]
                ),
                "Exit Date": pd.Timestamp(
                    prepared.index[exit_position]
                ),
                "Direction": direction,
                "Oil Move": oil_move,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Gross Return": gross_return,
                "Net Return": net_return,
                "Winner": net_return > 0,
            }
        )

        index_position = exit_position + 1

    return pd.DataFrame(trades), prepared


def format_macro_trade_table(trades):
    """Format recent macro-confirmed trades for display."""
    if trades.empty:
        return trades

    display = trades.copy()

    for column in ["Signal Date", "Entry Date", "Exit Date"]:
        display[column] = pd.to_datetime(
            display[column]
        ).dt.strftime("%Y-%m-%d")

    for column in ["Oil Move", "Gold Move", "VIX Move"]:
        if column in display:
            display[column] = display[column].map(
                lambda value: (
                    f"{value:+.1%}"
                    if pd.notna(value)
                    else "—"
                )
            )

    display["Net Return"] = display["Net Return"].map(
        lambda value: f"{value:+.2%}"
    )

    return display[
        [
            "Signal Date",
            "Direction",
            "Setup Quality",
            "Macro Score",
            "Oil Move",
            "Gold Move",
            "SPY Regime",
            "VIX Move",
            "Net Return",
        ]
    ]


def format_oil_study_table(trades):
    """Format recent oil-shock study trades."""
    if trades.empty:
        return trades

    display = trades.copy()

    for column in ["Signal Date", "Entry Date", "Exit Date"]:
        display[column] = pd.to_datetime(
            display[column]
        ).dt.strftime("%Y-%m-%d")

    display["Oil Move"] = display["Oil Move"].map(
        lambda value: f"{value:+.1%}"
    )
    display["Entry Price"] = display["Entry Price"].map(
        lambda value: f"${value:,.2f}"
    )
    display["Exit Price"] = display["Exit Price"].map(
        lambda value: f"${value:,.2f}"
    )
    display["Net Return"] = display["Net Return"].map(
        lambda value: f"{value:+.2%}"
    )

    return display[
        [
            "Signal Date",
            "Entry Date",
            "Exit Date",
            "Direction",
            "Oil Move",
            "Entry Price",
            "Exit Price",
            "Net Return",
        ]
    ]



def build_strategy_comparison_row(label, statistics):
    """Return one decision-useful comparison row for a completed backtest."""
    if statistics is None:
        return {
            "Strategy": label,
            "Edge grade": "No data",
            "Trades": 0,
            "Win rate": "—",
            "Average trade": "—",
            "Profit factor": "—",
            "Out-of-sample": "—",
            "Stable periods": "—",
            "Compounded return": "—",
            "Buy & hold": "—",
            "Exposure": "—",
            "Max drawdown": "—",
        }

    oos = statistics.get("out_of_sample") or {}
    pf = statistics["profit_factor"]
    pf_text = "∞" if math.isinf(pf) else f"{pf:.2f}"
    oos_text = (
        f"{int(oos.get('trades', 0))} trades • {float(oos.get('average_return', 0)):+.2%} avg"
        if oos
        else "—"
    )
    stable_count = int(statistics.get("positive_stability_periods") or 0)
    period_count = int(statistics.get("stability_period_count") or 0)

    return {
        "Strategy": label,
        "Edge grade": statistics["edge"]["label"],
        "Trades": statistics["total_trades"],
        "Win rate": f"{statistics['win_rate']:.1%}",
        "Average trade": f"{statistics['average_return']:+.2%}",
        "Profit factor": pf_text,
        "Out-of-sample": oos_text,
        "Stable periods": f"{stable_count}/{period_count}" if period_count else "—",
        "Compounded return": f"{statistics['total_return']:+.1%}",
        "Buy & hold": f"{statistics['buy_hold_return']:+.1%}",
        "Exposure": f"{statistics['exposure']:.0%}",
        "Max drawdown": f"{statistics['max_drawdown']:.1%}",
    }


def simulate_atr_trade_exit(
    prepared,
    entry_position,
    maximum_holding_days,
    direction,
    entry_price,
    atr_at_signal,
    stop_atr_multiple=1.25,
    reward_to_risk=2.0,
):
    """
    Simulate a trade using known-at-entry ATR risk levels.

    Daily OHLC data cannot reveal which level was touched first when both the
    stop and target occur in the same candle. The conservative assumption is
    that the stop was hit first.
    """
    atr_value = max(float(atr_at_signal), entry_price * 0.005, 0.01)
    risk_per_share = max(atr_value * stop_atr_multiple, entry_price * 0.005, 0.01)

    if direction == "LONG":
        stop_price = max(0.01, entry_price - risk_per_share)
        target_price = entry_price + (entry_price - stop_price) * reward_to_risk
    else:
        stop_price = entry_price + risk_per_share
        target_price = max(0.01, entry_price - (stop_price - entry_price) * reward_to_risk)

    final_position = min(
        entry_position + maximum_holding_days - 1,
        len(prepared) - 1,
    )
    exit_position = final_position
    exit_price = float(prepared["Close"].iloc[final_position])
    exit_reason = "TIME EXIT"

    for position in range(entry_position, final_position + 1):
        day_high = float(prepared["High"].iloc[position])
        day_low = float(prepared["Low"].iloc[position])

        if direction == "LONG":
            stop_touched = day_low <= stop_price
            target_touched = day_high >= target_price
        else:
            stop_touched = day_high >= stop_price
            target_touched = day_low <= target_price

        if stop_touched and target_touched:
            exit_position = position
            exit_price = stop_price
            exit_reason = "STOP FIRST (AMBIGUOUS BAR)"
            break
        if stop_touched:
            exit_position = position
            exit_price = stop_price
            exit_reason = "STOP"
            break
        if target_touched:
            exit_position = position
            exit_price = target_price
            exit_reason = "TARGET"
            break

    return {
        "exit_position": exit_position,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "risk_per_share": float(abs(entry_price - stop_price)),
    }



def run_strategy_backtest(
    data,
    test_start_date,
    holding_days,
    minimum_quality,
    cost_bps_per_side,
    stop_atr_multiple=1.25,
    reward_to_risk=2.0,
):
    """
    Backtest signals without look-ahead bias using realistic trade management.

    Signals are formed only after a completed daily close. Entry occurs at the
    next session's open. ATR stop and target levels are fixed from information
    known at the signal close. Only one position can be open at a time.
    """
    prepared = remove_unfinished_daily_bar(data)
    prepared = prepared.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    prepared = add_backtest_indicators(prepared)

    required_columns = [
        "Open", "High", "Low", "Close", "MA20", "MA50", "RSI", "MACD",
        "Signal", "Histogram", "Upper Band", "Lower Band",
        "Average Volume 20", "ATR14",
    ]
    prepared = prepared.dropna(subset=required_columns)

    trades = []
    index_position = 1

    while index_position < len(prepared) - 1:
        signal_row = prepared.iloc[index_position]
        previous_row = prepared.iloc[index_position - 1]
        signal_date = pd.Timestamp(prepared.index[index_position])

        if signal_date.date() < test_start_date:
            index_position += 1
            continue

        setup = calculate_trade_setup(
            price=float(signal_row["Close"]),
            previous_price=float(previous_row["Close"]),
            ma20=float(signal_row["MA20"]),
            ma50=float(signal_row["MA50"]),
            rsi=float(signal_row["RSI"]),
            macd=float(signal_row["MACD"]),
            signal=float(signal_row["Signal"]),
            histogram=float(signal_row["Histogram"]),
            previous_histogram=float(previous_row["Histogram"]),
            upper_band=float(signal_row["Upper Band"]),
            lower_band=float(signal_row["Lower Band"]),
            latest_volume=int(signal_row["Volume"]),
            average_volume_20=float(signal_row["Average Volume 20"]),
        )

        if setup["bias"] not in ("LONG BIAS", "SHORT BIAS"):
            index_position += 1
            continue
        if setup["setup_quality"] < minimum_quality:
            index_position += 1
            continue

        entry_position = index_position + 1
        if entry_position >= len(prepared):
            break

        entry_price = float(prepared["Open"].iloc[entry_position])
        if entry_price <= 0:
            index_position += 1
            continue

        direction = "LONG" if setup["bias"] == "LONG BIAS" else "SHORT"
        simulated = simulate_atr_trade_exit(
            prepared=prepared,
            entry_position=entry_position,
            maximum_holding_days=holding_days,
            direction=direction,
            entry_price=entry_price,
            atr_at_signal=float(signal_row["ATR14"]),
            stop_atr_multiple=stop_atr_multiple,
            reward_to_risk=reward_to_risk,
        )
        exit_position = simulated["exit_position"]
        exit_price = simulated["exit_price"]

        gross_return = (
            (exit_price / entry_price) - 1
            if direction == "LONG"
            else (entry_price / exit_price) - 1
        )
        round_trip_cost = 2 * cost_bps_per_side / 10000
        net_return = gross_return - round_trip_cost

        trades.append(
            {
                "Signal Date": signal_date,
                "Entry Date": pd.Timestamp(prepared.index[entry_position]),
                "Exit Date": pd.Timestamp(prepared.index[exit_position]),
                "Direction": direction,
                "Setup Quality": int(setup["setup_quality"]),
                "Direction Score": int(setup["direction_score"]),
                "Entry Price": entry_price,
                "Stop Price": simulated["stop_price"],
                "Target Price": simulated["target_price"],
                "Exit Price": exit_price,
                "Exit Reason": simulated["exit_reason"],
                "Gross Return": gross_return,
                "Net Return": net_return,
                "Winner": net_return > 0,
                "Holding Sessions": int(exit_position - entry_position + 1),
            }
        )

        index_position = exit_position + 1

    return pd.DataFrame(trades), prepared


def summarize_return_series(returns):
    """Return robust summary metrics for one chronological trade sample."""
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return None

    winning = returns[returns > 0]
    losing = returns[returns <= 0]
    gross_profit = float(winning.sum())
    gross_loss = float(abs(losing.sum()))
    profit_factor = float("inf") if gross_loss == 0 else gross_profit / gross_loss
    average_win = float(winning.mean()) if not winning.empty else 0.0
    average_loss = float(losing.mean()) if not losing.empty else 0.0
    payoff_ratio = (
        average_win / abs(average_loss)
        if average_loss < 0
        else float("inf") if average_win > 0 else 0.0
    )

    return {
        "trades": int(len(returns)),
        "win_rate": float((returns > 0).mean()),
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "profit_factor": float(profit_factor),
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": float(payoff_ratio),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
    }


def evaluate_backtest_edge(statistics):
    """Grade evidence using sample size, costs, out-of-sample results, and stability."""
    if not statistics:
        return {
            "grade": "INSUFFICIENT",
            "label": "No historical evidence",
            "score": -12,
            "reason": "No qualifying historical trades were found.",
        }

    total = statistics["total_trades"]
    full_average = statistics["average_return"]
    full_pf = statistics["profit_factor"]
    oos = statistics.get("out_of_sample") or {}
    oos_trades = int(oos.get("trades") or 0)
    oos_average = float(oos.get("average_return") or 0.0)
    oos_pf = float(oos.get("profit_factor") or 0.0)
    positive_periods = int(statistics.get("positive_stability_periods") or 0)
    tested_periods = int(statistics.get("stability_period_count") or 0)

    if total < 15 or oos_trades < 5:
        return {
            "grade": "INSUFFICIENT",
            "label": "Edge unproven",
            "score": -6,
            "reason": f"Only {total} trades and {oos_trades} out-of-sample trades were available.",
        }

    if full_average <= 0 or full_pf < 0.95 or oos_average < 0 or oos_pf < 0.90:
        return {
            "grade": "NEGATIVE",
            "label": "Historical edge failed",
            "score": -24,
            "reason": "Expectancy or out-of-sample performance was negative after costs.",
        }

    if tested_periods >= 3 and positive_periods < 2:
        return {
            "grade": "WEAK",
            "label": "Unstable historical edge",
            "score": -2,
            "reason": f"Only {positive_periods} of {tested_periods} chronological periods were positive.",
        }

    if (
        total >= 40
        and oos_trades >= 10
        and full_pf >= 1.35
        and oos_pf >= 1.15
        and full_average >= 0.002
        and oos_average >= 0.001
        and statistics["max_drawdown"] > -0.25
        and (tested_periods < 3 or positive_periods == tested_periods)
    ):
        return {
            "grade": "STRONG",
            "label": "Stronger historical edge",
            "score": 22,
            "reason": "Positive expectancy held up out of sample and across chronological periods.",
        }

    if (
        total >= 20
        and oos_trades >= 7
        and full_pf >= 1.15
        and oos_pf >= 1.0
        and full_average > 0
        and oos_average > 0
        and (tested_periods < 3 or positive_periods >= 2)
    ):
        return {
            "grade": "MODERATE",
            "label": "Moderate historical edge",
            "score": 12,
            "reason": "The strategy stayed positive after the chronological split and was reasonably stable.",
        }

    return {
        "grade": "WEAK",
        "label": "Weak historical edge",
        "score": 2,
        "reason": "Results were positive but not strong, stable, or large enough for confirmation.",
    }


def calculate_backtest_statistics(trades, prepared_history):
    """Calculate performance, exposure, benchmark context, and stability checks."""
    if trades.empty:
        return None

    ordered = trades.sort_values("Entry Date").reset_index(drop=True)
    returns = ordered["Net Return"].astype(float)
    strategy_growth = (1 + returns).cumprod()
    running_peak = strategy_growth.cummax()
    drawdown = (strategy_growth / running_peak) - 1
    full_summary = summarize_return_series(returns)

    split_position = max(1, int(len(ordered) * 0.70))
    if split_position >= len(ordered):
        split_position = max(1, len(ordered) - 1)
    in_sample_trades = ordered.iloc[:split_position]
    out_of_sample_trades = ordered.iloc[split_position:]
    in_sample_summary = summarize_return_series(in_sample_trades["Net Return"])
    out_of_sample_summary = summarize_return_series(out_of_sample_trades["Net Return"])

    stability_periods = []
    if len(ordered) >= 9:
        boundaries = [0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered)]
        for period_number in range(3):
            subset = ordered.iloc[boundaries[period_number]:boundaries[period_number + 1]]
            summary = summarize_return_series(subset["Net Return"])
            if summary:
                stability_periods.append(
                    {
                        "period": period_number + 1,
                        "trades": summary["trades"],
                        "average_return": summary["average_return"],
                        "profit_factor": summary["profit_factor"],
                        "positive": summary["average_return"] > 0 and summary["profit_factor"] >= 1.0,
                    }
                )
    positive_stability_periods = sum(1 for period in stability_periods if period["positive"])

    first_entry_date = pd.Timestamp(ordered["Entry Date"].iloc[0])
    last_exit_date = pd.Timestamp(ordered["Exit Date"].iloc[-1])
    benchmark_prices = prepared_history.loc[
        (prepared_history.index >= first_entry_date)
        & (prepared_history.index <= last_exit_date),
        "Close",
    ]
    buy_hold_return = (
        float(benchmark_prices.iloc[-1]) / float(benchmark_prices.iloc[0]) - 1
        if len(benchmark_prices) >= 2
        else 0.0
    )
    if len(benchmark_prices) >= 2:
        benchmark_growth = benchmark_prices.astype(float) / float(benchmark_prices.iloc[0])
        benchmark_drawdown = (benchmark_growth / benchmark_growth.cummax()) - 1
        buy_hold_max_drawdown = float(benchmark_drawdown.min())
    else:
        buy_hold_max_drawdown = 0.0

    if "Holding Sessions" in ordered:
        invested_sessions = int(ordered["Holding Sessions"].sum())
    else:
        invested_sessions = int(
            sum(
                max(1, len(prepared_history.loc[entry:exit]))
                for entry, exit in zip(ordered["Entry Date"], ordered["Exit Date"])
            )
        )
    exposure = min(1.0, invested_sessions / max(len(benchmark_prices), 1))

    equity_curve = pd.DataFrame(
        {"Strategy": 10000 * strategy_growth.values},
        index=pd.to_datetime(ordered["Exit Date"]),
    )
    if not benchmark_prices.empty:
        benchmark_at_exits = benchmark_prices.reindex(equity_curve.index, method="ffill")
        equity_curve["Buy and Hold"] = (
            10000 * benchmark_at_exits / float(benchmark_prices.iloc[0])
        )

    statistics = {
        "total_trades": len(ordered),
        "win_rate": full_summary["win_rate"],
        "average_return": full_summary["average_return"],
        "median_return": full_summary["median_return"],
        "total_return": float(strategy_growth.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": full_summary["profit_factor"],
        "payoff_ratio": full_summary["payoff_ratio"],
        "average_win": full_summary["average_win"],
        "average_loss": full_summary["average_loss"],
        "buy_hold_return": float(buy_hold_return),
        "buy_hold_max_drawdown": float(buy_hold_max_drawdown),
        "exposure": float(exposure),
        "in_sample": in_sample_summary,
        "out_of_sample": out_of_sample_summary,
        "split_date": (
            pd.Timestamp(out_of_sample_trades["Entry Date"].iloc[0])
            if not out_of_sample_trades.empty
            else None
        ),
        "stability_periods": stability_periods,
        "stability_period_count": len(stability_periods),
        "positive_stability_periods": positive_stability_periods,
        "equity_curve": equity_curve,
    }
    statistics["edge"] = evaluate_backtest_edge(statistics)
    return statistics


def build_direction_breakdown(trades):
    """Summarize long and short trades separately."""
    rows = []

    for direction in ["LONG", "SHORT"]:
        subset = trades[
            trades["Direction"] == direction
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "Direction": direction,
                "Trades": len(subset),
                "Win rate": subset["Winner"].mean(),
                "Average return": (
                    subset["Net Return"].mean()
                ),
                "Total compounded return": (
                    (1 + subset["Net Return"])
                    .prod()
                    - 1
                ),
            }
        )

    return pd.DataFrame(rows)


def format_backtest_trade_table(trades):
    """Format recent backtest trades for display."""
    if trades.empty:
        return trades

    display = trades.copy()

    for column in [
        "Signal Date",
        "Entry Date",
        "Exit Date",
    ]:
        display[column] = pd.to_datetime(
            display[column]
        ).dt.strftime("%Y-%m-%d")

    display["Entry Price"] = display[
        "Entry Price"
    ].map(lambda value: f"${value:,.2f}")

    display["Exit Price"] = display[
        "Exit Price"
    ].map(lambda value: f"${value:,.2f}")

    display["Gross Return"] = display[
        "Gross Return"
    ].map(lambda value: f"{value:+.2%}")

    display["Net Return"] = display[
        "Net Return"
    ].map(lambda value: f"{value:+.2%}")

    display["Result"] = display["Winner"].map(
        {
            True: "Win",
            False: "Loss",
        }
    )

    return display[
        [
            "Signal Date",
            "Entry Date",
            "Exit Date",
            "Direction",
            "Setup Quality",
            "Direction Score",
            "Entry Price",
            "Exit Price",
            "Net Return",
            "Result",
        ]
    ]



period_choices = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}

backtest_lookback_choices = {
    "2 years": 2,
    "5 years": 5,
    "10 years": 10,
}


# -----------------------------------------------------------------------------
# Clean, task-first interface
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1180px;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.06); padding: 0.8rem; border-radius: 0.75rem;}
    [data-testid="stMetricValue"] {white-space: normal; overflow-wrap: anywhere; line-height: 1.15;}
    [data-testid="stMetricLabel"] {white-space: normal; overflow-wrap: anywhere;}
    .summary-card {
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 0.75rem;
        padding: 0.72rem 0.82rem;
        min-height: 5.2rem;
        width: 100%;
        box-sizing: border-box;
        overflow: visible;
    }
    .summary-card-label {
        font-size: 0.82rem;
        opacity: 0.72;
        line-height: 1.2;
        margin-bottom: 0.35rem;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .summary-card-value {
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.25;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
    }
    div[data-testid="stButton"] > button {min-height: 2.7rem;}
    @media (max-width: 700px) {
        .block-container {padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 0.7rem;}
        [data-testid="stMetric"] {padding: 0.55rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_summary_card(label, value):
    """Render a flexible summary card that safely wraps long labels and values."""
    from html import escape

    card_html = (
        '<div class="summary-card">'
        f'<div class="summary-card-label">{escape(str(label))}</div>'
        f'<div class="summary-card-value">{escape(str(value))}</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Trade Finder"
if "analyze_ticker_input" not in st.session_state:
    st.session_state.analyze_ticker_input = "AAL"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "finder_results" not in st.session_state:
    st.session_state.finder_results = None
if "finder_scan_summary" not in st.session_state:
    st.session_state.finder_scan_summary = None
if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "decision_from_finder" not in st.session_state:
    st.session_state.decision_from_finder = False
if "pending_analysis_ticker" not in st.session_state:
    st.session_state.pending_analysis_ticker = None
if "finder_scan_mode" not in st.session_state:
    st.session_state.finder_scan_mode = "Scan the Market"

supabase, supabase_error = get_supabase_client()
logged_in = bool(st.session_state.get("supabase_user_id"))



def open_analysis_for(symbol):
    """Open a finder result directly, preserving its verified verdict."""
    clean_symbol = symbol.strip().upper()
    matched = next(
        (
            item for item in st.session_state.get("finder_results", [])
            if item.get("ticker") == clean_symbol and "error" not in item
        ),
        None,
    )
    st.session_state.analyze_ticker_input = clean_symbol
    st.session_state.pending_analysis_ticker = None if matched else clean_symbol
    st.session_state.decision_from_finder = True
    st.session_state.nav_page = "Analyze"
    st.session_state.analysis_result = matched

def return_to_trade_finder():
    st.session_state.nav_page = "Trade Finder"
    st.session_state.decision_from_finder = False
    st.session_state.pending_analysis_ticker = None


def analyze_another_stock():
    st.session_state.decision_from_finder = False
    st.session_state.pending_analysis_ticker = None
    st.session_state.analysis_result = None


def open_research_for(symbol):
    st.session_state.research_ticker_input = symbol
    st.session_state.nav_page = "Research"


def render_account_sidebar():
    with st.sidebar:
        st.header("Account")

        if supabase_error:
            st.error(supabase_error)
            st.caption("Analysis works, but cloud trade saving is unavailable.")

        elif not logged_in:
            sign_in_tab, create_account_tab = st.tabs(["Sign in", "Create account"])

            with sign_in_tab:
                with st.form("sign_in_form"):
                    email = st.text_input("Email", key="sign_in_email").strip()
                    password = st.text_input("Password", type="password", key="sign_in_password")
                    clicked = st.form_submit_button("Sign in", use_container_width=True)

                if clicked:
                    if not email or not password:
                        st.error("Enter your email and password.")
                    else:
                        try:
                            response = supabase.auth.sign_in_with_password(
                                {"email": email, "password": password}
                            )
                            remember_auth_response(response)
                            st.rerun()
                        except Exception as error:
                            st.error(f"Sign-in failed: {error}")

            with create_account_tab:
                with st.form("create_account_form"):
                    email = st.text_input("Email", key="create_email").strip()
                    password = st.text_input("Password", type="password", key="create_password")
                    again = st.text_input("Confirm password", type="password", key="create_password_again")
                    clicked = st.form_submit_button("Create account", use_container_width=True)

                if clicked:
                    if not email or not password:
                        st.error("Enter an email and password.")
                    elif password != again:
                        st.error("The passwords do not match.")
                    elif len(password) < 8:
                        st.error("Use a password with at least 8 characters.")
                    else:
                        try:
                            response = supabase.auth.sign_up(
                                {"email": email, "password": password}
                            )
                            remember_auth_response(response)
                            if response.session:
                                st.rerun()
                            else:
                                st.success("Account created. Confirm the email, then sign in.")
                        except Exception as error:
                            st.error(f"Account creation failed: {error}")
        else:
            email = st.session_state.get("supabase_user_email", "Signed-in user")
            st.success(f"Signed in as {email}")

            try:
                compact_trades = load_cloud_trades(
                    supabase,
                    st.session_state.supabase_user_id,
                )
                st.metric("Open trades", len(compact_trades))
            except Exception:
                pass

            if st.button("Sign out", use_container_width=True):
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                clear_auth_state()
                st.rerun()

        st.divider()
        st.caption("Account and app settings live here. Trading workflows use the main pages.")


@st.cache_data(ttl=120, show_spinner=False)
def build_stock_snapshot(ticker, period="1y"):
    symbol = ticker.strip().upper()
    stock = yf.Ticker(symbol)
    history = stock.history(period=period, auto_adjust=False)
    history = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if len(history) < 55:
        raise ValueError("Not enough daily price history was returned.")

    history["MA20"] = history["Close"].rolling(20).mean()
    history["MA50"] = history["Close"].rolling(50).mean()
    std = history["Close"].rolling(20).std()
    history["Upper Band"] = history["MA20"] + 2 * std
    history["Lower Band"] = history["MA20"] - 2 * std

    movement = history["Close"].diff()
    gains = movement.clip(lower=0).rolling(14).mean()
    losses = -movement.clip(upper=0).rolling(14).mean()
    rs = gains / losses
    history["RSI"] = 100 - (100 / (1 + rs))
    history.loc[(gains == 0) & (losses == 0), "RSI"] = 50.0
    history.loc[(gains > 0) & (losses == 0), "RSI"] = 100.0
    history.loc[(gains == 0) & (losses > 0), "RSI"] = 0.0

    ema12 = history["Close"].ewm(span=12, adjust=False).mean()
    ema26 = history["Close"].ewm(span=26, adjust=False).mean()
    history["MACD"] = ema12 - ema26
    history["Signal"] = history["MACD"].ewm(span=9, adjust=False).mean()
    history["Histogram"] = history["MACD"] - history["Signal"]
    history["Average Volume 20"] = history["Volume"].rolling(20).mean()

    prior_close = history["Close"].shift(1)
    true_range = pd.concat(
        [
            history["High"] - history["Low"],
            (history["High"] - prior_close).abs(),
            (history["Low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    history["ATR14"] = true_range.rolling(14).mean()

    latest = history.iloc[-1]
    previous = history.iloc[-2]
    rsi = float(latest["RSI"])
    if math.isnan(rsi):
        rsi = 50.0

    setup = calculate_trade_setup(
        price=float(latest["Close"]),
        previous_price=float(previous["Close"]),
        ma20=float(latest["MA20"]),
        ma50=float(latest["MA50"]),
        rsi=rsi,
        macd=float(latest["MACD"]),
        signal=float(latest["Signal"]),
        histogram=float(latest["Histogram"]),
        previous_histogram=float(previous["Histogram"]),
        upper_band=float(latest["Upper Band"]),
        lower_band=float(latest["Lower Band"]),
        latest_volume=int(latest["Volume"]),
        average_volume_20=float(latest["Average Volume 20"]),
    )

    quote = get_latest_quote(symbol)
    atr = float(latest["ATR14"])
    if math.isnan(atr) or atr <= 0:
        atr = max(float(latest["Close"]) * 0.02, 0.01)

    direction = None
    plan = None
    if setup["bias"] != "WAIT":
        direction = "LONG" if setup["bias"] == "LONG BIAS" else "SHORT"
        plan = build_suggested_trade_plan(
            ticker=symbol,
            direction=direction,
            fallback_price=float(latest["Close"]),
            atr_14=atr,
            entry_price_override=quote.get("price") or float(latest["Close"]),
        )

    return {
        "ticker": symbol,
        "history": history,
        "quote": quote,
        "setup": setup,
        "direction": direction,
        "plan": plan,
        "close": float(latest["Close"]),
        "previous_close": float(previous["Close"]),
        "ma20": float(latest["MA20"]),
        "ma50": float(latest["MA50"]),
        "rsi": rsi,
        "macd": float(latest["MACD"]),
        "signal": float(latest["Signal"]),
        "histogram": float(latest["Histogram"]),
        "atr": atr,
    }


def run_unconditional_long_study(data, test_start_date, holding_days, cost_bps_per_side):
    """Provide a same-horizon baseline for the direct oil-shock study."""
    prepared = remove_unfinished_daily_bar(data).dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )
    trades = []
    index_position = 0

    while index_position < len(prepared) - holding_days:
        signal_date = pd.Timestamp(prepared.index[index_position])
        if signal_date.date() < test_start_date:
            index_position += 1
            continue

        entry_position = index_position + 1
        exit_position = entry_position + holding_days - 1
        if exit_position >= len(prepared):
            break

        entry_price = float(prepared["Open"].iloc[entry_position])
        exit_price = float(prepared["Close"].iloc[exit_position])
        if entry_price > 0 and exit_price > 0:
            gross_return = (exit_price / entry_price) - 1
            net_return = gross_return - (2 * cost_bps_per_side / 10000)
            trades.append(
                {
                    "Signal Date": signal_date,
                    "Entry Date": pd.Timestamp(prepared.index[entry_position]),
                    "Exit Date": pd.Timestamp(prepared.index[exit_position]),
                    "Direction": "LONG",
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Gross Return": gross_return,
                    "Net Return": net_return,
                    "Winner": net_return > 0,
                }
            )
        index_position = exit_position + 1

    return pd.DataFrame(trades), prepared


BROAD_LIQUID_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "AVGO", "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "ARM", "SMCI",
    "NFLX", "CRM", "ORCL", "ADBE", "PLTR", "SNOW", "SHOP", "UBER",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "COF", "XLF", "KRE",
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "MPC", "XLE",
    "AAL", "DAL", "UAL", "LUV", "ALK", "JBLU", "BA", "JETS",
    "CAT", "DE", "GE", "RTX", "LMT", "NOC", "HON", "UPS", "FDX",
    "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
    "KO", "PEP", "PG", "PM", "MO", "DIS", "CMCSA", "T", "VZ",
    "JNJ", "LLY", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT",
    "NEM", "GOLD", "AEM", "GDX", "GLD", "SLV",
    "SPY", "QQQ", "IWM", "DIA", "SMH", "SOXX", "XLK", "XLI", "XLY",
    "RIVN", "LCID", "F", "GM", "CCL", "NCLH", "RCL", "MARA", "RIOT",
    "COIN", "HOOD", "SOFI", "PYPL", "SQ", "AFRM", "DKNG", "ROKU",
]


def get_market_session(now_eastern=None):
    """Return a time-based U.S. market session in Eastern time."""
    now_eastern = now_eastern or datetime.now(ZoneInfo("America/New_York"))
    current_time = now_eastern.time()

    if now_eastern.weekday() >= 5:
        return {"name": "closed", "label": "Market closed", "now": now_eastern}
    if clock_time(4, 0) <= current_time < clock_time(9, 30):
        return {"name": "premarket", "label": "Premarket", "now": now_eastern}
    if clock_time(9, 30) <= current_time < clock_time(16, 0):
        return {"name": "regular", "label": "Regular market", "now": now_eastern}
    if clock_time(16, 0) <= current_time < clock_time(20, 0):
        return {"name": "afterhours", "label": "After-hours", "now": now_eastern}
    return {"name": "closed", "label": "Market closed", "now": now_eastern}


def quote_session_metrics(quote, session_name):
    """Extract the most appropriate quote fields for the active session."""
    if session_name == "premarket":
        price = extract_number(quote.get("preMarketPrice"))
        change_percent = extract_number(quote.get("preMarketChangePercent"))
        volume = extract_number(quote.get("preMarketVolume"))
    elif session_name == "afterhours":
        price = extract_number(quote.get("postMarketPrice"))
        change_percent = extract_number(quote.get("postMarketChangePercent"))
        volume = extract_number(quote.get("postMarketVolume"))
    else:
        price = extract_number(quote.get("regularMarketPrice") or quote.get("intradayprice"))
        change_percent = extract_number(quote.get("regularMarketChangePercent"))
        volume = extract_number(
            quote.get("regularMarketVolume") or quote.get("dayvolume") or quote.get("eodvolume")
        )

    return price, change_percent, volume


@st.cache_data(ttl=180, show_spinner=False)
def get_market_scan_candidates(max_pool=220):
    """Collect a broad online mover pool before expensive stock analysis."""
    session = get_market_session()
    screens = [
        ("most_actives", "Most active"),
        ("day_gainers", "Day gainer"),
        ("day_losers", "Day loser"),
    ]
    candidates = {}

    for screen_name, source_label in screens:
        try:
            response = yf.screen(screen_name, count=100)
            quotes = response.get("quotes", [])
        except Exception:
            quotes = []

        for quote in quotes:
            symbol = str(quote.get("symbol") or "").strip().upper()
            if not symbol or symbol.startswith("^") or "=" in symbol:
                continue
            quote_type = str(quote.get("quoteType") or "").upper()
            if quote_type and quote_type not in {"EQUITY", "ETF"}:
                continue

            price, change_percent, volume = quote_session_metrics(quote, session["name"])
            regular_price = extract_number(quote.get("regularMarketPrice") or quote.get("intradayprice"))
            if price is None:
                price = regular_price

            item = candidates.setdefault(
                symbol,
                {
                    "ticker": symbol,
                    "sources": [],
                    "session_price": price,
                    "session_change_percent": change_percent,
                    "session_volume": volume,
                },
            )
            if source_label not in item["sources"]:
                item["sources"].append(source_label)
            if item.get("session_change_percent") is None and change_percent is not None:
                item["session_change_percent"] = change_percent
            if item.get("session_volume") is None and volume is not None:
                item["session_volume"] = volume
            if item.get("session_price") is None and price is not None:
                item["session_price"] = price

    for symbol in BROAD_LIQUID_UNIVERSE:
        candidates.setdefault(
            symbol,
            {
                "ticker": symbol,
                "sources": ["Liquid market universe"],
                "session_price": None,
                "session_change_percent": None,
                "session_volume": None,
            },
        )

    return {
        "session_name": session["name"],
        "session_label": session["label"],
        "candidates": list(candidates.values())[:max_pool],
    }


def _download_field(frame, symbol, field):
    """Extract one ticker/field series from a yfinance batch download."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    if isinstance(frame.columns, pd.MultiIndex):
        first = frame.columns.get_level_values(0)
        second = frame.columns.get_level_values(1)
        try:
            if symbol in first:
                selected = frame[symbol]
                return selected[field].dropna() if field in selected else pd.Series(dtype=float)
            if symbol in second and field in first:
                return frame[field][symbol].dropna()
        except Exception:
            return pd.Series(dtype=float)
    if field in frame.columns:
        return frame[field].dropna()
    return pd.Series(dtype=float)


def _eastern_index(index):
    converted = pd.DatetimeIndex(pd.to_datetime(index))
    if converted.tz is None:
        converted = converted.tz_localize("America/New_York")
    else:
        converted = converted.tz_convert("America/New_York")
    return converted


@st.cache_data(ttl=60, show_spinner=False)
def get_extended_session_metrics(symbols, session_name):
    """Batch-calculate premarket or after-hours movement using extended bars."""
    symbols = tuple(symbols)
    if not symbols or session_name not in {"premarket", "afterhours"}:
        return {}

    now_eastern = datetime.now(ZoneInfo("America/New_York"))
    today = now_eastern.date()
    results = {}

    for chunk_start in range(0, len(symbols), 45):
        chunk = list(symbols[chunk_start:chunk_start + 45])
        try:
            intraday = yf.download(
                tickers=chunk,
                period="5d",
                interval="5m",
                prepost=True,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            daily = yf.download(
                tickers=chunk,
                period="10d",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for symbol in chunk:
            close = _download_field(intraday, symbol, "Close")
            volume = _download_field(intraday, symbol, "Volume")
            daily_close = _download_field(daily, symbol, "Close")
            if close.empty or daily_close.empty:
                continue

            try:
                close_index = _eastern_index(close.index)
                close = pd.Series(close.values, index=close_index).dropna()
                volume = pd.Series(volume.values, index=_eastern_index(volume.index)).fillna(0) if not volume.empty else pd.Series(dtype=float)

                if session_name == "premarket":
                    mask = (
                        (close.index.date == today)
                        & (close.index.time >= clock_time(4, 0))
                        & (close.index.time < clock_time(9, 30))
                    )
                else:
                    mask = (
                        (close.index.date == today)
                        & (close.index.time >= clock_time(16, 0))
                        & (close.index.time < clock_time(20, 0))
                    )

                session_close = close[mask]
                if session_close.empty:
                    continue

                latest_price = float(session_close.iloc[-1])
                daily_index = _eastern_index(daily_close.index)
                daily_series = pd.Series(daily_close.values, index=daily_index).dropna()
                if session_name == "premarket":
                    prior = daily_series[daily_series.index.date < today]
                else:
                    prior = daily_series[daily_series.index.date <= today]
                if prior.empty:
                    continue
                previous_close = float(prior.iloc[-1])
                if previous_close <= 0:
                    continue

                if not volume.empty:
                    session_volume = float(volume.reindex(session_close.index).fillna(0).sum())
                else:
                    session_volume = 0.0

                results[symbol] = {
                    "session_price": latest_price,
                    "session_change_percent": ((latest_price / previous_close) - 1) * 100,
                    "session_volume": session_volume,
                }
            except Exception:
                continue

    return results


def rank_market_scan_candidates(candidate_items, session_name, deep_limit=24):
    """Quickly rank a broad pool, then return a practical deep-analysis set."""
    items = [dict(item) for item in candidate_items]

    if session_name in {"premarket", "afterhours"}:
        extended = get_extended_session_metrics(
            tuple(item["ticker"] for item in items),
            session_name,
        )
        for item in items:
            if item["ticker"] in extended:
                item.update(extended[item["ticker"]])
                session_source = "Extended-hours data"
                if session_source not in item["sources"]:
                    item["sources"].append(session_source)

    ranked = []
    fallback = []
    for item in items:
        price = item.get("session_price")
        change = item.get("session_change_percent")
        volume = item.get("session_volume")

        if price is not None and price < 3:
            continue
        if change is None:
            fallback.append(item)
            continue

        minimum_volume = 20_000 if session_name in {"premarket", "afterhours"} else 500_000
        if volume is not None and volume < minimum_volume:
            continue

        liquidity_component = math.log10(max(float(volume or 1), 1))
        item["quick_rank"] = abs(float(change)) * 2.0 + liquidity_component * 0.35
        ranked.append(item)

    ranked.sort(key=lambda item: item.get("quick_rank", 0), reverse=True)
    selected = ranked[:deep_limit]

    if len(selected) < deep_limit:
        selected_symbols = {item["ticker"] for item in selected}
        for item in fallback:
            if item["ticker"] not in selected_symbols:
                selected.append(item)
                selected_symbols.add(item["ticker"])
            if len(selected) >= deep_limit:
                break

    return selected, {
        "pool_count": len(items),
        "ranked_count": len(ranked),
        "deep_count": len(selected),
        "session_name": session_name,
    }



@st.cache_data(ttl=1800, show_spinner=False)
def load_finder_validation_history(ticker):
    """Load enough adjusted history to validate a current finder direction."""
    history = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
    return history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


@st.cache_data(ttl=900, show_spinner=False)
def get_current_ticker_macro_assessment(ticker, direction):
    """Return current ticker-aware macro alignment for a proposed direction."""
    classification = resolve_ticker_macro_profile(ticker, "Auto by ticker")
    plan = build_ticker_factor_plan(classification, oil_threshold=0.02)
    requested = tuple(sorted({factor["key"] for factor in plan}))
    start_date = (pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(days=260)).date().isoformat()
    history = load_macro_history(start_date, requested)
    context = build_macro_context(history, lookback_days=5)
    table, score = build_current_factor_table(context, plan)

    if direction == "LONG":
        alignment = "Supportive" if score >= 2 else "Opposing" if score <= -2 else "Neutral"
    else:
        alignment = "Supportive" if score <= -2 else "Opposing" if score >= 2 else "Neutral"

    relevant = table[table["Weight"] != "Ignored"] if not table.empty else table
    support = relevant[relevant["Current effect"] == "Supportive"]["Factor"].tolist()
    risks = relevant[relevant["Current effect"] == "Negative"]["Factor"].tolist()
    return {
        "alignment": alignment,
        "score": int(score),
        "profile": classification["profile"],
        "support": support[:3],
        "risks": risks[:3],
    }


@st.cache_data(ttl=21600, show_spinner=False)
def get_earnings_context(ticker):
    """Return the nearest known earnings event and a simple gap-risk label."""
    symbol = ticker.strip().upper()
    now = pd.Timestamp.now(tz="America/New_York")
    candidates = []
    timing = None

    stock = yf.Ticker(symbol)

    try:
        calendar = stock.calendar
        if isinstance(calendar, pd.DataFrame):
            calendar = calendar.to_dict()
        if isinstance(calendar, dict):
            for key, value in calendar.items():
                key_text = str(key).lower()
                if "earning" not in key_text:
                    continue
                values = value if isinstance(value, (list, tuple, pd.Series)) else [value]
                for item in values:
                    parsed = pd.to_datetime(item, errors="coerce", utc=True)
                    if pd.notna(parsed):
                        candidates.append(parsed.tz_convert("America/New_York"))
                if "time" in key_text and value:
                    timing = str(value)
    except Exception:
        pass

    try:
        dates = stock.get_earnings_dates(limit=8)
        if isinstance(dates, pd.DataFrame) and not dates.empty:
            for item in dates.index:
                parsed = pd.to_datetime(item, errors="coerce", utc=True)
                if pd.notna(parsed):
                    candidates.append(parsed.tz_convert("America/New_York"))
    except Exception:
        pass

    future = sorted(
        event for event in candidates
        if event >= now - pd.Timedelta(days=1)
    )
    if not future:
        return {
            "date": None,
            "days_away": None,
            "risk": "Unknown",
            "label": "Earnings date unavailable",
            "timing": timing,
        }

    event = future[0]
    days_away = int((event.date() - now.date()).days)
    if days_away <= 1:
        risk = "High"
        label = "Earnings within 1 day"
    elif days_away <= 5:
        risk = "Elevated"
        label = f"Earnings in {days_away} days"
    elif days_away <= 14:
        risk = "Upcoming"
        label = f"Earnings in {days_away} days"
    else:
        risk = "Low"
        label = f"Next earnings {event.strftime('%b %d')}"

    return {
        "date": event,
        "days_away": days_away,
        "risk": risk,
        "label": label,
        "timing": timing,
    }


def validate_finder_candidate(snapshot):
    """Turn a technical lean into a final verdict using history and macro context."""
    result = dict(snapshot)
    setup = result["setup"]
    direction = result.get("direction")

    if not direction:
        result["validation"] = {
            "final_verdict": "NO TRADE",
            "verdict_kind": "info",
            "historical_label": "No directional test",
            "macro_alignment": "Neutral",
            "final_score": 0,
            "reasons": ["The technical indicators do not agree on a direction."],
            "statistics": None,
        }
        return result

    history = load_finder_validation_history(result["ticker"])
    start_date = (pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(years=4)).date()
    trades, prepared = run_strategy_backtest(
        history,
        test_start_date=start_date,
        holding_days=3,
        minimum_quality=60,
        cost_bps_per_side=7.5,
        stop_atr_multiple=1.25,
        reward_to_risk=2.0,
    )
    direction_trades = trades[trades["Direction"] == direction].copy() if not trades.empty else trades
    statistics = calculate_backtest_statistics(direction_trades, prepared)
    edge = evaluate_backtest_edge(statistics)

    try:
        macro = get_current_ticker_macro_assessment(result["ticker"], direction)
    except Exception:
        macro = {
            "alignment": "Unavailable",
            "score": 0,
            "profile": "Unknown",
            "support": [],
            "risks": [],
        }

    try:
        earnings = get_earnings_context(result["ticker"])
    except Exception:
        earnings = {"risk": "Unknown", "label": "Earnings date unavailable", "date": None, "days_away": None}

    rr = float((result.get("plan") or {}).get("reward_to_risk") or 0.0)
    final_score = int(setup["setup_quality"]) + int(edge["score"])
    final_score += 8 if macro["alignment"] == "Supportive" else -12 if macro["alignment"] == "Opposing" else 0
    final_score += 5 if rr >= 1.5 else -8
    final_score += -18 if earnings["risk"] == "High" else -6 if earnings["risk"] == "Elevated" else 0

    if earnings["risk"] == "High":
        final_verdict = "WATCH — EARNINGS RISK"
        verdict_kind = "warning"
    elif edge["grade"] in {"STRONG", "MODERATE"} and setup["setup_quality"] >= 70 and macro["alignment"] != "Opposing" and rr >= 1.5:
        final_verdict = f"{direction} CANDIDATE"
        verdict_kind = "success"
    elif edge["grade"] == "NEGATIVE":
        final_verdict = "WAIT — HISTORICAL EDGE FAILED"
        verdict_kind = "error"
    elif macro["alignment"] == "Opposing":
        final_verdict = "WAIT — MACRO CONFLICT"
        verdict_kind = "warning"
    elif edge["grade"] == "INSUFFICIENT":
        final_verdict = "WATCH — EDGE UNPROVEN"
        verdict_kind = "warning"
    else:
        final_verdict = "WATCH / WAIT"
        verdict_kind = "warning"

    reasons = [
        f"Technical lean: {direction} with setup quality {setup['setup_quality']}/100.",
        f"Historical test: {edge['label']}. {edge['reason']}",
        f"Ticker-aware macro context: {macro['alignment']} ({macro['profile']} profile).",
        f"Earnings risk: {earnings['label']}.",
        f"Planned reward-to-risk: {rr:.1f}:1." if rr else "No valid reward-to-risk plan was available.",
    ]
    if statistics:
        oos = statistics.get("out_of_sample") or {}
        reasons.append(
            f"After costs: {statistics['total_trades']} {direction.lower()} trades, "
            f"{statistics['win_rate']:.1%} wins, {statistics['average_return']:+.2%} average, "
            f"profit factor {statistics['profit_factor']:.2f}."
        )
        if oos:
            reasons.append(
                f"Out-of-sample: {int(oos['trades'])} trades with "
                f"{float(oos['average_return']):+.2%} average return."
            )
        stable = int(statistics.get("positive_stability_periods") or 0)
        periods = int(statistics.get("stability_period_count") or 0)
        if periods:
            reasons.append(f"Stability check: {stable} of {periods} chronological periods were positive.")
        strategy_return = float(statistics.get("total_return") or 0.0)
        buy_hold_return = float(statistics.get("buy_hold_return") or 0.0)
        reasons.append(
            f"Benchmark context: strategy {strategy_return:+.1%} versus buy-and-hold {buy_hold_return:+.1%}, "
            f"with {float(statistics.get('exposure') or 0):.0%} market exposure."
        )
    if macro.get("support"):
        reasons.append("Macro support: " + ", ".join(macro["support"]) + ".")
    if macro.get("risks"):
        reasons.append("Macro risks: " + ", ".join(macro["risks"]) + ".")

    result["validation"] = {
        "final_verdict": final_verdict,
        "verdict_kind": verdict_kind,
        "historical_label": edge["label"],
        "historical_grade": edge["grade"],
        "macro_alignment": macro["alignment"],
        "macro_profile": macro["profile"],
        "earnings_risk": earnings["risk"],
        "earnings_label": earnings["label"],
        "earnings_date": earnings.get("date"),
        "final_score": final_score,
        "reasons": reasons,
        "statistics": statistics,
    }
    return result




def candidate_liquidity_assessment(candidate, session_name):
    """Grade current-session liquidity without pretending it is a live spread quote."""
    volume = candidate.get("session_volume")
    if volume is None:
        try:
            history = candidate.get("history")
            if history is not None and not history.empty:
                volume = float(history["Volume"].iloc[-1])
        except Exception:
            volume = None

    if volume is None:
        return {"label": "Unknown", "score": 0, "volume": None}

    volume = float(volume)
    if session_name in {"premarket", "afterhours"}:
        if volume >= 1_000_000:
            label, score = "Strong extended-hours volume", 6
        elif volume >= 250_000:
            label, score = "Usable extended-hours volume", 3
        elif volume >= 50_000:
            label, score = "Thin extended-hours volume", -5
        else:
            label, score = "Very thin extended-hours volume", -10
    else:
        if volume >= 10_000_000:
            label, score = "Very liquid", 6
        elif volume >= 2_000_000:
            label, score = "Good liquidity", 4
        elif volume >= 500_000:
            label, score = "Moderate liquidity", 0
        else:
            label, score = "Low liquidity", -8

    return {"label": label, "score": score, "volume": volume}


def candidate_confidence(candidate, session_name):
    """Convert verified evidence into a cautious confidence label."""
    validation = candidate.get("validation") or {}
    setup = candidate.get("setup") or {}
    statistics = validation.get("statistics") or {}
    edge_grade = validation.get("historical_grade")
    macro = validation.get("macro_alignment")
    earnings = validation.get("earnings_risk")
    verdict = validation.get("final_verdict", "")
    final_score = int(validation.get("final_score") or 0)
    setup_quality = int(setup.get("setup_quality") or 0)
    liquidity = candidate_liquidity_assessment(candidate, session_name)

    oos = statistics.get("out_of_sample") or {}
    oos_trades = int(oos.get("trades") or 0)
    oos_average = float(oos.get("average_return") or 0.0)
    stable = int(statistics.get("positive_stability_periods") or 0)
    stability_periods = int(statistics.get("stability_period_count") or 0)

    verified = "CANDIDATE" in verdict
    hard_block = (
        not verified
        or macro == "Opposing"
        or earnings == "High"
        or edge_grade in {"NEGATIVE", "INSUFFICIENT"}
        or liquidity["score"] <= -8
    )

    if hard_block:
        return {
            "label": "NO TRADE",
            "rank": 0,
            "selection_score": final_score + liquidity["score"],
            "liquidity": liquidity,
            "explanation": "One or more required confidence checks failed.",
        }

    high_confidence = (
        edge_grade == "STRONG"
        and setup_quality >= 80
        and macro == "Supportive"
        and earnings in {"Low", "Upcoming", "Unknown"}
        and final_score >= 100
        and oos_trades >= 10
        and oos_average > 0
        and (stability_periods < 3 or stable == stability_periods)
        and liquidity["score"] >= 0
    )
    if high_confidence:
        return {
            "label": "HIGH",
            "rank": 3,
            "selection_score": final_score + 18 + liquidity["score"],
            "liquidity": liquidity,
            "explanation": "Strong historical evidence, supportive context, and no major blocking risk.",
        }

    moderate_confidence = (
        edge_grade in {"STRONG", "MODERATE"}
        and setup_quality >= 70
        and macro != "Opposing"
        and earnings != "High"
        and oos_average > 0
        and final_score >= 78
        and liquidity["score"] > -8
    )
    if moderate_confidence:
        return {
            "label": "MODERATE",
            "rank": 2,
            "selection_score": final_score + 8 + liquidity["score"],
            "liquidity": liquidity,
            "explanation": "The evidence is positive, but not strong enough to call high confidence.",
        }

    return {
        "label": "LOW — WATCH ONLY",
        "rank": 1,
        "selection_score": final_score + liquidity["score"],
        "liquidity": liquidity,
        "explanation": "Some evidence is positive, but confidence requirements were not fully met.",
    }


def candidate_session_action(candidate, session):
    """Translate one verified setup into an action appropriate for the market session."""
    confidence = candidate.get("confidence") or candidate_confidence(candidate, session["name"])
    if confidence["rank"] < 2:
        return {
            "label": "NO TRADE",
            "detail": "Keep scanning or wait for conditions to improve.",
        }

    direction = candidate.get("direction")
    plan = candidate.get("plan") or {}
    quote = candidate.get("quote") or {}
    current_price = quote.get("price")
    entry = plan.get("entry")

    if session["name"] == "premarket":
        return {
            "label": "WATCH FOR THE OPEN",
            "detail": "Use the opening reaction to confirm the setup; do not treat a premarket quote as an automatic fill.",
        }
    if session["name"] == "afterhours":
        return {
            "label": "WATCH FOR NEXT SESSION",
            "detail": "After-hours liquidity can be thin. Recheck the setup before the next regular session.",
        }
    if session["name"] == "closed":
        return {
            "label": "PLAN FOR NEXT SESSION",
            "detail": "This is the strongest setup from the latest completed data, not an immediate trade.",
        }

    if current_price is not None and entry is not None and direction:
        current_price = float(current_price)
        entry = float(entry)
        chase_tolerance = max(0.02, entry * 0.004)
        if direction == "LONG" and current_price > entry + chase_tolerance:
            return {
                "label": "WAIT FOR ENTRY — DO NOT CHASE",
                "detail": f"The live price is above the planned entry near ${entry:.2f}.",
            }
        if direction == "SHORT" and current_price < entry - chase_tolerance:
            return {
                "label": "WAIT FOR ENTRY — DO NOT CHASE",
                "detail": f"The live price is below the planned short entry near ${entry:.2f}.",
            }

    return {
        "label": "TRADE SETUP AVAILABLE",
        "detail": "The planned entry remains close to the latest available price. Confirm the live quote before acting.",
    }


def enrich_and_rank_finder_results(results, session):
    """Attach confidence/action fields and rank strongest verified candidates first."""
    enriched = []
    for item in results:
        if "error" in item:
            continue
        item = dict(item)
        item["confidence"] = candidate_confidence(item, session["name"])
        item["session_action"] = candidate_session_action(item, session)
        enriched.append(item)

    enriched.sort(
        key=lambda item: (
            item["confidence"]["rank"],
            item["confidence"]["selection_score"],
            int((item.get("setup") or {}).get("setup_quality") or 0),
        ),
        reverse=True,
    )
    return enriched


def run_finder_scan(candidate_items, status):
    """Run the shared technical and deep-validation pipeline."""
    results = []
    progress = st.progress(0)
    for index, candidate in enumerate(candidate_items):
        symbol = candidate["ticker"]
        status.caption(f"Technical analysis {index + 1} of {len(candidate_items)} — {symbol}")
        try:
            snapshot = build_stock_snapshot(symbol, "1y")
            snapshot["finder_sources"] = candidate.get("sources", [])
            snapshot["session_change_percent"] = candidate.get("session_change_percent")
            snapshot["session_volume"] = candidate.get("session_volume")
            results.append(snapshot)
        except Exception as error:
            results.append({
                "ticker": symbol,
                "finder_sources": candidate.get("sources", []),
                "error": str(error),
            })
        progress.progress((index + 1) / max(len(candidate_items), 1))

    usable = [item for item in results if "error" not in item]
    validation_pool = sorted(
        [
            item for item in usable
            if item.get("direction") and item["setup"]["setup_quality"] >= 55
        ],
        key=lambda item: item["setup"]["setup_quality"],
        reverse=True,
    )[:8]
    validation_symbols = {item["ticker"] for item in validation_pool}

    for index, candidate in enumerate(validation_pool):
        status.caption(
            f"Deep validation {index + 1} of {len(validation_pool)} — {candidate['ticker']}"
        )
        try:
            validated = validate_finder_candidate(candidate)
            for position, existing in enumerate(results):
                if existing.get("ticker") == candidate["ticker"]:
                    results[position] = validated
                    break
        except Exception as error:
            candidate["validation"] = {
                "final_verdict": "WATCH — VALIDATION UNAVAILABLE",
                "verdict_kind": "warning",
                "historical_label": "Validation unavailable",
                "historical_grade": "INSUFFICIENT",
                "macro_alignment": "Unavailable",
                "earnings_risk": "Unknown",
                "earnings_label": "Earnings date unavailable",
                "final_score": candidate["setup"]["setup_quality"] - 10,
                "reasons": [f"Validation could not be completed: {error}"],
                "statistics": None,
            }

    for item in usable:
        if item["ticker"] not in validation_symbols:
            item["validation"] = {
                "final_verdict": "NOT FULLY VALIDATED",
                "verdict_kind": "info",
                "historical_label": "Outside validation shortlist",
                "historical_grade": "INSUFFICIENT",
                "macro_alignment": "Not checked",
                "earnings_risk": "Unknown",
                "earnings_label": "Not checked",
                "final_score": item["setup"]["setup_quality"] - 15,
                "reasons": [
                    "This ticker did not rank in the top eight technical leans, so slower validation was skipped."
                ],
                "statistics": None,
            }

    progress.empty()
    status.empty()
    return results


def render_best_trade_card(item, session):
    """Render the single primary Finder result."""
    setup = item["setup"]
    quote = item["quote"]
    plan = item.get("plan")
    validation = item["validation"]
    confidence = item["confidence"]
    action = item["session_action"]
    session_change = item.get("session_change_percent")
    sources = ", ".join(item.get("finder_sources") or [])

    with st.container(border=True):
        st.markdown(f"## {item['ticker']} — {item.get('direction') or 'WAIT'}")
        st.markdown(f"### {action['label']}")
        st.write(action["detail"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_summary_card("Confidence", confidence["label"])
        with c2:
            render_summary_card("Setup", f"{setup['setup_quality']} / 100")
        with c3:
            render_summary_card("Historical edge", validation.get("historical_label", "—"))
        with c4:
            render_summary_card("Macro", validation.get("macro_alignment", "—"))

        st.caption(
            "Confidence is an evidence grade, not a guaranteed probability of profit."
        )
        st.write(
            f"**Earnings:** {validation.get('earnings_label', 'Date unavailable')}  •  "
            f"**Liquidity:** {confidence['liquidity']['label']}  •  "
            "**Tested style:** short swing, up to 3 sessions"
        )
        if quote.get("price") is not None:
            latest_text = f"**Latest:** ${float(quote['price']):.2f}"
            if session_change is not None:
                latest_text += f" • **Session move:** {float(session_change):+.2f}%"
            st.write(latest_text)
        if plan:
            st.write(
                f"**Plan:** Entry ${plan['entry']:.2f} • Stop ${plan['stop']:.2f} • "
                f"Target ${plan['target']:.2f} • R:R {plan['reward_to_risk']:.1f}:1"
            )
        if sources:
            st.caption(f"Found through: {sources}")

        st.markdown("**Why this ranked first**")
        st.write(f"• {confidence['explanation']}")
        for reason in validation.get("reasons", [])[:4]:
            st.write(f"• {reason}")

        st.button(
            "View Full Decision",
            key=f"best_trade_view_{item['ticker']}",
            type="primary",
            use_container_width=True,
            on_click=open_analysis_for,
            args=(item["ticker"],),
        )


def render_trade_finder():
    st.header("Find Best Trade Now")
    st.caption("v23 • Session-aware ranked Trade Finder")
    st.caption(
        "One button searches the current session, deeply validates the strongest setups, "
        "and returns a ranked shortlist of qualified choices—or No Trade."
    )

    session = get_market_session()
    session_message = {
        "premarket": "The scanner uses extended-hours movement and looks for a setup to confirm at the open.",
        "regular": "The scanner uses current regular-session movers and checks whether an entry is still available.",
        "afterhours": "The scanner uses after-hours movement and looks for a setup to recheck next session.",
        "closed": "The scanner uses the latest completed data to prepare for the next session.",
    }[session["name"]]
    st.info(f"**{session['label']} mode.** {session_message}")

    market_scan_clicked = st.button(
        "🔎 Find Best Trade Now",
        type="primary",
        use_container_width=True,
    )

    custom_scan_clicked = False
    custom_symbols = []
    with st.expander("Optional: Scan My List"):
        finder_watchlist = st.text_input(
            "Tickers to scan",
            "AAL, INTC, BA, KO, CVX, XOM, NVDA, AMD",
            help="Separate ticker symbols with commas. Up to 12.",
        )
        custom_scan_clicked = st.button(
            "Scan My List",
            use_container_width=True,
        )
        if custom_scan_clicked:
            custom_symbols = parse_watchlist(finder_watchlist, "")[:12]

    if market_scan_clicked or custom_scan_clicked:
        status = st.empty()
        if market_scan_clicked:
            st.session_state.finder_scan_mode = "Scan the Market"
            status.caption("Collecting current movers and liquid stocks…")
            scan_payload = get_market_scan_candidates(max_pool=220)
            status.caption(
                f"Pre-ranking {len(scan_payload['candidates'])} candidates using "
                f"{scan_payload['session_label'].lower()} data…"
            )
            candidate_items, scan_summary = rank_market_scan_candidates(
                scan_payload["candidates"],
                scan_payload["session_name"],
                deep_limit=24,
            )
            scan_summary["session_label"] = scan_payload["session_label"]
            scan_summary["scanned_at"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
        else:
            st.session_state.finder_scan_mode = "Scan My List"
            candidate_items = [
                {
                    "ticker": symbol,
                    "sources": ["My list"],
                    "session_price": None,
                    "session_change_percent": None,
                    "session_volume": None,
                }
                for symbol in custom_symbols
            ]
            scan_summary = {
                "pool_count": len(candidate_items),
                "ranked_count": len(candidate_items),
                "deep_count": len(candidate_items),
                "session_name": session["name"],
                "session_label": session["label"],
                "scanned_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            }

        st.session_state.finder_scan_summary = scan_summary
        if not candidate_items:
            status.empty()
            st.session_state.finder_results = []
            st.warning("No candidate tickers were returned.")
        else:
            st.session_state.finder_results = run_finder_scan(candidate_items, status)

    results = st.session_state.get("finder_results")
    scan_summary = st.session_state.get("finder_scan_summary")
    if scan_summary:
        st.caption(
            f"Last scan: {scan_summary.get('session_label', 'Market')} • "
            f"{scan_summary.get('pool_count', 0)} collected • "
            f"{scan_summary.get('ranked_count', 0)} with usable mover data • "
            f"{scan_summary.get('deep_count', 0)} technically analyzed • up to 8 deeply validated"
        )

    if not results:
        st.info("Press **Find Best Trade Now** to run the session-aware scan.")
        return

    scanned_at = pd.to_datetime((scan_summary or {}).get("scanned_at"), errors="coerce")
    scan_session_name = (scan_summary or {}).get("session_name")
    stale_reasons = []
    if scan_session_name and scan_session_name != session["name"]:
        stale_reasons.append("the market session changed")
    if pd.isna(scanned_at):
        stale_reasons.append("the scan predates this version")
    else:
        if scanned_at.tzinfo is None:
            scanned_at = scanned_at.tz_localize("America/New_York")
        else:
            scanned_at = scanned_at.tz_convert("America/New_York")
        age_minutes = (pd.Timestamp.now(tz="America/New_York") - scanned_at).total_seconds() / 60
        freshness_limit = 20 if session["name"] == "regular" else 45
        if age_minutes > freshness_limit:
            stale_reasons.append(f"the results are about {int(age_minutes)} minutes old")

    if stale_reasons:
        st.warning(
            "These results are stale because " + " and ".join(stale_reasons) + ". "
            "Press **Find Best Trade Now** again before treating anything as the current best setup."
        )
        return

    ranked = enrich_and_rank_finder_results(results, session)
    confident = [item for item in ranked if item["confidence"]["rank"] >= 2]

    if confident:
        qualified = confident[:5]
        best = qualified[0]
        st.success(
            f"{len(qualified)} qualified choice{'s' if len(qualified) != 1 else ''} found. "
            f"Top-ranked: {best['ticker']} • {best['confidence']['label']} confidence • "
            f"{best['session_action']['label']}"
        )
        st.caption(
            "The first result ranks highest, but the other qualified choices remain visible so "
            "you can compare the setup, historical edge, macro support, and entry before deciding."
        )
        render_best_trade_card(best, session)

        alternatives = qualified[1:]
        if alternatives:
            st.subheader("Other qualified choices")
            for rank, item in enumerate(alternatives, start=2):
                validation = item["validation"]
                plan = item.get("plan")
                confidence = item.get("confidence") or {}
                with st.container(border=True):
                    st.markdown(
                        f"### {rank}. {item['ticker']} — {confidence.get('label', '—')} confidence"
                    )
                    st.write(
                        f"**Action:** {item['session_action']['label']}  •  "
                        f"**Technical:** {item.get('direction') or 'WAIT'}  •  "
                        f"**Historical:** {validation.get('historical_label', '—')}  •  "
                        f"**Macro:** {validation.get('macro_alignment', '—')}  •  "
                        f"**Earnings:** {validation.get('earnings_label', '—')}"
                    )
                    if plan:
                        st.write(
                            f"**Plan:** Entry ${plan['entry']:.2f} • Stop ${plan['stop']:.2f} • "
                            f"Target ${plan['target']:.2f} • R:R {plan['reward_to_risk']:.1f}:1"
                        )
                    reasons = validation.get("reasons", [])
                    if reasons:
                        st.caption("Why it qualified: " + " ".join(reasons[:2]))
                    st.button(
                        "View Full Decision",
                        key=f"ranked_choice_view_{rank}_{item['ticker']}",
                        use_container_width=True,
                        on_click=open_analysis_for,
                        args=(item["ticker"],),
                    )
        else:
            st.info(
                "Only one setup met the confidence requirements in this scan. "
                "The app is not adding weaker names just to create more choices."
            )
    else:
        st.warning("## No confident trade right now")
        st.write(
            "The strongest setups failed one or more required checks. The app is choosing No Trade "
            "instead of forcing a recommendation."
        )
        closest = ranked[:3]
        if closest:
            with st.expander("See the closest watch-only setups"):
                for item in closest:
                    validation = item.get("validation") or {}
                    st.markdown(
                        f"**{item['ticker']} — {validation.get('final_verdict', 'Watch')}**"
                    )
                    st.caption(
                        f"Technical {item.get('direction') or 'WAIT'} • "
                        f"Setup {item['setup']['setup_quality']}/100 • "
                        f"Historical {validation.get('historical_label', '—')} • "
                        f"Macro {validation.get('macro_alignment', '—')}"
                    )

    with st.expander("See all analyzed stocks and scan details"):
        rows = []
        for item in ranked:
            validation = item.get("validation") or {}
            confidence = item.get("confidence") or {}
            rows.append(
                {
                    "Ticker": item["ticker"],
                    "Source": ", ".join(item.get("finder_sources") or []),
                    "Session move": (
                        f"{item.get('session_change_percent'):+.2f}%"
                        if item.get("session_change_percent") is not None
                        else "—"
                    ),
                    "Technical lean": item.get("direction") or "WAIT",
                    "Setup": item["setup"]["setup_quality"],
                    "Confidence": confidence.get("label", "—"),
                    "Action": (item.get("session_action") or {}).get("label", "—"),
                    "Final verdict": validation.get("final_verdict", "Not validated"),
                    "Historical edge": validation.get("historical_label", "—"),
                    "Macro": validation.get("macro_alignment", "—"),
                    "Earnings": validation.get("earnings_label", "—"),
                }
            )
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )

def render_analyze():
    pending_ticker = st.session_state.get("pending_analysis_ticker")

    if pending_ticker:
        try:
            with st.spinner(f"Opening and validating {pending_ticker} decision…"):
                snapshot = build_stock_snapshot(pending_ticker, "1y")
                st.session_state.analysis_result = validate_finder_candidate(snapshot)
        except Exception as error:
            st.session_state.analysis_result = None
            st.error(f"Decision could not be opened: {error}")
        finally:
            st.session_state.pending_analysis_ticker = None

    result = st.session_state.analysis_result
    direct_decision = bool(st.session_state.get("decision_from_finder") and result)

    if direct_decision:
        navigation_1, navigation_2 = st.columns(2)
        navigation_1.button("← Back to Trade Finder", use_container_width=True, on_click=return_to_trade_finder)
        navigation_2.button("Analyze another ticker", use_container_width=True, on_click=analyze_another_stock)
        st.header(f"Decision for {result['ticker']}")
        st.caption("Opened directly from Trade Finder with its historical and macro validation preserved.")
    else:
        st.header("Analyze a Stock")
        st.caption("The app separates the current technical lean from the evidence-based final verdict.")
        with st.form("clean_analysis_form"):
            ticker = st.text_input("Ticker", key="analyze_ticker_input").strip().upper()
            analyze_clicked = st.form_submit_button("Analyze stock", type="primary", use_container_width=True)
        if analyze_clicked:
            st.session_state.decision_from_finder = False
            try:
                with st.spinner(f"Analyzing and validating {ticker}…"):
                    snapshot = build_stock_snapshot(ticker, "1y")
                    st.session_state.analysis_result = validate_finder_candidate(snapshot)
            except Exception as error:
                st.session_state.analysis_result = None
                st.error(f"Analysis failed: {error}")
        result = st.session_state.analysis_result

    if not result:
        st.info("Enter a ticker and press Analyze stock.")
        return

    ticker = result["ticker"]
    setup = result["setup"]
    quote = result["quote"]
    plan = result["plan"]
    validation = result.get("validation") or {
        "final_verdict": "TECHNICAL VIEW ONLY",
        "verdict_kind": "info",
        "historical_label": "Not validated",
        "macro_alignment": "Not checked",
        "reasons": [],
        "statistics": None,
    }
    final_is_candidate = "CANDIDATE" in validation["final_verdict"]

    with st.container(border=True):
        st.markdown(f"### Final verdict: {validation['final_verdict']}")
        top1, top2, top3, top4 = st.columns(4)
        with top1:
            render_summary_card("Latest price", f"${quote['price']:.2f}" if quote.get("price") else "Unavailable")
        with top2:
            render_summary_card("Technical lean", result.get("direction") or "WAIT")
        with top3:
            render_summary_card("Setup", f"{setup['setup_quality']} / 100")
        with top4:
            render_summary_card("Historical edge", validation["historical_label"])
        st.write(f"**Ticker-aware macro:** {validation['macro_alignment']}")
        st.write(f"**Earnings:** {validation.get('earnings_label', 'Date unavailable')}  •  **Tested style:** short swing, up to 3 sessions")

        for reason in validation.get("reasons", [])[:5]:
            st.write(f"• {reason}")

        if plan:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Direction", result["direction"])
            p2.metric("Entry", f"${plan['entry']:.2f}")
            p3.metric("Stop", f"${plan['stop']:.2f}")
            p4.metric("Target", f"${plan['target']:.2f}")
            if not final_is_candidate:
                st.caption("These levels are a hypothetical risk plan, not an app recommendation to enter.")

    if plan and final_is_candidate and logged_in:
        try:
            live_quote = get_latest_quote(ticker)
            live_price = float(live_quote.get("price"))
        except Exception:
            live_quote = quote
            live_price = None

        recommended_entry = float(plan["entry"])
        original_entry_waiting = entry_is_waiting(
            result["direction"],
            recommended_entry,
            live_price,
        )

        if original_entry_waiting and live_price is not None:
            st.warning(
                f"The original setup entry is not available now. Current price: ${live_price:.2f}. "
                f"Plan entry: ${recommended_entry:.2f}. Wait for the level or recalculate instead of chasing."
            )
            if st.button(
                "Recalculate setup from current price",
                use_container_width=True,
                key=f"recalculate_plan_{ticker}_{result['direction']}",
            ):
                refreshed_plan = build_suggested_trade_plan(
                    ticker=ticker,
                    direction=result["direction"],
                    fallback_price=result["close"],
                    atr_14=result["atr"],
                    entry_price_override=live_price,
                )
                st.session_state.analysis_result["plan"] = refreshed_plan
                st.session_state.analysis_result["quote"] = live_quote
                st.rerun()

        with st.expander("Add to Active Trades (optional)", expanded=False):
            st.caption(
                "Use this only after you actually enter the trade in a broker or paper account. "
                "Active Trades is an organizer, not an order queue."
            )
            with st.form(f"optional_active_entry_{ticker}_{result['direction']}"):
                c1, c2 = st.columns(2)
                with c1:
                    quantity = st.number_input(
                        "Shares",
                        min_value=1,
                        value=1,
                        step=1,
                        key=f"active_qty_{ticker}_{result['direction']}",
                    )
                    entry = st.number_input(
                        "Actual fill price",
                        min_value=0.01,
                        value=float(live_price or plan["entry"]),
                        step=0.01,
                        format="%.2f",
                        key=f"active_entry_{ticker}_{result['direction']}",
                    )
                    paper_trade = st.checkbox(
                        "Paper trade",
                        value=True,
                        key=f"active_paper_{ticker}_{result['direction']}",
                    )
                with c2:
                    stop = st.number_input(
                        "Stop loss",
                        min_value=0.01,
                        value=float(plan["stop"]),
                        step=0.01,
                        format="%.2f",
                        key=f"active_stop_{ticker}_{result['direction']}",
                    )
                    target = st.number_input(
                        "Target price",
                        min_value=0.01,
                        value=float(plan["target"]),
                        step=0.01,
                        format="%.2f",
                        key=f"active_target_{ticker}_{result['direction']}",
                    )
                confirmed = st.checkbox(
                    "I actually entered this trade at the fill price above",
                    key=f"active_confirm_{ticker}_{result['direction']}",
                )
                save_clicked = st.form_submit_button(
                    "Add entered trade",
                    type="primary",
                    use_container_width=True,
                )

            if save_clicked:
                direction = result["direction"]
                levels_valid = (
                    stop < entry < target
                    if direction == "LONG"
                    else target < entry < stop
                )
                if not levels_valid:
                    st.error(
                        "Check the levels. For a long: stop < entry < target. "
                        "For a short: target < entry < stop."
                    )
                elif not confirmed:
                    st.error("Confirm that you actually entered the trade before adding it to Active Trades.")
                else:
                    try:
                        add_cloud_trade(
                            supabase,
                            st.session_state.supabase_user_id,
                            {
                                "ticker": ticker,
                                "direction": direction,
                                "quantity": int(quantity),
                                "entry": float(entry),
                                "stop": float(stop),
                                "target": float(target),
                                "notes": (
                                    "PAPER TRADE | Entered from app decision"
                                    if paper_trade
                                    else "Entered from app decision"
                                ),
                            },
                        )
                        st.success(f"{ticker} was added to Active Trades.")
                    except Exception as error:
                        st.error(f"Trade could not be saved: {error}")
    elif plan and not final_is_candidate:
        st.info("Trade entry is withheld because the final evidence does not confirm this setup. You can still add a trade manually from Active Trades.")
    elif plan and not logged_in:
        st.info("Sign in from the sidebar to save a verified trade.")

    with st.expander("Full decision evidence"):
        st.markdown("**Current technical evidence**")
        for reason in setup["evidence"]:
            st.write(f"• {reason}")
        if setup["risk_flags"]:
            st.markdown("**Technical risk flags**")
            for flag in setup["risk_flags"]:
                st.write(f"• {flag}")
        stats = validation.get("statistics")
        if stats:
            st.markdown("**Historical validation**")
            st.dataframe(
                pd.DataFrame([build_strategy_comparison_row(f"{result.get('direction')} signals", stats)]),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "The last 30% of trades are held out as a chronological out-of-sample check. "
                "Buy-and-hold is always invested; the strategy exposure is shown separately."
            )

    with st.expander("Technical details"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("RSI", f"{result['rsi']:.1f}")
        d2.metric("MACD", f"{result['macd']:.3f}")
        d3.metric("20-day average", f"${result['ma20']:.2f}")
        d4.metric("50-day average", f"${result['ma50']:.2f}")

    with st.expander("Price chart"):
        chart = go.Figure()
        history = result["history"]
        chart.add_trace(go.Scatter(x=history.index, y=history["Close"], mode="lines", name="Close"))
        chart.add_trace(go.Scatter(x=history.index, y=history["MA20"], mode="lines", name="MA20"))
        chart.add_trace(go.Scatter(x=history.index, y=history["MA50"], mode="lines", name="MA50"))
        chart.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), legend_orientation="h")
        st.plotly_chart(chart, use_container_width=True)

    st.button("Research this ticker", use_container_width=True, on_click=open_research_for, args=(ticker,))

def render_active_trades():
    st.header("Active Trades")
    st.caption(
        "Optional organizer for trades you actually entered from the app. "
        "It tracks price, P/L, stop, target, and your history; it does not place or queue orders."
    )

    if not logged_in:
        st.info("Sign in from the sidebar to organize active trades.")
        return

    refresh_col, active_count_col = st.columns(2)
    if refresh_col.button("Refresh prices", use_container_width=True):
        get_latest_quote.clear()
        get_latest_trade_price.clear()
        st.rerun()

    try:
        active_trades = load_cloud_trades(
            supabase,
            st.session_state.supabase_user_id,
        )
    except Exception as error:
        active_trades = []
        st.error(f"Cloud trades could not be loaded: {error}")

    active_count_col.metric("Active trades", len(active_trades))

    if not active_trades:
        st.info("No active trades are being organized right now.")
    else:
        for trade in active_trades:
            entry = float(trade["entry_price"])
            stop = float(trade["stop_price"])
            target = float(trade["target_price"])
            quantity = int(trade.get("quantity") or 1)
            try:
                current = get_latest_trade_price(trade["ticker"])
            except Exception:
                current = None

            with st.container(border=True):
                title_col, side_col = st.columns([3, 1])
                paper_label = " · PAPER" if "PAPER" in str(trade.get("notes") or "").upper() else ""
                title_col.markdown(f"### {trade['ticker']}{paper_label}")
                side_col.markdown(f"**{trade['direction']} · {quantity} share(s)**")

                if current is None:
                    st.warning("Current price could not be loaded.")
                    default_exit = entry
                else:
                    metrics = calculate_live_trade_metrics(trade, current)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current", f"${current:.2f}")
                    m2.metric("Unrealized P/L", f"${metrics['total_pnl']:+.2f}", f"{metrics['pnl_percent']:+.2%}")
                    m3.metric("Status", metrics["status"])
                    st.write(f"**Entry:** ${entry:.2f}  •  **Stop:** ${stop:.2f}  •  **Target:** ${target:.2f}")
                    st.caption(f"${metrics['stop_distance']:.2f} from stop · ${metrics['target_distance']:.2f} from target")
                    default_exit = current

                with st.expander("Close this trade"):
                    with st.form(f"clean_close_trade_{trade['id']}"):
                        exit_price = st.number_input("Exit price", min_value=0.01, value=float(default_exit), step=0.01, format="%.2f")
                        notes = st.text_area("Notes (optional)")
                        close_clicked = st.form_submit_button("Close and record trade", use_container_width=True)
                    if close_clicked:
                        try:
                            close_cloud_trade(supabase, trade["id"], float(exit_price), notes.strip())
                            st.rerun()
                        except Exception as error:
                            st.error(f"Trade close failed: {error}")

    with st.expander("Add a trade you already entered"):
        st.caption(
            "This is optional. Use it after a real or paper position has actually filled."
        )
        with st.form("clean_add_trade", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                ticker = st.text_input("Ticker").strip().upper()
                direction = st.selectbox("Direction", ["LONG", "SHORT"])
                quantity = st.number_input("Shares", min_value=1, value=1, step=1)
                paper_trade = st.checkbox("Paper trade", value=True, key="manual_paper_trade")
            with c2:
                entry = st.number_input("Actual fill price", min_value=0.0, step=0.01, format="%.2f")
                stop = st.number_input("Stop loss", min_value=0.0, step=0.01, format="%.2f")
                target = st.number_input("Target price", min_value=0.0, step=0.01, format="%.2f")
            already_filled = st.checkbox("I confirm this trade has already filled")
            clicked = st.form_submit_button("Add entered trade", use_container_width=True)

        if clicked:
            valid = ticker and entry > 0 and stop > 0 and target > 0
            valid = valid and (stop < entry < target if direction == "LONG" else target < entry < stop)
            if not valid:
                st.error("Check the ticker and make sure the stop and target are on the correct sides of entry.")
            elif not already_filled:
                st.error("Confirm that the trade has actually filled before adding it.")
            else:
                try:
                    add_cloud_trade(
                        supabase,
                        st.session_state.supabase_user_id,
                        {
                            "ticker": ticker,
                            "direction": direction,
                            "quantity": int(quantity),
                            "entry": float(entry),
                            "stop": float(stop),
                            "target": float(target),
                            "notes": "PAPER TRADE | Manually entered" if paper_trade else "Manually entered",
                        },
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Trade could not be saved: {error}")

    try:
        legacy_pending = load_pending_orders(supabase, st.session_state.supabase_user_id)
    except Exception:
        legacy_pending = []

    if legacy_pending:
        with st.expander(f"Legacy pending orders from the old workflow ({len(legacy_pending)})"):
            st.caption(
                "The app no longer queues orders. Cancel these old pending records, then add a trade only if you actually enter it."
            )
            for order in legacy_pending:
                c1, c2 = st.columns([3, 1])
                c1.write(
                    f"**{order['ticker']} {order['direction']}** · old entry ${float(order['entry_price']):.2f}"
                )
                if c2.button("Cancel", key=f"cancel_legacy_{order['id']}", use_container_width=True):
                    try:
                        cancel_pending_order(supabase, order["id"])
                        st.rerun()
                    except Exception as error:
                        st.error(f"Legacy order could not be cancelled: {error}")

    try:
        closed = load_closed_trades(supabase, st.session_state.supabase_user_id, limit=50)
    except Exception as error:
        closed = []
        st.error(f"Trade history could not be loaded: {error}")

    with st.expander(f"Trade History ({len(closed)})"):
        rows = []
        for trade in closed:
            if trade.get("exit_price") is None:
                continue
            rows.append(
                {
                    "Ticker": trade["ticker"],
                    "Side": trade["direction"],
                    "Shares": int(trade.get("quantity") or 1),
                    "Entry": f"${float(trade['entry_price']):.2f}",
                    "Exit": f"${float(trade['exit_price']):.2f}",
                    "P/L": f"${calculate_closed_trade_result(trade):+.2f}",
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No closed trades yet.")


def run_clean_research(
    ticker,
    years,
    holding_days,
    minimum_quality,
    minimum_macro_score,
    profile_selection,
):
    symbol = ticker.strip().upper()
    start_timestamp = pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(years=years)
    download_start = start_timestamp - pd.DateOffset(days=220)

    stock_history = yf.Ticker(symbol).history(
        start=download_start.date().isoformat(),
        auto_adjust=True,
    )
    classification = resolve_ticker_macro_profile(symbol, profile_selection)

    initial_plan = build_ticker_factor_plan(classification, oil_threshold=0.02)
    requested_labels = tuple(sorted({factor["key"] for factor in initial_plan}))
    macro_history = load_macro_history(
        download_start.date().isoformat(),
        requested_labels,
    )
    macro_context = build_macro_context(macro_history, lookback_days=5)

    oil_moves = (
        macro_context["Oil Return"].dropna().abs()
        if "Oil Return" in macro_context
        else pd.Series(dtype=float)
    )
    if oil_moves.empty:
        oil_threshold = 0.02
    else:
        oil_threshold = float(oil_moves.quantile(0.75))
        oil_threshold = max(0.01, min(0.05, oil_threshold))

    factor_plan = build_ticker_factor_plan(classification, oil_threshold=oil_threshold)
    factor_table, current_macro_score = build_current_factor_table(macro_context, factor_plan)

    technical_trades, prepared = run_strategy_backtest(
        stock_history,
        start_timestamp.date(),
        holding_days,
        minimum_quality,
        7.5,
        stop_atr_multiple=1.25,
        reward_to_risk=2.0,
    )
    technical_stats = calculate_backtest_statistics(technical_trades, prepared)

    macro_trades, macro_prepared = run_macro_strategy_backtest(
        data=stock_history,
        macro_context=macro_context,
        test_start_date=start_timestamp.date(),
        holding_days=holding_days,
        minimum_quality=minimum_quality,
        cost_bps_per_side=7.5,
        factor_plan=factor_plan,
        minimum_macro_score=minimum_macro_score,
        stop_atr_multiple=1.25,
        reward_to_risk=2.0,
    )
    macro_stats = calculate_backtest_statistics(macro_trades, macro_prepared)

    oil_is_relevant = any(
        factor["key"] == "Oil" and factor["weight"] > 0
        for factor in factor_plan
    )
    oil_trades = pd.DataFrame()
    oil_stats = None
    baseline_stats = None

    if oil_is_relevant and "Oil Return" in macro_context:
        study_mode = (
            "Long stock after oil drop"
            if classification["profile"] == "Airline"
            else "Long stock after oil spike"
        )
        oil_trades, oil_prepared = run_oil_shock_study(
            data=stock_history,
            macro_context=macro_context,
            test_start_date=start_timestamp.date(),
            holding_days=holding_days,
            cost_bps_per_side=5.0,
            oil_threshold=oil_threshold,
            study_mode=study_mode,
        )
        oil_stats = calculate_backtest_statistics(oil_trades, oil_prepared)
        baseline_trades, baseline_prepared = run_unconditional_long_study(
            stock_history,
            start_timestamp.date(),
            holding_days,
            5.0,
        )
        baseline_stats = calculate_backtest_statistics(baseline_trades, baseline_prepared)

    return {
        "ticker": symbol,
        "years": years,
        "holding_days": holding_days,
        "classification": classification,
        "factor_plan": factor_plan,
        "factor_table": factor_table,
        "current_macro_score": current_macro_score,
        "oil_threshold": oil_threshold,
        "macro_context": macro_context,
        "technical_trades": technical_trades,
        "technical_stats": technical_stats,
        "macro_trades": macro_trades,
        "macro_stats": macro_stats,
        "oil_trades": oil_trades,
        "oil_stats": oil_stats,
        "baseline_stats": baseline_stats,
    }


def render_research():
    st.header("Research Lab")
    st.caption(
        "Compare a technical strategy with a ticker-aware macro model. The app "
        "emphasizes factors that make economic sense for the company and marks "
        "unrelated factors as observed but not used."
    )

    if "research_ticker_input" not in st.session_state:
        st.session_state.research_ticker_input = "AAL"

    with st.form("clean_research_form"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input(
                "Ticker",
                key="research_ticker_input",
            ).strip().upper()
            years = st.selectbox("History", [2, 5, 10], index=1)
        with c2:
            holding_days = st.selectbox("Holding period", [1, 2, 3, 5, 10], index=2)
            minimum_quality = st.slider("Minimum setup quality", 40, 90, 55, 5)

        with st.expander("Advanced research controls"):
            minimum_macro_score = st.slider(
                "Minimum macro confirmation score",
                1, 6, 2,
                help="Higher values require more relevant macro factors to agree.",
            )
            profile = st.selectbox(
                "Relationship profile",
                [
                    "Auto by ticker", "Airline", "Semiconductor", "Bank",
                    "Energy", "Gold miner", "Industrial", "Growth / technology",
                    "General market",
                ],
            )

        run_clicked = st.form_submit_button(
            "Run Research",
            type="primary",
            use_container_width=True,
        )

    if run_clicked:
        try:
            with st.spinner("Detecting the company profile and running ticker-aware tests…"):
                st.session_state.research_result = run_clean_research(
                    ticker=ticker,
                    years=years,
                    holding_days=holding_days,
                    minimum_quality=minimum_quality,
                    minimum_macro_score=minimum_macro_score,
                    profile_selection=profile,
                )
        except Exception as error:
            st.session_state.research_result = None
            st.error(f"Research could not be completed: {error}")

    result = st.session_state.research_result
    if not result:
        st.info("Run Research to compare the technical strategy with relevant macro confirmation.")
        return

    technical = result["technical_stats"]
    macro = result["macro_stats"]
    technical_count = technical["total_trades"] if technical else 0
    macro_count = macro["total_trades"] if macro else 0

    if macro and technical:
        macro_average_edge = macro["average_return"] - technical["average_return"]
        macro_win_edge = macro["win_rate"] - technical["win_rate"]
        if macro_average_edge >= 0.002 and macro_win_edge >= 0.03:
            macro_verdict = "Ticker-aware macro confirmation helped"
            macro_kind = "success"
        elif macro_average_edge <= -0.002 and macro_win_edge <= -0.03:
            macro_verdict = "Ticker-aware macro confirmation hurt"
            macro_kind = "error"
        else:
            macro_verdict = "No clear macro improvement"
            macro_kind = "info"
    else:
        macro_average_edge = None
        macro_verdict = "Not enough data"
        macro_kind = "warning"

    macro_confidence = "Low" if macro_count < 10 else "Moderate" if macro_count < 30 else "Higher"
    classification = result["classification"]
    factor_table = result["factor_table"]

    st.subheader(f"Ticker-aware macro verdict for {result['ticker']}")
    st.caption(
        f"Detected profile: **{classification['profile']}** • "
        f"Sector: {classification.get('sector', 'Unknown')} • "
        f"Industry: {classification.get('industry', 'Unknown')}"
    )

    with st.container(border=True):
        verdict_icon = {"success": "✅", "error": "⚠️", "warning": "⚠️", "info": "ℹ️"}.get(macro_kind, "ℹ️")
        st.markdown(f"### {verdict_icon} {macro_verdict}")
        st.write(
            f"Confidence: {macro_confidence}. Technical-only sample: {technical_count} trades. "
            f"Macro-confirmed sample: {macro_count} trades."
        )

        relevant = factor_table[factor_table["Weight"] != "Ignored"] if not factor_table.empty else factor_table
        supportive = relevant[relevant["Current effect"] == "Supportive"]["Factor"].tolist()
        negative = relevant[relevant["Current effect"] == "Negative"]["Factor"].tolist()
        if supportive:
            st.write("**Current support:** " + ", ".join(supportive[:3]))
        if negative:
            st.write("**Current risks:** " + ", ".join(negative[:3]))

        m1, m2, m3 = st.columns(3)
        m1.metric("Macro-confirmed trades", macro_count)
        m2.metric("Macro win rate", f"{macro['win_rate']:.1%}" if macro else "—")
        m3.metric(
            "Average-return difference",
            f"{macro_average_edge:+.2%}" if macro_average_edge is not None else "—",
        )

        if macro_kind == "info":
            st.caption(
                "The relevant macro filter did not clearly improve both win rate and average return. "
                "That means the historical evidence is inconclusive, not automatically bearish."
            )

    st.markdown("### Factors used for this ticker")
    if factor_table.empty:
        st.info("Macro factor data was unavailable.")
    else:
        main_table = factor_table[factor_table["Weight"] != "Ignored"]
        st.dataframe(
            main_table[["Factor", "Relevance", "Current effect", "Reading", "Weight"]],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Full macro context, including ignored factors"):
        if factor_table.empty:
            st.info("Macro factor data was unavailable.")
        else:
            st.dataframe(factor_table, hide_index=True, use_container_width=True)
            st.caption(
                "Ignored factors remain visible for context, but they do not affect the headline score. "
                "For example, oil is observed for NVDA but receives no weight."
            )

    with st.expander("Strategy comparison"):
        rows = [
            build_strategy_comparison_row("Technical only", technical),
            build_strategy_comparison_row("Technical + ticker-aware macro", macro),
        ]
        if result["oil_stats"] is not None:
            rows.append(build_strategy_comparison_row("Oil relationship study", result["oil_stats"]))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Backtests enter at the next session open, use ATR-based stops and 2:1 targets, "
            "assume the stop was hit first when one daily candle touches both levels, include "
            "7.5 basis points per side, reserve the final 30% as an out-of-sample check, and test "
            "three chronological stability periods. Buy-and-hold is always invested, while strategy exposure is shown separately."
        )

    if result["oil_stats"] is not None:
        oil_stats = result["oil_stats"]
        baseline_stats = result["baseline_stats"]
        with st.expander("Ticker-specific oil relationship"):
            oil_edge = (
                oil_stats["average_return"] - baseline_stats["average_return"]
                if oil_stats and baseline_stats else None
            )
            st.write(
                f"Oil is highly relevant to the detected **{classification['profile']}** profile, "
                "so it receives a dedicated historical study."
            )
            o1, o2, o3 = st.columns(3)
            o1.metric("Matching trades", oil_stats["total_trades"] if oil_stats else 0)
            o2.metric("Win rate", f"{oil_stats['win_rate']:.1%}" if oil_stats else "—")
            o3.metric("Average-return edge", f"{oil_edge:+.2%}" if oil_edge is not None else "—")
            st.caption(
                f"Automatic meaningful-oil threshold: {result['oil_threshold']:.1%} over five trading days."
            )

    with st.expander("Equity curves"):
        for label, stats in [
            ("Technical only", technical),
            ("Technical + ticker-aware macro", macro),
            ("Oil relationship study", result["oil_stats"]),
        ]:
            if stats is None:
                continue
            st.markdown(f"**{label}**")
            chart = go.Figure()
            for column in stats["equity_curve"].columns:
                chart.add_trace(
                    go.Scatter(
                        x=stats["equity_curve"].index,
                        y=stats["equity_curve"][column],
                        mode="lines",
                        name=column,
                    )
                )
            chart.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(chart, use_container_width=True)

    with st.expander("Recent macro-confirmed trades"):
        if result["macro_trades"].empty:
            st.info("No historical trades met both the technical and ticker-aware macro filters.")
        else:
            display = result["macro_trades"].tail(25).sort_values("Signal Date", ascending=False).copy()
            for column in ["Signal Date", "Entry Date", "Exit Date"]:
                display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d")
            display["Net Return"] = display["Net Return"].map(lambda value: f"{value:+.2%}")
            st.dataframe(
                display[["Signal Date", "Direction", "Setup Quality", "Macro Score", "Macro Evidence", "Net Return"]],
                hide_index=True,
                use_container_width=True,
            )

    st.caption("Backtests describe historical relationships, not a guarantee that the next trade will work.")


render_account_sidebar()

st.radio(
    "Main navigation",
    ["Trade Finder", "Analyze", "Active Trades", "Research"],
    key="nav_page",
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

if st.session_state.nav_page == "Trade Finder":
    render_trade_finder()
elif st.session_state.nav_page == "Analyze":
    render_analyze()
elif st.session_state.nav_page == "Active Trades":
    render_active_trades()
else:
    render_research()

st.divider()
st.caption("Yahoo Finance prices may be delayed. Finder results are decision support, not guarantees. Active Trades only organizes positions you choose to enter.")
