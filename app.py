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
    "A beginner-friendly decision dashboard for finding, "
    "testing, entering, and tracking higher-quality trades."
)


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
            "target_price,quantity,status,created_at"
        )
        .eq("user_id", user_id)
        .eq("status", "ACTIVE")
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
    response = (
        client.table("trades")
        .insert(
            {
                "user_id": user_id,
                "ticker": trade["ticker"],
                "direction": trade["direction"],
                "entry_price": trade["entry"],
                "stop_price": trade["stop"],
                "target_price": trade["target"],
                "quantity": trade["quantity"],
                "status": "ACTIVE",
            }
        )
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


@st.cache_data(ttl=60, show_spinner=False)
def get_latest_trade_price(ticker):
    """Return Yahoo's latest available price, cached for 60 seconds."""
    symbol = ticker.strip().upper()
    stock = yf.Ticker(symbol)

    try:
        fast_info = stock.fast_info
        latest = fast_info.get("last_price")
        if latest is not None and float(latest) > 0:
            return float(latest)
    except Exception:
        pass

    intraday = stock.history(
        period="5d",
        interval="5m",
        prepost=True,
        auto_adjust=False,
    )

    intraday = intraday.dropna(subset=["Close"])

    if intraday.empty:
        return None

    return float(intraday["Close"].iloc[-1])


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


supabase, supabase_error = get_supabase_client()
logged_in = bool(st.session_state.get("supabase_user_id"))

with st.sidebar:
    st.header("👤 Account")

    if supabase_error:
        st.error(supabase_error)
        st.caption(
            "Stock analysis still works, but cloud trade saving is disabled."
        )

    elif not logged_in:
        sign_in_tab, create_account_tab = st.tabs(
            ["Sign in", "Create account"]
        )

        with sign_in_tab:
            with st.form("sign_in_form"):
                sign_in_email = st.text_input(
                    "Email",
                    key="sign_in_email",
                ).strip()

                sign_in_password = st.text_input(
                    "Password",
                    type="password",
                    key="sign_in_password",
                )

                sign_in_clicked = st.form_submit_button(
                    "Sign in",
                    use_container_width=True,
                )

            if sign_in_clicked:
                if not sign_in_email or not sign_in_password:
                    st.error("Enter your email and password.")
                else:
                    try:
                        auth_response = (
                            supabase.auth.sign_in_with_password(
                                {
                                    "email": sign_in_email,
                                    "password": sign_in_password,
                                }
                            )
                        )
                        remember_auth_response(auth_response)
                        st.rerun()
                    except Exception as error:
                        st.error(f"Sign-in failed: {error}")

        with create_account_tab:
            with st.form("create_account_form"):
                create_email = st.text_input(
                    "Email",
                    key="create_email",
                ).strip()

                create_password = st.text_input(
                    "Password",
                    type="password",
                    key="create_password",
                )

                create_password_again = st.text_input(
                    "Confirm password",
                    type="password",
                    key="create_password_again",
                )

                create_clicked = st.form_submit_button(
                    "Create account",
                    use_container_width=True,
                )

            if create_clicked:
                if not create_email or not create_password:
                    st.error("Enter an email and password.")
                elif create_password != create_password_again:
                    st.error("The passwords do not match.")
                elif len(create_password) < 8:
                    st.error("Use a password with at least 8 characters.")
                else:
                    try:
                        auth_response = supabase.auth.sign_up(
                            {
                                "email": create_email,
                                "password": create_password,
                            }
                        )

                        remember_auth_response(auth_response)

                        if auth_response.session:
                            st.rerun()
                        else:
                            st.success(
                                "Account created. Check your email to "
                                "confirm it, then return here and sign in."
                            )
                    except Exception as error:
                        st.error(f"Account creation failed: {error}")

        st.caption(
            "Sign in once on each device. Encrypted browser cookies "
            "restore your login after a refresh."
        )

    else:
        user_email = st.session_state.get(
            "supabase_user_email",
            "Signed-in user",
        )

        st.success(f"Signed in as {user_email}")

        if st.button(
            "Sign out",
            use_container_width=True,
        ):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass

            clear_auth_state()
            st.rerun()

    st.divider()
    st.header("📌 Active Trades")
    st.caption("Prices are Yahoo Finance estimates and may be delayed.")

    if not logged_in:
        st.info("Sign in to save and sync active trades.")

    else:
        refresh_column, count_column = st.columns([1, 1])

        if refresh_column.button(
            "Refresh prices",
            use_container_width=True,
        ):
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

        count_column.metric("Open", len(active_trades))

        if active_trades:
            for active_trade in active_trades:
                entry_price = float(active_trade["entry_price"])
                stop_price = float(active_trade["stop_price"])
                target_price = float(active_trade["target_price"])
                quantity = int(active_trade.get("quantity") or 1)
                direction = active_trade["direction"]

                try:
                    current_price = get_latest_trade_price(
                        active_trade["ticker"]
                    )
                except Exception:
                    current_price = None

                with st.container(border=True):
                    st.markdown(
                        f"### {active_trade['ticker']} — {direction}"
                    )
                    st.caption(f"{quantity} share(s)")

                    if current_price is None:
                        st.warning("Current price could not be loaded.")
                        default_exit_price = entry_price
                    else:
                        metrics = calculate_live_trade_metrics(
                            active_trade,
                            current_price,
                        )

                        price_column, pnl_column = st.columns(2)

                        price_column.metric(
                            "Current",
                            f"${current_price:,.2f}",
                        )

                        pnl_column.metric(
                            "Unrealized P/L",
                            f"${metrics['total_pnl']:+,.2f}",
                            f"{metrics['pnl_percent']:+.2%}",
                        )

                        status_message = (
                            f"{metrics['status']} — "
                            f"${metrics['stop_distance']:.2f} from stop, "
                            f"${metrics['target_distance']:.2f} from target."
                        )

                        if metrics["status_kind"] == "success":
                            st.success(status_message)
                        elif metrics["status_kind"] == "error":
                            st.error(status_message)
                        elif metrics["status_kind"] == "warning":
                            st.warning(status_message)
                        else:
                            st.info(status_message)

                        default_exit_price = current_price

                    st.write(
                        f"**Entry:** ${entry_price:.2f}  •  "
                        f"**Stop:** ${stop_price:.2f}  •  "
                        f"**Target:** ${target_price:.2f}"
                    )

                    with st.expander("Close this trade"):
                        with st.form(
                            f"close_trade_form_{active_trade['id']}"
                        ):
                            exit_price = st.number_input(
                                "Exit price",
                                min_value=0.01,
                                value=float(default_exit_price),
                                step=0.01,
                                format="%.2f",
                                key=f"exit_price_{active_trade['id']}",
                            )

                            close_notes = st.text_area(
                                "Notes (optional)",
                                key=f"close_notes_{active_trade['id']}",
                                placeholder=(
                                    "Why did you exit? What did you learn?"
                                ),
                            )

                            close_clicked = st.form_submit_button(
                                "Close and record trade",
                                use_container_width=True,
                            )

                        if close_clicked:
                            try:
                                close_cloud_trade(
                                    supabase,
                                    active_trade["id"],
                                    float(exit_price),
                                    close_notes.strip(),
                                )
                                get_latest_trade_price.clear()
                                st.rerun()
                            except Exception as error:
                                st.error(f"Trade close failed: {error}")
        else:
            st.info("No active trades yet.")

        st.divider()
        st.subheader("Add a trade")

        with st.form("add_trade_form", clear_on_submit=True):
            trade_ticker = st.text_input(
                "Ticker"
            ).strip().upper()

            trade_direction = st.selectbox(
                "Direction",
                ["LONG", "SHORT"],
            )

            trade_quantity = st.number_input(
                "Shares",
                min_value=1,
                value=1,
                step=1,
            )

            trade_entry = st.number_input(
                "Entry price",
                min_value=0.0,
                step=0.01,
                format="%.2f",
            )

            trade_stop = st.number_input(
                "Stop loss",
                min_value=0.0,
                step=0.01,
                format="%.2f",
            )

            trade_target = st.number_input(
                "Target price",
                min_value=0.0,
                step=0.01,
                format="%.2f",
            )

            add_trade_clicked = st.form_submit_button(
                "Add active trade",
                use_container_width=True,
            )

        if add_trade_clicked:
            missing_required_value = (
                not trade_ticker
                or trade_entry <= 0
                or trade_stop <= 0
                or trade_target <= 0
            )

            long_prices_invalid = (
                trade_direction == "LONG"
                and not (trade_stop < trade_entry < trade_target)
            )

            short_prices_invalid = (
                trade_direction == "SHORT"
                and not (trade_target < trade_entry < trade_stop)
            )

            if missing_required_value:
                st.error("Enter a ticker, entry, stop, and target.")
            elif long_prices_invalid:
                st.error(
                    "For a long trade, the stop must be below "
                    "the entry and the target must be above it."
                )
            elif short_prices_invalid:
                st.error(
                    "For a short trade, the target must be below "
                    "the entry and the stop must be above it."
                )
            else:
                try:
                    add_cloud_trade(
                        supabase,
                        st.session_state.supabase_user_id,
                        {
                            "ticker": trade_ticker,
                            "direction": trade_direction,
                            "quantity": int(trade_quantity),
                            "entry": float(trade_entry),
                            "stop": float(trade_stop),
                            "target": float(trade_target),
                        },
                    )
                    get_latest_trade_price.clear()
                    st.rerun()
                except Exception as error:
                    st.error(f"Trade could not be saved: {error}")

        try:
            closed_trades = load_closed_trades(
                supabase,
                st.session_state.supabase_user_id,
            )
        except Exception as error:
            closed_trades = []
            st.error(f"Trade history could not be loaded: {error}")

        if closed_trades:
            with st.expander(
                f"📚 Trade History ({len(closed_trades)})"
            ):
                realized_total = sum(
                    calculate_closed_trade_result(trade)
                    for trade in closed_trades
                    if trade.get("exit_price") is not None
                )

                st.metric(
                    "Realized P/L shown",
                    f"${realized_total:+,.2f}",
                )

                history_rows = []

                for trade in closed_trades:
                    if trade.get("exit_price") is None:
                        continue

                    realized_pnl = calculate_closed_trade_result(trade)

                    history_rows.append(
                        {
                            "Ticker": trade["ticker"],
                            "Side": trade["direction"],
                            "Shares": int(trade.get("quantity") or 1),
                            "Entry": f"${float(trade['entry_price']):.2f}",
                            "Exit": f"${float(trade['exit_price']):.2f}",
                            "P/L": f"${realized_pnl:+.2f}",
                        }
                    )

                if history_rows:
                    st.dataframe(
                        pd.DataFrame(history_rows),
                        hide_index=True,
                        use_container_width=True,
                    )


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
    """Calculate the same indicators used by the current trade setup."""
    result = data.copy()

    result["MA20"] = result["Close"].rolling(20).mean()
    result["MA50"] = result["Close"].rolling(50).mean()

    standard_deviation = result["Close"].rolling(20).std()

    result["Upper Band"] = (
        result["MA20"] + (2 * standard_deviation)
    )

    result["Lower Band"] = (
        result["MA20"] - (2 * standard_deviation)
    )

    movement = result["Close"].diff()

    average_gain = (
        movement.clip(lower=0)
        .rolling(14)
        .mean()
    )

    average_loss = (
        -movement.clip(upper=0)
        .rolling(14)
        .mean()
    )

    relative_strength = average_gain / average_loss

    result["RSI"] = 100 - (
        100 / (1 + relative_strength)
    )

    result.loc[
        (average_gain == 0) & (average_loss == 0),
        "RSI",
    ] = 50.0

    result.loc[
        (average_gain > 0) & (average_loss == 0),
        "RSI",
    ] = 100.0

    result.loc[
        (average_gain == 0) & (average_loss > 0),
        "RSI",
    ] = 0.0

    ema_12 = result["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = result["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    result["MACD"] = ema_12 - ema_26

    result["Signal"] = result["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    result["Histogram"] = (
        result["MACD"] - result["Signal"]
    )

    result["Average Volume 20"] = (
        result["Volume"].rolling(20).mean()
    )

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


def run_strategy_backtest(
    data,
    test_start_date,
    holding_days,
    minimum_quality,
    cost_bps_per_side,
):
    """
    Backtest the dashboard's long/short setup without look-ahead bias.

    A signal is calculated after a daily close. A qualifying position
    enters at the next session's open and exits at the close after the
    selected number of trading sessions. Only one position is open at
    a time.
    """
    prepared = remove_unfinished_daily_bar(data)

    prepared = prepared.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    prepared = add_backtest_indicators(prepared)

    required_columns = [
        "Open",
        "Close",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "Signal",
        "Histogram",
        "Upper Band",
        "Lower Band",
        "Average Volume 20",
    ]

    prepared = prepared.dropna(
        subset=required_columns
    )

    trades = []
    index_position = 1

    while index_position < len(prepared) - holding_days:
        signal_row = prepared.iloc[index_position]
        previous_row = prepared.iloc[index_position - 1]

        signal_date = pd.Timestamp(
            prepared.index[index_position]
        )

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
            previous_histogram=float(
                previous_row["Histogram"]
            ),
            upper_band=float(signal_row["Upper Band"]),
            lower_band=float(signal_row["Lower Band"]),
            latest_volume=int(signal_row["Volume"]),
            average_volume_20=float(
                signal_row["Average Volume 20"]
            ),
        )

        qualifying_direction = setup["bias"] in (
            "LONG BIAS",
            "SHORT BIAS",
        )

        qualifying_quality = (
            setup["setup_quality"] >= minimum_quality
        )

        if not (
            qualifying_direction
            and qualifying_quality
        ):
            index_position += 1
            continue

        entry_position = index_position + 1
        exit_position = (
            entry_position + holding_days - 1
        )

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

        direction = (
            "LONG"
            if setup["bias"] == "LONG BIAS"
            else "SHORT"
        )

        if direction == "LONG":
            gross_return = (
                exit_price / entry_price
            ) - 1
        else:
            gross_return = (
                entry_price / exit_price
            ) - 1

        round_trip_cost = (
            2 * cost_bps_per_side / 10000
        )

        net_return = gross_return - round_trip_cost

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
                "Setup Quality": int(
                    setup["setup_quality"]
                ),
                "Direction Score": int(
                    setup["direction_score"]
                ),
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Gross Return": gross_return,
                "Net Return": net_return,
                "Winner": net_return > 0,
            }
        )

        # Do not allow overlapping positions.
        index_position = exit_position + 1

    trades_frame = pd.DataFrame(trades)

    return trades_frame, prepared


def calculate_backtest_statistics(
    trades,
    prepared_history,
):
    """Calculate summary statistics and a comparison equity curve."""
    if trades.empty:
        return None

    returns = trades["Net Return"].astype(float)

    strategy_growth = (1 + returns).cumprod()

    running_peak = strategy_growth.cummax()

    drawdown = (
        strategy_growth / running_peak
    ) - 1

    winning_returns = returns[returns > 0]
    losing_returns = returns[returns <= 0]

    if losing_returns.empty:
        profit_factor = float("inf")
    else:
        profit_factor = (
            winning_returns.sum()
            / abs(losing_returns.sum())
        )

    first_entry_date = pd.Timestamp(
        trades["Entry Date"].iloc[0]
    )

    last_exit_date = pd.Timestamp(
        trades["Exit Date"].iloc[-1]
    )

    benchmark_prices = prepared_history.loc[
        (
            prepared_history.index
            >= first_entry_date
        )
        & (
            prepared_history.index
            <= last_exit_date
        ),
        "Close",
    ]

    if benchmark_prices.empty:
        buy_hold_return = 0.0
    else:
        buy_hold_return = (
            float(benchmark_prices.iloc[-1])
            / float(benchmark_prices.iloc[0])
        ) - 1

    equity_curve = pd.DataFrame(
        {
            "Strategy": 10000 * strategy_growth.values,
        },
        index=pd.to_datetime(
            trades["Exit Date"]
        ),
    )

    if not benchmark_prices.empty:
        benchmark_at_exits = (
            benchmark_prices
            .reindex(
                equity_curve.index,
                method="ffill",
            )
        )

        equity_curve["Buy and Hold"] = (
            10000
            * benchmark_at_exits
            / float(benchmark_prices.iloc[0])
        )

    statistics = {
        "total_trades": len(trades),
        "win_rate": float(trades["Winner"].mean()),
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "total_return": float(
            strategy_growth.iloc[-1] - 1
        ),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(profit_factor),
        "buy_hold_return": float(buy_hold_return),
        "equity_curve": equity_curve,
    }

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

with st.container(border=True):
    st.subheader("Analyze a stock")
    st.caption(
        "Enter a ticker, then open optional settings only when "
        "you want to change the watchlist or backtest."
    )

    with st.form("analysis_form"):
        ticker_column, range_column = st.columns([2, 1])

        with ticker_column:
            ticker = st.text_input(
                "Ticker",
                "AAL",
                help="Examples: AAL, INTC, CVX, VTI",
            ).strip().upper()

        with range_column:
            selected_period = st.selectbox(
                "Chart range",
                list(period_choices.keys()),
            )

        with st.expander("Optional watchlist and backtest settings"):
            watchlist_text = st.text_input(
                "Watchlist tickers",
                "VTI, VXUS, AAL, CVX, INTC",
                help="Separate ticker symbols with commas.",
            )

            st.markdown("#### Backtesting")

            backtest_column_1, backtest_column_2 = st.columns(2)

            with backtest_column_1:
                backtest_lookback_label = st.selectbox(
                    "Historical test period",
                    list(backtest_lookback_choices.keys()),
                    index=1,
                )

                backtest_holding_days = st.selectbox(
                    "Hold each trade for",
                    [1, 3, 5, 10, 20],
                    index=2,
                    format_func=lambda value: (
                        f"{value} trading day"
                        if value == 1
                        else f"{value} trading days"
                    ),
                )

            with backtest_column_2:
                backtest_minimum_quality = st.slider(
                    "Minimum setup quality",
                    min_value=50,
                    max_value=90,
                    value=70,
                    step=5,
                )

                backtest_cost_bps = st.number_input(
                    "Estimated cost per side (basis points)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                )

        analyze = st.form_submit_button(
            "Analyze stock",
            type="primary",
            use_container_width=True,
        )


if analyze:
    if not ticker:
        st.error("Please enter a stock ticker.")
        st.stop()

    try:
        with st.spinner(f"Loading {ticker}..."):
            stock = yf.Ticker(ticker)

            history = stock.history(
                period=period_choices[selected_period],
                auto_adjust=False,
            )

            backtest_years = (
                backtest_lookback_choices[
                    backtest_lookback_label
                ]
            )

            backtest_start_timestamp = (
                pd.Timestamp.now(
                    tz="America/New_York"
                )
                - pd.DateOffset(
                    years=backtest_years
                )
            )

            backtest_download_start = (
                backtest_start_timestamp
                - pd.DateOffset(days=180)
            )

            backtest_history = stock.history(
                start=backtest_download_start.date().isoformat(),
                auto_adjust=True,
            )

    except Exception as error:
        st.error(f"Unable to download stock data: {error}")
        st.stop()

    history = history.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    if history.empty or len(history) < 50:
        st.error(
            "Not enough stock data was found. "
            "Check the ticker symbol."
        )
        st.stop()

    # Moving averages and Bollinger Bands
    history["MA20"] = history["Close"].rolling(20).mean()
    history["MA50"] = history["Close"].rolling(50).mean()

    standard_deviation = history["Close"].rolling(20).std()

    history["Upper Band"] = (
        history["MA20"] + (2 * standard_deviation)
    )

    history["Lower Band"] = (
        history["MA20"] - (2 * standard_deviation)
    )

    # RSI
    movement = history["Close"].diff()

    gains = movement.clip(lower=0).rolling(14).mean()
    losses = -movement.clip(upper=0).rolling(14).mean()

    relative_strength = gains / losses

    history["RSI"] = 100 - (
        100 / (1 + relative_strength)
    )

    # MACD
    ema_12 = history["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = history["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    history["MACD"] = ema_12 - ema_26

    history["Signal"] = history["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    history["Histogram"] = (
        history["MACD"] - history["Signal"]
    )

    latest = history.iloc[-1]
    previous = history.iloc[-2]

    price = float(latest["Close"])
    previous_price = float(previous["Close"])

    change = price - previous_price
    change_percent = (change / previous_price) * 100

    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])

    rsi = float(latest["RSI"])
    if math.isnan(rsi):
        rsi = 50.0

    macd = float(latest["MACD"])
    signal = float(latest["Signal"])
    histogram = float(latest["Histogram"])
    previous_histogram = float(
        history["Histogram"].iloc[-2]
    )

    upper_band = float(latest["Upper Band"])
    lower_band = float(latest["Lower Band"])

    latest_volume = int(latest["Volume"])

    average_volume_20 = float(
        history["Volume"].rolling(20).mean().iloc[-1]
    )

    # Trend analysis
    if price > ma20 and price > ma50:
        trend = "Uptrend"
        trend_text = "Price is above both moving averages."
        trend_box = st.success

    elif price < ma20 and price < ma50:
        trend = "Downtrend"
        trend_text = "Price is below both moving averages."
        trend_box = st.error

    else:
        trend = "Mixed trend"
        trend_text = "Price is between the moving averages."
        trend_box = st.warning

    # RSI analysis
    if rsi >= 70:
        rsi_status = "Potentially overbought"
        rsi_text = (
            "RSI is above 70. Recent upward movement "
            "may be stretched."
        )

    elif rsi <= 30:
        rsi_status = "Potentially oversold"
        rsi_text = (
            "RSI is below 30. Recent downward movement "
            "may be stretched."
        )

    else:
        rsi_status = "Neutral"
        rsi_text = "RSI is between 30 and 70."

    # MACD analysis
    if macd > signal:
        macd_status = "Bullish momentum"
        macd_text = "MACD is above its signal line."
        macd_box = st.success

    else:
        macd_status = "Bearish momentum"
        macd_text = "MACD is below its signal line."
        macd_box = st.error

    # Bollinger Band analysis
    if price > upper_band:
        band_status = "Above the upper band"

    elif price < lower_band:
        band_status = "Below the lower band"

    else:
        band_status = "Inside the bands"

    trade_setup = calculate_trade_setup(
        price=price,
        previous_price=previous_price,
        ma20=ma20,
        ma50=ma50,
        rsi=rsi,
        macd=macd,
        signal=signal,
        histogram=histogram,
        previous_histogram=previous_histogram,
        upper_band=upper_band,
        lower_band=lower_band,
        latest_volume=latest_volume,
        average_volume_20=average_volume_20,
    )

    try:
        backtest_trades, prepared_backtest_history = (
            run_strategy_backtest(
                data=backtest_history,
                test_start_date=(
                    backtest_start_timestamp.date()
                ),
                holding_days=backtest_holding_days,
                minimum_quality=(
                    backtest_minimum_quality
                ),
                cost_bps_per_side=backtest_cost_bps,
            )
        )

        backtest_statistics = (
            calculate_backtest_statistics(
                backtest_trades,
                prepared_backtest_history,
            )
        )

    except Exception as error:
        backtest_trades = pd.DataFrame()
        prepared_backtest_history = pd.DataFrame()
        backtest_statistics = None
        backtest_error = str(error)

    (
        trade_tab,
        backtest_tab,
        summary_tab,
        price_tab,
        momentum_tab,
        watchlist_tab,
        news_tab,
    ) = st.tabs(
        [
            "Decision",
            "Backtest",
            "Snapshot",
            "Chart",
            "Momentum",
            "Watchlist",
            "News",
        ]
    )

    with trade_tab:
        st.subheader(f"Decision for {ticker}")
        st.caption(
            "Use this as a decision aid: trade only when the setup, "
            "entry, risk, and current news all make sense together."
        )

        bias_column, quality_column, status_column = st.columns(3)

        bias_column.metric(
            "Technical direction",
            trade_setup["bias"],
        )

        quality_column.metric(
            "Setup quality",
            f"{trade_setup['setup_quality']} / 100",
        )

        status_column.metric(
            "Trade opportunity",
            trade_setup["trade_status"],
        )

        direction_score = trade_setup["direction_score"]

        st.write(
            f"Directional score: **{direction_score:+d} out of 7**"
        )

        st.progress(
            trade_setup["setup_quality"] / 100
        )
        st.caption(
            "The bar measures technical signal agreement, not the "
            "chance that a trade will make money."
        )

        if trade_setup["trade_status"] == "POTENTIAL TRADE SETUP":
            if trade_setup["bias"] == "LONG BIAS":
                st.success(
                    "Possible long setup to investigate. "
                    + trade_setup["status_text"]
                )
            else:
                st.error(
                    "Possible short setup to investigate. "
                    + trade_setup["status_text"]
                )

        elif trade_setup["trade_status"] == "WATCH FOR CONFIRMATION":
            st.warning(trade_setup["status_text"])

        else:
            st.info(trade_setup["status_text"])

        with st.expander("Why the dashboard gave this reading"):
            for reason in trade_setup["evidence"]:
                st.write(f"• {reason}")

        with st.expander("Risk flags"):
            if trade_setup["risk_flags"]:
                for warning in trade_setup["risk_flags"]:
                    st.write(f"• {warning}")
            else:
                st.write(
                    "• No major indicator-stretch warnings were detected."
                )

        st.warning(
            "This score measures indicator agreement, not the probability "
            "of making money. It does not know your entry, stop loss, "
            "position size, breaking news, or personal risk tolerance."
        )

    with backtest_tab:
        st.subheader(
            f"{ticker} Backtesting Lab"
        )

        st.caption(
            "This test recalculates the dashboard's indicators "
            "using only information available at each historical "
            "close. It enters at the next trading day's open and "
            f"exits after {backtest_holding_days} trading "
            "session(s)."
        )

        settings_column_1, settings_column_2, settings_column_3 = (
            st.columns(3)
        )

        settings_column_1.metric(
            "Test period",
            backtest_lookback_label,
        )

        settings_column_2.metric(
            "Minimum quality",
            f"{backtest_minimum_quality} / 100",
        )

        settings_column_3.metric(
            "Round-trip cost assumption",
            f"{2 * backtest_cost_bps:.0f} basis points",
        )

        if backtest_statistics is None:
            if "backtest_error" in locals():
                st.error(
                    "The historical test could not be completed: "
                    + backtest_error
                )
            else:
                st.info(
                    "No completed historical trades met the "
                    "selected rules."
                )

        else:
            metric_1, metric_2, metric_3, metric_4, metric_5 = (
                st.columns(5)
            )

            metric_1.metric(
                "Completed trades",
                f"{backtest_statistics['total_trades']}",
            )

            metric_2.metric(
                "Historical win rate",
                f"{backtest_statistics['win_rate']:.1%}",
            )

            metric_3.metric(
                "Average trade",
                f"{backtest_statistics['average_return']:+.2%}",
            )

            metric_4.metric(
                "Compounded return",
                f"{backtest_statistics['total_return']:+.1%}",
            )

            metric_5.metric(
                "Maximum drawdown",
                f"{backtest_statistics['max_drawdown']:.1%}",
            )

            comparison_1, comparison_2, comparison_3 = (
                st.columns(3)
            )

            if math.isinf(
                backtest_statistics["profit_factor"]
            ):
                profit_factor_text = "No losing trades"
            else:
                profit_factor_text = (
                    f"{backtest_statistics['profit_factor']:.2f}"
                )

            comparison_1.metric(
                "Profit factor",
                profit_factor_text,
            )

            comparison_2.metric(
                "Median trade",
                f"{backtest_statistics['median_return']:+.2%}",
            )

            comparison_3.metric(
                f"{ticker} buy-and-hold return",
                f"{backtest_statistics['buy_hold_return']:+.1%}",
            )

            st.subheader(
                "Strategy equity versus buy and hold"
            )

            equity_figure = go.Figure()

            equity_curve = (
                backtest_statistics["equity_curve"]
            )

            for column in equity_curve.columns:
                equity_figure.add_trace(
                    go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve[column],
                        mode="lines",
                        name=column,
                    )
                )

            equity_figure.update_layout(
                height=500,
                yaxis_title=(
                    "Hypothetical value of $10,000"
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )

            st.plotly_chart(
                equity_figure,
                use_container_width=True,
            )

            st.subheader(
                "Long and short results"
            )

            direction_breakdown = (
                build_direction_breakdown(
                    backtest_trades
                )
            )

            if not direction_breakdown.empty:
                display_breakdown = (
                    direction_breakdown.copy()
                )

                display_breakdown["Win rate"] = (
                    display_breakdown["Win rate"]
                    .map(
                        lambda value: (
                            f"{value:.1%}"
                        )
                    )
                )

                display_breakdown["Average return"] = (
                    display_breakdown[
                        "Average return"
                    ]
                    .map(
                        lambda value: (
                            f"{value:+.2%}"
                        )
                    )
                )

                display_breakdown[
                    "Total compounded return"
                ] = (
                    display_breakdown[
                        "Total compounded return"
                    ]
                    .map(
                        lambda value: (
                            f"{value:+.1%}"
                        )
                    )
                )

                st.dataframe(
                    display_breakdown,
                    hide_index=True,
                    use_container_width=True,
                )

            current_direction = None

            if trade_setup["bias"] == "LONG BIAS":
                current_direction = "LONG"

            elif trade_setup["bias"] == "SHORT BIAS":
                current_direction = "SHORT"

            st.subheader(
                "Historical context for today's reading"
            )

            if current_direction is None:
                st.info(
                    "Today's dashboard reading is WAIT, "
                    "so there is no matching long or short "
                    "signal to compare."
                )

            else:
                similar_trades = backtest_trades[
                    backtest_trades["Direction"]
                    == current_direction
                ]

                if similar_trades.empty:
                    st.info(
                        "No historical trades matched today's "
                        f"{current_direction.lower()} direction "
                        "under these settings."
                    )

                else:
                    context_1, context_2, context_3 = (
                        st.columns(3)
                    )

                    context_1.metric(
                        "Matching past trades",
                        f"{len(similar_trades)}",
                    )

                    context_2.metric(
                        "Past win rate",
                        (
                            f"{similar_trades['Winner'].mean():.1%}"
                        ),
                    )

                    context_3.metric(
                        "Average past return",
                        (
                            f"{similar_trades['Net Return'].mean():+.2%}"
                        ),
                    )

                    if len(similar_trades) < 10:
                        st.warning(
                            "This matching sample is small, so "
                            "its win rate is especially uncertain."
                        )

            st.subheader(
                "Most recent historical trades"
            )

            recent_trades = (
                backtest_trades
                .tail(20)
                .sort_values(
                    "Signal Date",
                    ascending=False,
                )
            )

            st.dataframe(
                format_backtest_trade_table(
                    recent_trades
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.warning(
                "Historical win rate is not a probability that "
                "the next trade will win. Results can change "
                "substantially with the ticker, holding period, "
                "quality threshold, costs, and market regime. "
                "Short results do not include stock-borrow fees."
            )

    with summary_tab:
        st.subheader(f"{ticker} snapshot")

        column_1, column_2, column_3 = st.columns(3)

        column_1.metric(
            "Latest closing price",
            f"${price:.2f}",
        )

        column_2.metric(
            "Daily change",
            f"${change:+.2f}",
            f"{change_percent:+.2f}%",
        )

        column_3.metric(
            "Trading volume",
            f"{int(latest['Volume']):,}",
        )

        column_4, column_5 = st.columns(2)

        column_4.metric(
            f"{selected_period.title()} high",
            f"${history['High'].max():.2f}",
        )

        column_5.metric(
            f"{selected_period.title()} low",
            f"${history['Low'].min():.2f}",
        )

        st.subheader("Technical trend")

        trend_box(f"{trend}: {trend_text}")

        column_6, column_7 = st.columns(2)

        column_6.metric(
            "20-day average",
            f"${ma20:.2f}",
        )

        column_7.metric(
            "50-day average",
            f"${ma50:.2f}",
        )

        st.subheader("RSI momentum")

        st.metric(
            "14-day RSI",
            f"{rsi:.1f}",
        )

        st.info(rsi_text)

        st.subheader("MACD momentum")

        column_8, column_9, column_10 = st.columns(3)

        column_8.metric(
            "MACD",
            f"{macd:.3f}",
        )

        column_9.metric(
            "Signal line",
            f"{signal:.3f}",
        )

        column_10.metric(
            "Histogram",
            f"{histogram:+.3f}",
        )

        macd_box(f"{macd_status}: {macd_text}")

        st.subheader("Bollinger Bands")

        column_11, column_12, column_13 = st.columns(3)

        column_11.metric(
            "Upper band",
            f"${upper_band:.2f}",
        )

        column_12.metric(
            "Middle band",
            f"${ma20:.2f}",
        )

        column_13.metric(
            "Lower band",
            f"${lower_band:.2f}",
        )

        st.info(f"Current position: {band_status}.")

        st.subheader("Dashboard summary")

        st.write(
            f"**{ticker}** shows a "
            f"**{trend.lower()}**, "
            f"**{rsi_status.lower()} RSI**, "
            f"**{macd_status.lower()}**, and is "
            f"**{band_status.lower()}**."
        )

        st.write(
            f"Current dashboard reading: "
            f"**{trade_setup['bias']}** with a "
            f"**{trade_setup['setup_quality']} / 100** "
            f"setup-quality score."
        )

    with price_tab:
        st.subheader(
            f"{ticker} — {selected_period.title()} Price Chart"
        )

        price_chart = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
        )

        price_chart.add_trace(
            go.Candlestick(
                x=history.index,
                open=history["Open"],
                high=history["High"],
                low=history["Low"],
                close=history["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        selected_overlays = st.multiselect(
            "Chart overlays",
            [
                "20-day average",
                "50-day average",
                "Bollinger Bands",
            ],
            default=[
                "20-day average",
                "50-day average",
            ],
            help=(
                "Turn overlays on or off to keep the chart easier "
                "to read."
            ),
        )

        chart_lines = []

        if "20-day average" in selected_overlays:
            chart_lines.append(
                ("MA20", "20-Day Average")
            )

        if "50-day average" in selected_overlays:
            chart_lines.append(
                ("MA50", "50-Day Average")
            )

        if "Bollinger Bands" in selected_overlays:
            chart_lines.extend(
                [
                    ("Upper Band", "Upper Bollinger Band"),
                    ("Lower Band", "Lower Bollinger Band"),
                ]
            )

        for column, name in chart_lines:
            price_chart.add_trace(
                go.Scatter(
                    x=history.index,
                    y=history[column],
                    mode="lines",
                    name=name,
                ),
                row=1,
                col=1,
            )

        price_chart.add_trace(
            go.Bar(
                x=history.index,
                y=history["Volume"],
                name="Volume",
            ),
            row=2,
            col=1,
        )

        price_chart.update_layout(
            height=720,
            xaxis_rangeslider_visible=False,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        price_chart.update_yaxes(
            title_text="Price",
            row=1,
            col=1,
        )

        price_chart.update_yaxes(
            title_text="Volume",
            row=2,
            col=1,
        )

        st.plotly_chart(
            price_chart,
            use_container_width=True,
        )

    with momentum_tab:
        st.subheader("RSI Chart")

        rsi_chart = go.Figure()

        rsi_chart.add_trace(
            go.Scatter(
                x=history.index,
                y=history["RSI"],
                mode="lines",
                name="RSI",
            )
        )

        rsi_chart.add_hline(
            y=70,
            line_dash="dash",
        )

        rsi_chart.add_hline(
            y=30,
            line_dash="dash",
        )

        rsi_chart.update_yaxes(
            range=[0, 100],
            title_text="RSI",
        )

        rsi_chart.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            rsi_chart,
            use_container_width=True,
        )

        st.subheader("MACD Chart")

        macd_chart = go.Figure()

        macd_chart.add_trace(
            go.Scatter(
                x=history.index,
                y=history["MACD"],
                mode="lines",
                name="MACD",
            )
        )

        macd_chart.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Signal"],
                mode="lines",
                name="Signal Line",
            )
        )

        macd_chart.add_trace(
            go.Bar(
                x=history.index,
                y=history["Histogram"],
                name="Histogram",
            )
        )

        macd_chart.update_layout(
            height=450,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            macd_chart,
            use_container_width=True,
        )

    with watchlist_tab:
        st.subheader("Watchlist snapshot")

        watchlist_symbols = parse_watchlist(
            watchlist_text,
            ticker,
        )

        st.caption(
            "The analyzed ticker is automatically included. "
            "Up to eight symbols are shown."
        )

        try:
            with st.spinner("Loading watchlist data..."):
                (
                    watchlist_table,
                    normalized_prices,
                    failed_symbols,
                ) = build_watchlist_data(watchlist_symbols)

        except Exception as error:
            watchlist_table = pd.DataFrame()
            normalized_prices = pd.DataFrame()
            failed_symbols = watchlist_symbols

            st.warning(
                f"Watchlist data could not be loaded: {error}"
            )

        if watchlist_table.empty:
            st.info(
                "No usable watchlist data was returned."
            )

        else:
            st.dataframe(
                watchlist_table,
                hide_index=True,
                use_container_width=True,
            )

        if failed_symbols:
            st.warning(
                "No usable data was returned for: "
                + ", ".join(failed_symbols)
            )

        if not normalized_prices.empty:
            st.subheader("Three-month performance comparison")

            comparison_chart = go.Figure()

            for symbol in normalized_prices.columns:
                comparison_chart.add_trace(
                    go.Scatter(
                        x=normalized_prices.index,
                        y=normalized_prices[symbol],
                        mode="lines",
                        name=symbol,
                    )
                )

            comparison_chart.add_hline(
                y=100,
                line_dash="dash",
            )

            comparison_chart.update_layout(
                height=520,
                yaxis_title="Starting value = 100",
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
            )

            st.plotly_chart(
                comparison_chart,
                use_container_width=True,
            )

            st.info(
                "Each line starts at 100, making percentage "
                "performance easier to compare even when the "
                "stocks have very different prices."
            )

    with news_tab:
        st.subheader(f"Latest {ticker} news")

        try:
            news_items = stock.get_news(
                count=10,
                tab="news",
            )
        except Exception as error:
            news_items = []
            st.warning(
                f"Company news could not be loaded: {error}"
            )

        parsed_news = [
            parse_news_item(item)
            for item in news_items
        ]

        parsed_news = [
            article
            for article in parsed_news
            if article["title"]
        ]

        if not parsed_news:
            st.info(
                "No recent company news was returned for this ticker."
            )

        else:
            for article in parsed_news:
                if article["url"]:
                    st.markdown(
                        f"#### [{article['title']}]"
                        f"({article['url']})"
                    )
                else:
                    st.markdown(
                        f"#### {article['title']}"
                    )

                details = article["publisher"]

                if article["published"]:
                    details += f" · {article['published']}"

                st.caption(details)

                if article["summary"]:
                    st.write(article["summary"])

                st.divider()

        st.subheader("Trending market screens")
        st.caption(
            "These tables are based on Yahoo Finance's "
            "predefined U.S. stock screeners."
        )

        movers_tab_1, movers_tab_2, movers_tab_3 = st.tabs(
            [
                "Most Active",
                "Day Gainers",
                "Day Losers",
            ]
        )

        screener_settings = [
            (
                movers_tab_1,
                "most_actives",
                "Most-active screen unavailable",
            ),
            (
                movers_tab_2,
                "day_gainers",
                "Day-gainers screen unavailable",
            ),
            (
                movers_tab_3,
                "day_losers",
                "Day-losers screen unavailable",
            ),
        ]

        for tab, screen_name, error_label in screener_settings:
            with tab:
                try:
                    table = build_screener_table(
                        screen_name,
                        count=10,
                    )

                    if table.empty:
                        st.info(
                            "No stocks were returned by this screen."
                        )
                    else:
                        st.dataframe(
                            table,
                            hide_index=True,
                            use_container_width=True,
                        )

                except Exception as error:
                    st.warning(f"{error_label}: {error}")

    st.caption(
        "These indicators and screens describe historical or "
        "current market data. They are not predictions or "
        "recommendations to buy or sell."
    )
