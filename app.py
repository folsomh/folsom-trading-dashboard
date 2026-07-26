import math
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Folsom's Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Folsom's Trading Dashboard")
st.caption(
    "Technical indicators, company news, and market-mover screens "
    "based on Yahoo Finance data."
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


period_choices = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}

with st.form("analysis_form"):
    ticker = st.text_input(
        "Enter a stock ticker",
        "AAL",
    ).strip().upper()

    selected_period = st.selectbox(
        "Choose a chart range",
        list(period_choices.keys()),
    )

    watchlist_text = st.text_input(
        "Watchlist tickers (separate them with commas)",
        "VTI, VXUS, AAL, CVX, INTC",
    )

    analyze = st.form_submit_button("Analyze")


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

    (
        trade_tab,
        summary_tab,
        price_tab,
        momentum_tab,
        watchlist_tab,
        news_tab,
    ) = st.tabs(
        [
            "Trade Setup",
            "Summary",
            "Price Chart",
            "Momentum Charts",
            "Watchlist & Compare",
            "News & Trending",
        ]
    )

    with trade_tab:
        st.subheader(f"{ticker} simple trade setup")

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

        quality_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=trade_setup["setup_quality"],
                title={
                    "text": "Technical setup quality"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                },
            )
        )

        quality_gauge.update_layout(
            height=320,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
        )

        st.plotly_chart(
            quality_gauge,
            use_container_width=True,
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

        st.subheader("Why the dashboard gave this reading")

        for reason in trade_setup["evidence"]:
            st.write(f"• {reason}")

        st.subheader("Risk flags")

        if trade_setup["risk_flags"]:
            for warning in trade_setup["risk_flags"]:
                st.write(f"• {warning}")
        else:
            st.write("• No major indicator-stretch warnings were detected.")

        st.warning(
            "This score measures indicator agreement, not the probability "
            "of making money. It does not know your entry, stop loss, "
            "position size, breaking news, or personal risk tolerance."
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

        chart_lines = [
            ("MA20", "20-Day Average"),
            ("MA50", "50-Day Average"),
            ("Upper Band", "Upper Bollinger Band"),
            ("Lower Band", "Lower Bollinger Band"),
        ]

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
