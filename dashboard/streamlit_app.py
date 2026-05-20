import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from automation.signal_engine import (
    generate_signal
)

from automation.database import (
    fetch_prediction_history
)


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(

    page_title="SBI AI Forecasting System",

    layout="wide"
)


# =========================================
# TITLE
# =========================================

st.title("📈 SBI AI Forecasting Dashboard")

st.markdown(
    "Transformer-Based Financial Intelligence System"
)

st.divider()


# =========================================
# LOAD STOCK DATA
# =========================================

@st.cache_data

def load_stock_data():

    df = pd.read_csv(
        "data/processed/sbi_features.csv"
    )

    return df


df = load_stock_data()


# =========================================
# GENERATE SIGNALS
# =========================================

results = generate_signal()


# =========================================
# METRICS ROW
# =========================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Current Price",

        f"₹{results['current_price']}"
    )

with col2:

    st.metric(

        "Predicted Price",

        f"₹{results['predicted_price']}"
    )

with col3:

    st.metric(

        "Predicted Move",

        f"{results['movement_percent']}%"
    )

with col4:

    st.metric(

        "Confidence",

        f"{results['confidence']}%"
    )


st.divider()


# =========================================
# SIGNAL DISPLAY
# =========================================

signal = results["final_signal"]

if signal == "STRONG BUY":

    st.success(
        f"🚀 SIGNAL: {signal}"
    )

elif signal == "BUY":

    st.success(
        f"📈 SIGNAL: {signal}"
    )

elif signal == "SELL":

    st.error(
        f"📉 SIGNAL: {signal}"
    )

elif signal == "STRONG SELL":

    st.error(
        f"⚠️ SIGNAL: {signal}"
    )

else:

    st.warning(
        f"⏸️ SIGNAL: {signal}"
    )


# =========================================
# SENTIMENT SECTION
# =========================================

st.subheader(
    "📰 Market Sentiment"
)

sentiment_col1, sentiment_col2 = st.columns(2)

with sentiment_col1:

    st.metric(

        "Sentiment",

        results["sentiment"]
    )

with sentiment_col2:

    st.metric(

        "Sentiment Score",

        results["sentiment_score"]
    )


st.divider()


# =========================================
# CANDLESTICK CHART
# =========================================

st.subheader(
    "📊 SBI Candlestick Chart"
)

fig = go.Figure(

    data=[

        go.Candlestick(

            x=df["Date"],

            open=df["Open"],

            high=df["High"],

            low=df["Low"],

            close=df["Close"]
        )
    ]
)

fig.update_layout(

    height=600,

    xaxis_rangeslider_visible=False
)

st.plotly_chart(

    fig,

    use_container_width=True
)


# =========================================
# HISTORICAL PREDICTIONS
# =========================================

st.divider()

st.subheader(
    "🗂️ Historical Predictions"
)

history = fetch_prediction_history()

if history:

    history_df = pd.DataFrame(history)

    st.dataframe(

        history_df,

        use_container_width=True
    )

else:

    st.warning(
        "No prediction history available"
    )


# =========================================
# PERFORMANCE ANALYTICS
# =========================================

st.divider()

st.subheader(
    "📈 Performance Analytics"
)

analytics_col1, analytics_col2 = st.columns(2)

with analytics_col1:

    st.metric(

        "Average Confidence",

        f"{history_df['confidence'].mean():.2f}%"
        if history else "N/A"
    )

with analytics_col2:

    st.metric(

        "Average Movement",

        f"{history_df['movement_percent'].mean():.2f}%"
        if history else "N/A"
    )


# =========================================
# FOOTER
# =========================================

st.divider()

st.caption(
    "Built with PatchTST + FinBERT + Streamlit"
)