import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="Cambodia Food Price Dashboard", layout="wide", page_icon="🍚")

EXCLUDE_COLS = ['N0', 'Food commodity', 'Pricing Type', 'Currency', "YTD'24", "YTD'25", 'Change (%)']

# ---------- Data loading ----------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file) if hasattr(file, "read") else pd.read_csv(file)
    date_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    df[date_cols] = df[date_cols].replace('-', np.nan)
    df[date_cols] = df[date_cols].apply(pd.to_numeric, errors='coerce')

    long_df = df.melt(
        id_vars=['Food commodity', 'Pricing Type', 'Currency'],
        value_vars=date_cols,
        var_name='Date',
        value_name='Price'
    )
    long_df['Date'] = pd.to_datetime(long_df['Date'], errors='coerce')
    long_df.dropna(subset=['Price'], inplace=True)

    yoy = df[['Food commodity', 'Pricing Type', "YTD'24", "YTD'25", 'Change (%)']].copy()
    yoy['Change (%)'] = yoy['Change (%)'] * 100

    return df, long_df, yoy, date_cols

# ---------- ML model training ----------
@st.cache_resource
def train_models(long_df):
    ml_df = long_df.copy()
    ml_df['Year'] = ml_df['Date'].dt.year
    ml_df['Month'] = ml_df['Date'].dt.month

    commodity_encoder = LabelEncoder()
    pricing_encoder = LabelEncoder()
    ml_df['Commodity_ID'] = commodity_encoder.fit_transform(ml_df['Food commodity'])
    ml_df['Pricing_ID'] = pricing_encoder.fit_transform(ml_df['Pricing Type'])

    features = ['Year', 'Month', 'Commodity_ID', 'Pricing_ID']
    X = ml_df[features]
    y = ml_df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    metrics = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    models['Linear Regression'] = lr
    metrics.append(['Linear Regression', mean_absolute_error(y_test, pred),
                     np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)])

    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    models['Random Forest'] = rf
    metrics.append(['Random Forest', mean_absolute_error(y_test, pred),
                     np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)])

    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        models['XGBoost'] = xgb
        metrics.append(['XGBoost', mean_absolute_error(y_test, pred),
                         np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)])

    metrics_df = pd.DataFrame(metrics, columns=['Model', 'MAE', 'RMSE', 'R2'])
    return models, metrics_df, commodity_encoder, pricing_encoder, features


def linear_trend_forecast(item_df, periods):
    """Simple per-item linear trend extrapolation as a sanity-check baseline."""
    item_df = item_df.sort_values('Date').reset_index(drop=True)
    t = np.arange(len(item_df)).reshape(-1, 1)
    y = item_df['Price'].values
    if len(item_df) < 2:
        return None
    model = LinearRegression().fit(t, y)
    future_t = np.arange(len(item_df), len(item_df) + periods).reshape(-1, 1)
    return model.predict(future_t)


st.sidebar.title("🍚 Food Price Dashboard")
st.sidebar.caption("Cambodia Retail & Wholesale Commodity Prices (KHR)")

uploaded = st.sidebar.file_uploader("Upload dataset (.xlsx or .csv)", type=["xlsx", "csv"])
default_path = "food_prices.csv"

try:
    if uploaded is not None:
        raw_df, long_df, yoy, date_cols = load_data(uploaded)
    else:
        raw_df, long_df, yoy, date_cols = load_data(default_path)
except FileNotFoundError:
    st.error("No dataset found. Please upload the Excel/CSV file using the sidebar.")
    st.stop()

# ---------- Sidebar filters ----------
pricing_types = sorted(long_df['Pricing Type'].unique())
commodities = sorted(long_df['Food commodity'].unique())

selected_pricing = st.sidebar.multiselect("Pricing Type", pricing_types, default=pricing_types)
selected_commodities = st.sidebar.multiselect(
    "Commodities (leave empty = all)", commodities, default=[]
)

min_date, max_date = long_df['Date'].min(), long_df['Date'].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

filtered = long_df[long_df['Pricing Type'].isin(selected_pricing)]
if selected_commodities:
    filtered = filtered[filtered['Food commodity'].isin(selected_commodities)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered['Date'] >= start) & (filtered['Date'] <= end)]

yoy_filtered = yoy[yoy['Pricing Type'].isin(selected_pricing)]
if selected_commodities:
    yoy_filtered = yoy_filtered[yoy_filtered['Food commodity'].isin(selected_commodities)]

st.title("🍚 Cambodia Food Commodity Price Dashboard")
st.caption(f"{raw_df['Food commodity'].nunique()} commodities · Retail & Wholesale · KHR · "
           f"{long_df['Date'].min().strftime('%b %Y')} – {long_df['Date'].max().strftime('%b %Y')}")

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Commodities", filtered['Food commodity'].nunique())
col2.metric("Avg Price (KHR)", f"{filtered['Price'].mean():,.0f}" if len(filtered) else "N/A")
avg_change = yoy_filtered['Change (%)'].mean()
col3.metric("Avg YoY Change", f"{avg_change:+.1f}%" if pd.notna(avg_change) else "N/A")
col4.metric("Records", len(filtered))

st.divider()

tabs = st.tabs([
    "📈 Trends", "💰 Rankings", "📊 Retail vs Wholesale",
    "🔥 YoY Inflation", "🌡️ Volatility", "🔗 Correlation",
    "🔮 Forecast", "🗂️ Raw Data"
])

# ---------- Trends ----------
with tabs[0]:
    st.subheader("Monthly Average Price Trend")
    monthly = filtered.groupby(['Date', 'Pricing Type'])['Price'].mean().reset_index()
    fig = px.line(monthly, x='Date', y='Price', color='Pricing Type', markers=True)
    fig.update_layout(yaxis_title="Average Price (KHR)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Commodity Price Trend")
    default_sel = selected_commodities if selected_commodities else commodities[:5]
    trend_items = st.multiselect("Select commodities to compare", commodities, default=default_sel, key="trend_items")
    trend_df = filtered[filtered['Food commodity'].isin(trend_items)]
    if len(trend_df):
        fig2 = px.line(trend_df, x='Date', y='Price', color='Food commodity',
                        line_dash='Pricing Type', markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Select at least one commodity above.")

# ---------- Rankings ----------
with tabs[1]:
    st.subheader("Most & Least Expensive Commodities (Average Price)")
    n = st.slider("Number of items", 5, 30, 10)
    avg_price = filtered.groupby('Food commodity')['Price'].mean().sort_values(ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Most Expensive**")
        top_n = avg_price.head(n).reset_index()
        fig3 = px.bar(top_n, x='Price', y='Food commodity', orientation='h', color='Price',
                       color_continuous_scale='Reds')
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.markdown("**Top Cheapest**")
        bottom_n = avg_price.tail(n).sort_values().reset_index()
        fig4 = px.bar(bottom_n, x='Price', y='Food commodity', orientation='h', color='Price',
                       color_continuous_scale='Greens')
        fig4.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Price Distribution")
    fig5 = px.histogram(filtered, x='Price', color='Pricing Type', marginal='box', nbins=40, opacity=0.7)
    st.plotly_chart(fig5, use_container_width=True)

# ---------- Retail vs Wholesale ----------
with tabs[2]:
    st.subheader("Retail vs Wholesale — Average Price by Commodity")
    avg_compare = filtered.groupby(['Food commodity', 'Pricing Type'])['Price'].mean().reset_index()
    pivot_compare = avg_compare.pivot(index='Food commodity', columns='Pricing Type', values='Price')
    if 'Retail' in pivot_compare.columns and 'Wholesale' in pivot_compare.columns:
        pivot_compare['Difference'] = pivot_compare['Retail'] - pivot_compare['Wholesale']
        pivot_compare = pivot_compare.sort_values('Difference', ascending=False)
        st.dataframe(pivot_compare.style.format("{:,.0f}"), use_container_width=True)

        fig6 = px.bar(pivot_compare.reset_index().head(20), x='Food commodity', y='Difference',
                       color='Difference', color_continuous_scale='RdBu_r',
                       title="Retail Premium Over Wholesale (Top 20)")
        fig6.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Select both Retail and Wholesale in the sidebar to compare.")

    st.subheader("Distribution: Retail vs Wholesale")
    fig7 = px.box(filtered, x='Pricing Type', y='Price', color='Pricing Type')
    st.plotly_chart(fig7, use_container_width=True)

# ---------- YoY Inflation ----------
with tabs[3]:
    st.subheader("YTD'24 vs YTD'25 — Highest Gainers & Decliners")
    n2 = st.slider("Number of items to show", 5, 30, 15, key="yoy_n")

    top_gain = yoy_filtered.sort_values('Change (%)', ascending=False).head(n2)
    top_decline = yoy_filtered.sort_values('Change (%)', ascending=True).head(n2)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Biggest Price Increases**")
        fig8 = px.bar(top_gain, x='Change (%)', y='Food commodity', color='Pricing Type', orientation='h')
        fig8.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig8, use_container_width=True)
    with c2:
        st.markdown("**Biggest Price Declines**")
        fig9 = px.bar(top_decline, x='Change (%)', y='Food commodity', color='Pricing Type', orientation='h')
        fig9.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Average YTD Comparison by Pricing Type")
    comparison = yoy_filtered.groupby('Pricing Type')[["YTD'24", "YTD'25"]].mean().reset_index()
    comparison_melt = comparison.melt(id_vars='Pricing Type', var_name='Period', value_name='KHR')
    fig10 = px.bar(comparison_melt, x='Pricing Type', y='KHR', color='Period', barmode='group')
    st.plotly_chart(fig10, use_container_width=True)

    st.dataframe(yoy_filtered.sort_values('Change (%)', ascending=False), use_container_width=True)

# ---------- Volatility ----------
with tabs[4]:
    st.subheader("Price Volatility (Std. Dev.) by Commodity")
    n3 = st.slider("Number of items", 5, 30, 15, key="vol_n")
    volatility = (
        filtered.groupby(['Pricing Type', 'Food commodity'])['Price']
        .std().reset_index().rename(columns={'Price': 'Volatility'})
        .sort_values('Volatility', ascending=False)
    )
    fig11 = px.bar(volatility.head(n3), x='Volatility', y='Food commodity', color='Pricing Type', orientation='h')
    fig11.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig11, use_container_width=True)
    st.dataframe(volatility, use_container_width=True)

# ---------- Correlation ----------
with tabs[5]:
    st.subheader("Correlation Heatmap — Top Commodities by Avg Price")
    n4 = st.slider("Top N commodities", 5, 40, 20, key="corr_n")
    top_items = filtered.groupby('Food commodity')['Price'].mean().sort_values(ascending=False).head(n4).index
    corr_df = (
        filtered[filtered['Food commodity'].isin(top_items)]
        .pivot_table(index='Date', columns='Food commodity', values='Price')
    )
    corr_matrix = corr_df.corr()
    fig12 = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
    st.plotly_chart(fig12, use_container_width=True)

# ---------- Forecast ----------
with tabs[6]:
    st.subheader("🔮 Price Forecasting (Retail & Wholesale)")
    st.caption(
        "Models are trained once on the full historical dataset (Year, Month, Commodity, "
        "Pricing Type → Price) and reused across every commodity you select below."
    )

    with st.spinner("Training models..."):
        models, metrics_df, commodity_encoder, pricing_encoder, features = train_models(long_df)

    st.markdown("**Model Comparison (on held-out 20% test set)**")
    st.dataframe(
        metrics_df.style.format({'MAE': '{:,.0f}', 'RMSE': '{:,.0f}', 'R2': '{:.3f}'}),
        use_container_width=True
    )

    model_options = list(models.keys())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fc_commodity = st.selectbox("Commodity", commodities, key="fc_commodity")
    with c2:
        fc_pricing = st.selectbox("Pricing Type", pricing_types, key="fc_pricing")
    with c3:
        fc_model_name = st.selectbox("Model", model_options, index=len(model_options) - 1, key="fc_model")
    with c4:
        fc_horizon = st.slider("Forecast horizon (months)", 3, 24, 12, key="fc_horizon")

    item_hist = long_df[
        (long_df['Food commodity'] == fc_commodity) & (long_df['Pricing Type'] == fc_pricing)
    ].sort_values('Date')

    if item_hist.empty:
        st.warning("No historical data for this commodity / pricing type combination.")
    else:
        last_date = item_hist['Date'].max()
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=fc_horizon, freq='MS')

        future_X = pd.DataFrame({
            'Year': future_dates.year,
            'Month': future_dates.month,
            'Commodity_ID': commodity_encoder.transform([fc_commodity] * fc_horizon),
            'Pricing_ID': pricing_encoder.transform([fc_pricing] * fc_horizon),
        })[features]

        chosen_model = models[fc_model_name]
        ml_forecast = chosen_model.predict(future_X)

        trend_forecast = linear_trend_forecast(item_hist, fc_horizon)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=item_hist['Date'], y=item_hist['Price'], mode='lines+markers', name='Historical'
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=ml_forecast, mode='lines+markers', name=f'{fc_model_name} Forecast',
            line=dict(dash='dash')
        ))
        if trend_forecast is not None:
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=trend_forecast, mode='lines', name='Linear Trend (baseline)',
                line=dict(dash='dot')
            ))
        fig_fc.update_layout(
            title=f"{fc_commodity} — {fc_pricing} Price Forecast",
            yaxis_title="Price (KHR)", xaxis_title="Date"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.info(
            "⚠️ Tree-based models (Random Forest / XGBoost) learn patterns from historical "
            "Year/Month combinations and tend to flatten out beyond the training range rather than "
            "extrapolate a trend. The dotted **Linear Trend** line is a simple per-item baseline "
            "that better reflects directional momentum — compare both before relying on the forecast."
        )

        forecast_table = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m'),
            f'{fc_model_name} Forecast (KHR)': ml_forecast.round(0),
            'Linear Trend (KHR)': (trend_forecast.round(0) if trend_forecast is not None else np.nan)
        })
        st.dataframe(forecast_table, use_container_width=True)
        st.download_button(
            "Download forecast as CSV", forecast_table.to_csv(index=False),
            f"forecast_{fc_commodity.replace(' ', '_')}_{fc_pricing}.csv"
        )

        if fc_model_name in ('Random Forest', 'XGBoost'):
            st.markdown("**Feature Importance**")
            imp_df = pd.DataFrame({
                'Feature': features, 'Importance': chosen_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

# ---------- Raw data ----------
with tabs[7]:
    st.subheader("Raw Dataset")
    st.dataframe(raw_df, use_container_width=True)
    st.subheader("Cleaned Long-Format Data (filtered)")
    st.dataframe(filtered, use_container_width=True)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False), "filtered_food_prices.csv")

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit · Data: Cambodia food commodity retail & wholesale prices")
