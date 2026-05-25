import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(
    page_title="Crypto Sentiment Dashboard",
    layout="wide"
)


st.title("📈 Crypto Sentiment Analytics Dashboard")

st.markdown("""
### Analyze trader behavior using:

- Bitcoin Fear & Greed Index
- Hyperliquid Trading Data
- Machine Learning Models
""")


sentiment = pd.read_csv("fear_greed_index (1).csv")
trades = pd.read_csv("historical_data (1).csv")


sentiment['date'] = pd.to_datetime(sentiment['date'])

trades['Timestamp IST'] = pd.to_datetime(
    trades['Timestamp IST'],
    format="%d-%m-%Y %H:%M",
    errors='coerce'
)

trades['Date'] = trades['Timestamp IST'].dt.date
sentiment['Date'] = sentiment['date'].dt.date


merged = pd.merge(
    trades.astype({'Date':'datetime64[ns]'}),
    sentiment.astype({'Date':'datetime64[ns]'}),
    on='Date',
    how='inner'
)

merged['Closed PnL'] = pd.to_numeric(
    merged['Closed PnL'],
    errors='coerce'
)

merged['win'] = merged['Closed PnL'] > 0


st.success("Datasets merged successfully!")


st.header("📊 Dashboard Metrics")

st.write(f"### Total Trades: {len(merged)}")

st.write(
    f"### Average PnL: {round(merged['Closed PnL'].mean(),2)}"
)

st.write(
    f"### Win Rate: {round(merged['win'].mean()*100,2)}%"
)

st.write(
    f"### Average Trade Size: {round(merged['Size Tokens'].mean(),2)}"
)



st.header("📄 Merged Dataset Preview")

st.write(merged.head())



st.header("💰 PnL by Market Sentiment")

pnl = merged.groupby(
    'classification'
)['Closed PnL'].mean()

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.barplot(
    x=pnl.index,
    y=pnl.values,
    ax=ax1
)

plt.xticks(rotation=15)

st.pyplot(fig1)


st.header("🏆 Win Rate Analysis")

winrate = merged.groupby(
    'classification'
)['win'].mean()

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.barplot(
    x=winrate.index,
    y=winrate.values,
    ax=ax2
)

plt.xticks(rotation=15)

st.pyplot(fig2)



st.header("📦 PnL Distribution")

fig3, ax3 = plt.subplots(figsize=(10,5))

sns.boxplot(
    x='classification',
    y='Closed PnL',
    data=merged,
    ax=ax3
)

plt.xticks(rotation=15)

st.pyplot(fig3)



st.header("📈 Buy vs Sell Analysis")

bs = merged.groupby(
    ['classification','Side']
)['Closed PnL'].mean().reset_index()

fig4, ax4 = plt.subplots(figsize=(10,5))

sns.barplot(
    data=bs,
    x='classification',
    y='Closed PnL',
    hue='Side',
    ax=ax4
)

plt.xticks(rotation=15)

st.pyplot(fig4)


st.header("🤖 Machine Learning")

encoder = LabelEncoder()

merged['side_encoded'] = encoder.fit_transform(
    merged['Side']
)

merged['sentiment_encoded'] = encoder.fit_transform(
    merged['classification']
)

merged['hour'] = merged[
    'Timestamp IST'
].dt.hour

features = [
    'Size Tokens',
    'hour',
    'side_encoded',
    'sentiment_encoded'
]

X = merged[features]
y = merged['win']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.write(
    f"### Random Forest Accuracy: {round(accuracy*100,2)}%"
)


importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

st.header("🔥 Feature Importance")

fig5, ax5 = plt.subplots(figsize=(8,5))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature',
    ax=ax5
)

st.pyplot(fig5)


st.header("🧠 Trader Clustering")

cluster_data = merged[
    ['Size Tokens', 'Closed PnL']
].dropna()

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

cluster_data['cluster'] = kmeans.fit_predict(
    cluster_data
)

fig6, ax6 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x='Size Tokens',
    y='Closed PnL',
    hue='cluster',
    data=cluster_data,
    ax=ax6
)

st.pyplot(fig6)



st.header("📉 ROC Curve")

probs = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(
    y_test,
    probs
)

auc_score = roc_auc_score(
    y_test,
    probs
)

fig7, ax7 = plt.subplots(figsize=(8,5))

ax7.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.2f}"
)

ax7.plot([0,1],[0,1],'--')

ax7.legend()

st.pyplot(fig7)


st.header("📌 Key Insights")

st.markdown("""
- Traders behave differently during Fear and Greed markets.
- Market sentiment impacts profitability.
- Machine learning predicts trade success patterns.
- Clustering reveals trader behavior groups.
- Sentiment influences win rates significantly.
""")