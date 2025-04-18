import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="No-Code SPSS++", layout="wide")
st.title("📊 No-Code SPSS++ — Now with Regression & Smart Insights")

uploaded_file = st.file_uploader("📤 Upload a CSV or Excel file", type=["csv", "xlsx"])

def detect_multiple_choice_columns(df, threshold=0.8):
    mc_cols = []
    for col in df.columns:
        filled_ratio = df[col].notna().mean()
        unique_vals = df[col].dropna().unique()
        if 0 < filled_ratio < threshold and len(unique_vals) < 10:
            mc_cols.append(col)
    return mc_cols

def generate_insight_cards(df, segment_col, numeric_col):
    insights = []
    grouped = df[[segment_col, numeric_col]].dropna()
    groups = grouped.groupby(segment_col)[numeric_col]
    means = groups.mean()
    top_group = means.idxmax()
    bottom_group = means.idxmin()
    diff = means[top_group] - means[bottom_group]
    if len(groups) >= 2:
        stat, p = stats.ttest_ind(*[g[1] for g in grouped.groupby(segment_col)])
        if p < 0.05:
            insights.append(f"🔍 **{top_group}** has the highest average score ({means[top_group]:.2f}) for **{numeric_col}**, significantly higher than **{bottom_group}** ({means[bottom_group]:.2f}) (p = {p:.4f}).")
    return insights

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    mc_columns = detect_multiple_choice_columns(df)

    st.markdown("### 🧠 Smart Insight Cards")
    selected_segment = st.selectbox("Segment by (for insights)", categorical_columns)
    selected_metric = st.selectbox("Numeric variable to analyze", numeric_columns, key="insight_metric")
    insight_results = generate_insight_cards(df, selected_segment, selected_metric)
    for insight in insight_results:
        st.markdown(insight)

    st.markdown("## 📉 Linear Regression")
    with st.expander("Linear Regression"):
        target_var = st.selectbox("Target (numeric)", numeric_columns, key="lin_target")
        predictors = st.multiselect("Predictors", numeric_columns, default=[col for col in numeric_columns if col != target_var])
        if st.button("Run Linear Regression"):
            X = df[predictors].dropna()
            y = df[target_var].dropna()
            common_idx = X.index.intersection(y.index)
            X = sm.add_constant(X.loc[common_idx])
            y = y.loc[common_idx]
            model = sm.OLS(y, X).fit()
            st.text(model.summary())

    st.markdown("## 🔀 Logistic Regression (Binary)")
    with st.expander("Logistic Regression"):
        cat_targets = [col for col in categorical_columns if df[col].nunique() == 2]
        if cat_targets:
            target_col = st.selectbox("Target (binary categorical)", cat_targets)
            predictors = st.multiselect("Predictors (numeric only)", numeric_columns, key="log_preds")
            if st.button("Run Logistic Regression"):
                df_clean = df[[target_col] + predictors].dropna()
                X = df_clean[predictors]
                y = LabelEncoder().fit_transform(df_clean[target_col])
                model = LogisticRegression(max_iter=1000).fit(X, y)
                st.markdown("### Coefficients:")
                for name, coef in zip(predictors, model.coef_[0]):
                    st.write(f"- **{name}**: {coef:.4f}")
        else:
            st.warning("No binary categorical variables found for logistic regression.")

    st.markdown("## 📈 Correlation Matrix")
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.markdown("## 📊 Raw Data Table")
    st.dataframe(df.head())

    st.markdown("## 📈 Full Summary Statistics")
    st.write(df.describe(include='all'))