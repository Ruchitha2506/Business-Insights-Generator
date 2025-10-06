# Importing necessary Libraries

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ---- Streamlit page config (must be before any UI) ----
st.set_page_config(page_title="Business Insights Generator", page_icon="📊", layout="wide")

# ---- Simple visual theme ----
COLOR_PRIMARY = "#4A90E2"   # blue
COLOR_ACCENT  = "#50E3C2"   # teal
COLOR_WARN    = "#F5A623"   # orange

def fmt_currency(ax):
    """Format y-axis with thousands separators and add a light grid."""
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.grid(True, linestyle="--", alpha=0.35)

def show_section_header(name: str, available: bool) -> bool:
    """Show a section header; if not available, show 'None' message."""
    st.subheader(name)
    if not available:
        st.caption("**None** — not applicable for this dataset.")
    return available

# ---- LLM toggle ----
USE_LLM = True if os.getenv("OPENAI_API_KEY") else False

# ---- App title & intro ----
st.title("📊 Business Insights Generator")
st.markdown(
    "Upload a CSV (sales, ecommerce, marketing, finance, etc.) to get **automated KPIs**, "
    "**clean visualizations**, and an optional **executive narrative**."
)

# ---- File inputs ----
uploaded = st.file_uploader("Upload CSV", type=["csv"])
sample_btn = st.button("Use bundled sample dataset")

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
def load_data(file):
    """
    Robust CSV loader:
    - Parses 'date' if present
    - Coerces key numeric cols (revenue, unit_price, quantity, discount)
    - Handles thousand separators and string numbers
    """
    # Peek columns (without consuming file-like)
    preview = pd.read_csv(file, nrows=1)
    parse_dates = ["date"] if "date" in preview.columns else None

    # Rewind file-like object if needed
    if hasattr(file, "seek"):
        file.seek(0)

    # Full read; accept thousands separator
    df = pd.read_csv(file, parse_dates=parse_dates, thousands=",")

    # Coerce likely numeric columns
    for col in ["revenue", "unit_price", "quantity", "discount"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False)
                .replace({"": None})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize object cols to string (keeps app stable)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)

    return df

# ------------------------------------------------------------
# KPI computation
# ------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Build KPIs using a robust revenue series.
    - Detects price-like column
    - Uses quantity*price if available, else price as value proxy
    - Computes MoM growth if 'date' present
    """
    kpis = {}

    # detect a "price" column (e.g., price, unit price)
    colnames_lower = [c.lower() for c in df.columns]
    price_col = next((c for c in df.columns if "price" in c.lower()), None)

    # build a revenue/value series
    if "quantity" in colnames_lower and price_col:
        rev = pd.to_numeric(df[price_col], errors="coerce") * pd.to_numeric(df["quantity"], errors="coerce")
    elif price_col:
        rev = pd.to_numeric(df[price_col], errors="coerce")
    elif "revenue" in df.columns:
        rev = pd.to_numeric(df["revenue"], errors="coerce")
    else:
        rev = None

    if rev is not None:
        rev = rev.fillna(0.0)
        df["__rev__"] = rev
        kpis["Total Revenue"] = float(rev.sum())
        kpis["Avg Value"] = float(rev.mean())
    else:
        kpis["Total Rows"] = int(len(df))

    # MoM growth (if date)
    if "date" in df.columns:
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"])
        tmp["month"] = tmp["date"].dt.to_period("M")
        if "__rev__" in tmp.columns:
            mom = tmp.groupby("month")["__rev__"].sum().pct_change().dropna()
            if len(mom) > 0:
                kpis["Latest MoM Growth %"] = round(float(mom.iloc[-1] * 100), 2)

    return kpis

def summarize_for_prompt(df: pd.DataFrame, kpis: dict, max_rows: int = 20) -> dict:
    """Small JSON context for the LLM."""
    sample = df.head(max_rows).to_dict(orient="records")
    return {
        "basic_columns": list(df.columns),
        "row_count": int(len(df)),
        "kpis": kpis,
        "sample_records": sample
    }

# ------------------------------------------------------------
# LLM narrative (optional)
# ------------------------------------------------------------
def generate_llm_insights(context: dict) -> str:
    if not USE_LLM:
        return "LLM not configured. Set OPENAI_API_KEY to receive narrative insights."
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = f"""You are a seasoned business analyst. Based on the structured context below,
return an executive summary that includes: (1) Key trends, (2) Anomalies or risks,
(3) 3–5 actionable recommendations. Be concise and business-friendly.

Context (JSON):
{context}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating LLM insights: {e}"

# ------------------------------------------------------------
# Main flow
# ------------------------------------------------------------
data_path = None
if sample_btn:
    data_path = os.path.join(os.path.dirname(__file__), "sample_sales.csv")
elif uploaded is not None:
    data_path = uploaded

if not data_path:
    st.info("Upload a CSV or click **Use bundled sample dataset** to begin.")
    st.stop()

# Load + show preview
df = load_data(data_path)
st.subheader("Preview")
st.dataframe(df.head(15), use_container_width=True)

# KPIs
kpis = compute_kpis(df)
st.subheader("KPIs")
kpi_cols = st.columns(max(len(kpis), 1))
for i, (k, v) in enumerate(kpis.items()):
    with kpi_cols[i]:
        st.metric(k, f"{v:,.2f}" if isinstance(v, float) else v)

# Executive narrative (LLM)
st.subheader("Executive Narrative (LLM)")
context = summarize_for_prompt(df, kpis)
insights = generate_llm_insights(context)
st.write(insights)
st.download_button("Download Summary", data=insights, file_name="executive_summary.txt")

# ------------------------------------------------------------
# Visualizations (polished + NA handling)
# ------------------------------------------------------------
st.markdown("## 📊 Visualizations")

rev_col = "__rev__" if "__rev__" in df.columns else ("revenue" if "revenue" in df.columns else None)

# 1) Revenue by Category (Top 10)
available = bool(rev_col and "category" in df.columns)
if show_section_header("💰 Total Revenue by Category (Top 10)", available):
    cat = (
        df.groupby("category")[rev_col]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cat["category"].astype(str), cat[rev_col], color=COLOR_PRIMARY)
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")
    ax.set_title("Total Revenue by Category (Top 10)")
    plt.xticks(rotation=45, ha="right")
    fmt_currency(ax)
    st.pyplot(fig)

# 2) Revenue by Region
available = bool(rev_col and "region" in df.columns)
if show_section_header("🌎 Total Revenue by Region", available):
    reg = (
        df.groupby("region")[rev_col]
          .sum()
          .sort_values(ascending=False)
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(reg["region"].astype(str), reg[rev_col], color=COLOR_PRIMARY)
    ax.set_xlabel("Region")
    ax.set_ylabel("Revenue")
    ax.set_title("Total Revenue by Region")
    fmt_currency(ax)
    st.pyplot(fig)

# 3) Monthly Revenue Trend
available = bool(rev_col and "date" in df.columns)
if show_section_header("📈 Monthly Revenue Trend", available):
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])
    if len(tmp) == 0:
        st.caption("**None** — dates could not be parsed.")
    else:
        tmp["month"] = tmp["date"].dt.to_period("M")
        monthly = tmp.groupby("month")[rev_col].sum().reset_index()
        monthly["month"] = monthly["month"].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(monthly["month"], monthly[rev_col], marker="o", color=COLOR_PRIMARY)
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.set_title("Monthly Revenue Trend")
        plt.xticks(rotation=45, ha="right")
        fmt_currency(ax)
        fig.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------------
# Data Explorer (dataset-agnostic)
# ------------------------------------------------------------
st.markdown("## 🧭 Data Explorer")

all_cols = list(df.columns)
if len(all_cols) == 0:
    st.caption("**None** — dataset is empty.")
else:
    col_to_explore = st.selectbox("Choose a column to explore", all_cols)
    series = df[col_to_explore]

    # Try datetime coercion on the fly
    maybe_dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    is_dt_like = maybe_dt.notna().sum() > 0.8 * len(series)

    if is_dt_like:
        # DATETIME → counts by day or month
        st.markdown(f"### 🗓️ Records over time — `{col_to_explore}`")
        granularity = st.radio("Group by", ["Day", "Month"], horizontal=True, key="gran_"+col_to_explore)
        buckets = (
            maybe_dt.dt.to_period("M").dt.to_timestamp()
            if granularity == "Month" else
            maybe_dt.dt.to_period("D").dt.to_timestamp()
        )
        counts = pd.DataFrame({"bucket": buckets}).dropna().groupby("bucket").size().reset_index(name="count")

        if len(counts) == 0:
            st.caption("**None** — no valid dates.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(counts["bucket"], counts["count"], marker="o", color=COLOR_PRIMARY)
            ax.set_xlabel("Time")
            ax.set_ylabel("Count")
            ax.set_title(f"Records per {granularity.lower()}")
            plt.xticks(rotation=45, ha="right")
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            st.pyplot(fig)

    elif pd.api.types.is_numeric_dtype(series):
        # NUMERIC → histogram + quick stats
        st.markdown(f"### 🔢 Distribution — `{col_to_explore}`")
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) == 0:
            st.caption("**None** — no numeric values.")
        else:
            bins = st.slider("Number of bins", 10, 100, 40, 5, key="bins_"+col_to_explore)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(clean, bins=bins, color=COLOR_PRIMARY)
            ax.set_xlabel(col_to_explore)
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram of {col_to_explore}")
            ax.grid(True, linestyle="--", alpha=0.35)
            st.pyplot(fig)
            st.caption(
                f"**count**: {clean.count():,} • **mean**: {clean.mean():,.2f} • "
                f"**std**: {clean.std():,.2f} • **min**: {clean.min():,.2f} • "
                f"**median**: {clean.median():,.2f} • **max**: {clean.max():,.2f}"
            )

    else:
        # CATEGORICAL → Top-N by revenue (if available) else by count
        st.markdown(f"### 🏷️ Top categories — `{col_to_explore}`")
        top_n = st.slider("Show top N", 5, 30, 15, 1, key="top_"+col_to_explore)

        rev_col_explorer = "__rev__" if "__rev__" in df.columns else ("revenue" if "revenue" in df.columns else None)
        if rev_col_explorer is not None and rev_col_explorer in df.columns:
            grouped = (
                df.groupby(col_to_explore)[rev_col_explorer]
                  .sum()
                  .sort_values(ascending=False)
                  .head(top_n)
                  .reset_index()
            )
            y_label = "Revenue"
            y_vals = grouped[rev_col_explorer]
        else:
            vc = df[col_to_explore].astype(str).value_counts().head(top_n)
            grouped = vc.rename_axis(col_to_explore).reset_index(name="count")
            y_label = "Count"
            y_vals = grouped["count"]

        if len(grouped) == 0:
            st.caption("**None** — no values to display.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(grouped[col_to_explore].astype(str), y_vals, color=COLOR_PRIMARY)
            ax.set_xlabel(col_to_explore)
            ax.set_ylabel(y_label)
            ax.set_title(f"Top {top_n} categories in {col_to_explore}")
            plt.xticks(rotation=45, ha="right")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            st.pyplot(fig)
