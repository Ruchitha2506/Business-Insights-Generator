
import os
import io
import pandas as pd
import numpy as np
import streamlit as st

# Optional: use OpenAI for narrative insights if API key is set
USE_LLM = True if os.getenv("OPENAI_API_KEY") else False

st.set_page_config(page_title="Business Insights Generator", page_icon="📊", layout="wide")
st.title("📊 Business Insights Generator")
st.write("Upload a CSV (e.g., sales, marketing, finance) and get automated KPIs + executive narrative insights.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
sample_btn = st.button("Use bundled sample dataset")

def load_data(file):
    """
    Robust CSV loader:
    - Parses 'date' if present
    - Coerces key numeric columns (revenue, unit_price, quantity, discount)
    - Handles thousand separators and string numbers
    """
    # Peek at columns first (without consuming the file)
    preview = pd.read_csv(file, nrows=1)
    parse_dates = ["date"] if "date" in preview.columns else None

    # If file-like object (uploaded), rewind it before the real read
    if hasattr(file, "seek"):
        file.seek(0)

    # Read full CSV; allow thousands separators
    df = pd.read_csv(file, parse_dates=parse_dates, thousands=",")

    # Coerce likely numeric columns
    for col in ["revenue", "unit_price", "quantity", "discount"]:
        if col in df.columns:
            # remove commas then coerce
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False)
                .replace({"": None})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize object columns to string (safe)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)

    return df


# ----- KPI + basic analytics -----
def compute_kpis(df):
    kpis = {}

    # Detect possible numeric columns
    colnames = [c.lower() for c in df.columns]

    # Try to find something like "price" or "unit price"
    price_col = next((c for c in df.columns if "price" in c.lower()), None)

    # If we have both quantity & price, calculate revenue
    if "quantity" in colnames and price_col:
        rev = pd.to_numeric(df[price_col], errors="coerce") * pd.to_numeric(df["quantity"], errors="coerce")
    elif price_col:
        # No quantity? Just use price as total value proxy
        rev = pd.to_numeric(df[price_col], errors="coerce")
    else:
        rev = None

    if rev is not None:
        rev = rev.fillna(0)
        df["__rev__"] = rev
        kpis["Total Revenue"] = float(rev.sum())
        kpis["Avg Value"] = float(rev.mean())
    else:
        kpis["Total Rows"] = len(df)

    # Month-over-month growth if 'date' column exists
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



def summarize_for_prompt(df, kpis, max_rows=20):
    """Keep the AI context compact."""
    sample = df.head(max_rows).to_dict(orient="records")
    context = {
        "basic_columns": list(df.columns),
        "row_count": int(len(df)),
        "kpis": kpis,
        "sample_records": sample
    }
    return context


def generate_llm_insights(context):
    # Using OpenAI if OPENAI_API_KEY is set, else fallback
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
            messages=[{"role":"user","content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating LLM insights: {e}"

# Main flow
data_path = None
if sample_btn:
    data_path = os.path.join(os.path.dirname(__file__), "sample_sales.csv")
elif uploaded is not None:
    data_path = uploaded

if data_path:
    df = load_data(data_path)
    st.subheader("Preview")
    st.dataframe(df.head(15))

    kpis = compute_kpis(df)
    st.subheader("KPIs")
    kpi_cols = st.columns(len(kpis) if len(kpis)>0 else 1)
    for i, (k,v) in enumerate(kpis.items()):
        with kpi_cols[i]:
            st.metric(k, f"{v:,.2f}" if isinstance(v, float) else v)

    # Optional charts (matplotlib)
    if "date" in df.columns and "revenue" in df.columns:
        import matplotlib.pyplot as plt
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        rev_by_month = df2.groupby(df2["date"].dt.to_period("M"))["revenue"].sum().reset_index()
        rev_by_month["date"] = rev_by_month["date"].astype(str)

        st.subheader("Revenue by Month")
        fig = plt.figure()
        plt.plot(rev_by_month["date"], rev_by_month["revenue"])
        plt.title("Revenue by Month")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        st.pyplot(fig)

    # LLM narrative
    st.subheader("Executive Narrative (LLM)")
    context = summarize_for_prompt(df, kpis)
    insights = generate_llm_insights(context)
    st.write(insights)

    # Download executive summary
    st.download_button("Download Summary", data=insights, file_name="executive_summary.txt")
else:
    st.info("Upload a CSV or click 'Use bundled sample dataset' to begin.")
