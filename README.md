# Business Insights Generator

An friendly and interactive **Streamlit** app that turns raw CSVs into clear **business KPIs, charts, and AI executive summaries**.  
Works with many kinds of datasets (sales, ecommerce, marketing, finance) and gracefully handles missing columns.

**Live demo:** _add your Streamlit Cloud URL here_  
**Repo:** https://github.com/Ruchitha2506/Business-Insights-Generator

---

## Features

- **CSV Upload or Sample Data** – Instantly preview data and compute KPIs  
- **Robust KPIs** – Auto-detects `revenue` or computes it from `unit_price * quantity * (1 - discount)`  
- **Resilient Types** – Coerces numeric columns (handles strings like `1,234.56`) and parses dates safely  
- **GenAI Executive Summary** (optional) – Uses OpenAI to produce narrative insights  
- **Clean Visual Theme** – Consistent, portfolio-ready colors  
- **NA Handling** – Every visualization shows its title and **“None — not applicable”** if a dataset lacks needed columns  
- **Smart Data Explorer** – A dataset-agnostic explorer (numeric, categorical, or datetime)

---

## KPIs (auto-computed)

The app will compute the best available metrics given your columns:

- **Total Revenue** & **Avg Order Value**  
  - Uses `revenue` if present  
  - Else computes from `unit_price * quantity * (1 - discount)`; scales `discount` if given as `%`  
- **Top Category / Top Product** (if `category` / `product` present)  
- **Latest MoM Growth %** (if `date` is present)

If none of the columns exist, it falls back to **Total Rows**.

---

## 📈 Visualizations

Each block shows its name and, if not applicable, **“None — not applicable for this dataset.”**

1. **💰 Total Revenue by Category (Top 10)**  
   _Needs_: `revenue` (or computed revenue) + `category`

2. **🌎 Total Revenue by Region**  
   _Needs_: `revenue` (or computed revenue) + `region`

3. **📈 Monthly Revenue Trend**  
   _Needs_: `revenue` (or computed) + `date`  
   Notes: Converts month to timestamps to avoid dtype issues.

4. **🧭 Data Explorer (dataset-agnostic)**
   - **Numeric** → Histogram + quick stats  
   - **Datetime** → Records over **Day** or **Month**  
   - **Categorical** → Top-N bar chart (by **Revenue** if available, else by **Count**)  
   _Always available_ (works with any single selected column)

**Theme Colors**
- Primary: `#4A90E2` (blue)  
- Accent: `#50E3C2` (teal)  
- Warn:   `#F5A623` (orange)

## 🧰 Tech Stack

- **Python**, **Streamlit**, **Pandas**, **Matplotlib**  
- **OpenAI API** (optional) for executive narrative


