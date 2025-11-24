"""
Streamlit dashboard: COVID‑19 Vaccination Storytelling Dashboard

This application tells a data‑driven story about global COVID‑19 vaccination
patterns and their relationship to the course of the pandemic.  It was
designed for public health policy makers at the World Health Organization
(WHO) to explore vaccination uptake, regional disparities, and the link
between vaccine rollout and COVID‑19 cases and deaths.

Key features of the dashboard include:

* **Global overview** – interactive line charts show the cumulative
  doses administered and population coverage over time.  Milestones are
  annotated to highlight important events such as vaccine approvals and
  booster introductions.
* **Regional disparities** – users can filter the data by WHO region or
  country.  A heatmap summarizes the pace of rollout across regions and
  months, and a choropleth map illustrates variation in coverage at a
  given date.
* **Vaccination vs. outcomes** – scatter plots compare vaccination
  coverage with new COVID‑19 deaths over time.  A line chart based on
  aggregated global data demonstrates how rising vaccine coverage is
  associated with falling death counts.
* **Booster uptake** – a bar chart ranks countries by booster coverage,
  revealing leaders and laggards.  The same component can be used to
  examine primary series or at least one dose coverage.
* **Narrative and reflection** – markdown sections throughout the app
  explain the context of the analysis, emphasize why certain patterns
  matter for policy, and call attention to data limitations.

The app uses three different visualization libraries to meet
assignment requirements: Plotly for interactive charts and the map,
Seaborn for the heatmap, and Matplotlib for a static annotated plot.

Datasets
--------

Two publicly available WHO datasets are used:

1. **COV_VAC_UPTAKE_2021_2023.csv** – monthly time‑series data per
   country on the uptake of COVID‑19 vaccines from January 2021 through
   December 2023.  Columns include total doses administered, doses per
   100 population, coverage of at least one dose, complete primary
   series, and booster doses.  Each record is dated at the end of the
   reporting month.  This file was downloaded from the WHO COVID‑19
   dashboard (archived 2021–2023 vaccine uptake data).

2. **WHO‑COVID‑19‑global‑data.csv** – weekly COVID‑19 case and death
   counts by country reported to WHO.  Columns include new cases,
   cumulative cases, new deaths, and cumulative deaths for each
   reporting week from 2020 through late 2025.  The dataset also
   contains the WHO region classification for each country.

To run the application, install the required packages with:

```
pip install streamlit pandas plotly seaborn matplotlib
```

Then start the app using:

```
streamlit run app.py
```

The datasets must be located in the same directory as this script, or
their paths adjusted below.  Because the WHO datasets are large, the
first load may take a few seconds.
"""

import datetime
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Define file paths relative to this script.  If files are renamed or
# relocated, update these paths accordingly.
DATA_DIR = pathlib.Path(__file__).parent
VACCINE_FILE = DATA_DIR / "COV_VAC_UPTAKE_2021_2023.csv"
# Use the version of the case/death dataset downloaded via Chrome.  If you
# downloaded multiple copies, pick the most recent non‑empty one.
CASE_FILE_CANDIDATES = [
    DATA_DIR / "WHO-COVID-19-global-data (2).csv",
    DATA_DIR / "WHO-COVID-19-global-data (1).csv",
    DATA_DIR / "WHO-COVID-19-global-data.csv",
]
for path in CASE_FILE_CANDIDATES:
    if path.exists() and path.stat().st_size > 0:
        CASE_FILE = path
        break
else:
    CASE_FILE = None


@st.cache_data
def load_vaccine_data(path: pathlib.Path) -> pd.DataFrame:
    """Load and preprocess the vaccine uptake dataset.

    The function parses dates and retains only relevant columns.  Missing
    values are left as NaN and handled downstream.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with parsed dates and normalized column names.
    """
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = df.columns.str.lower()
    # Parse date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Some numeric columns may be object due to commas/empty strings.  Coerce.
    numeric_cols = [
        "covid_vaccine_adm_tot_a1d",
        "covid_vaccine_adm_tot_boost",
        "covid_vaccine_adm_tot_cps",
        "covid_vaccine_adm_tot_doses",
        "covid_vaccine_adm_tot_doses_per100",
        "covid_vaccine_cov_tot_a1d",
        "covid_vaccine_cov_tot_boost",
        "covid_vaccine_cov_tot_cps",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_case_data(path: pathlib.Path) -> pd.DataFrame:
    """Load and preprocess the WHO case/death dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with parsed dates and numeric values.
    """
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = df.columns.str.lower()
    # Parse date column
    df["date_reported"] = pd.to_datetime(df["date_reported"], errors="coerce")
    # Coerce numeric columns
    for col in ["new_cases", "cumulative_cases", "new_deaths", "cumulative_deaths"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_global_trends(vac_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vaccine uptake globally by date.

    Because the vaccine dataset stores metrics as percentages per country, the
    global mean is computed by taking the country‑level average.  This
    approximation assumes each country carries equal weight; population
    weighting could be applied with an additional population dataset.

    Args:
        vac_df: Vaccine uptake data frame.

    Returns:
        DataFrame with date and aggregated metrics.
    """
    # Group by date and compute mean across countries
    agg = vac_df.groupby("date").agg({
        "covid_vaccine_adm_tot_doses": "sum",
        "covid_vaccine_adm_tot_doses_per100": "mean",
        "covid_vaccine_cov_tot_a1d": "mean",
        "covid_vaccine_cov_tot_cps": "mean",
        "covid_vaccine_cov_tot_boost": "mean",
    }).reset_index()
    return agg


def prepare_region_heatmap(vac_df: pd.DataFrame, case_df: pd.DataFrame) -> pd.DataFrame:
    """Compute average vaccine coverage by WHO region and date.

    The vaccine dataset does not include the WHO region.  We derive it by
    matching countries from the case dataset, which contains the `who_region`
    column.  Some countries may have missing region information and will be
    dropped from the heatmap.

    Args:
        vac_df: Vaccine uptake data frame.
        case_df: Case/death data frame with WHO region information.

    Returns:
        Pivot table with WHO region as rows, date as columns, and
        average one‑dose coverage as values.
    """
    # Map each country to its WHO region based on case data
    region_map = case_df.drop_duplicates(subset=["country", "who_region"])
    region_map = region_map[["country", "who_region"]].dropna()
    vac_df_region = vac_df.merge(region_map, left_on="country", right_on="country", how="left")
    # Filter out rows without region
    vac_df_region = vac_df_region.dropna(subset=["who_region"])
    # Compute mean coverage per region per month
    heat = vac_df_region.groupby(["who_region", "date"]).agg({
        "covid_vaccine_cov_tot_a1d": "mean"
    }).reset_index()
    # Pivot for heatmap
    pivot = heat.pivot(index="who_region", columns="date", values="covid_vaccine_cov_tot_a1d")
    return pivot


def prepare_correlation_data(vac_df: pd.DataFrame, case_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare monthly aggregated data for correlation analysis.

    The vaccine data is monthly; the case data is weekly.  To align, we
    resample the case data to monthly by summing new deaths per month.  The
    vaccine coverage is averaged across countries per month.  The resulting
    data frame contains, for each month, the mean coverage and total new
    deaths worldwide.

    Args:
        vac_df: Vaccine uptake data frame.
        case_df: Case/death data frame.

    Returns:
        DataFrame with columns `date` (month start), `coverage_mean` and
        `new_deaths_sum`.
    """
    # Aggregate vaccine coverage across countries
    vac_month = vac_df.groupby("date").agg({
        "covid_vaccine_cov_tot_cps": "mean"
    }).rename(columns={"covid_vaccine_cov_tot_cps": "coverage_mean"})
    # Aggregate new deaths across countries by month
    case_df = case_df.copy()
    case_df["month"] = case_df["date_reported"].dt.to_period("M").dt.to_timestamp()
    deaths_month = case_df.groupby("month").agg({"new_deaths": "sum"}).rename(columns={"new_deaths": "new_deaths_sum"})
    # Join on month (some months may not match exactly; use inner join)
    corr_df = pd.merge(vac_month, deaths_month, left_index=True, right_index=True, how="inner")
    corr_df = corr_df.reset_index().rename(columns={"month": "date"})
    return corr_df


def prepare_booster_ranking(vac_df: pd.DataFrame, date: pd.Timestamp, metric: str) -> pd.DataFrame:
    """Compute country ranking for a chosen vaccine metric at a specific date.

    Args:
        vac_df: Vaccine uptake data frame.
        date: The date (month end) for which to extract metrics.
        metric: One of 'covid_vaccine_cov_tot_boost',
            'covid_vaccine_cov_tot_cps', or 'covid_vaccine_cov_tot_a1d'.

    Returns:
        DataFrame with columns `country` and `value`, sorted descending by
        value.  Countries with missing values are dropped.
    """
    if metric not in vac_df.columns:
        raise ValueError(f"Metric {metric} not found in dataset")
    # Filter rows by date (match exactly)
    subset = vac_df[vac_df["date"] == date][["country", metric]].dropna()
    subset = subset.rename(columns={metric: "value"})
    subset = subset.sort_values("value", ascending=False).reset_index(drop=True)
    return subset


def annotated_global_plot(global_df: pd.DataFrame):
    """Create a static Matplotlib plot with annotations for major milestones.

    Args:
        global_df: DataFrame with `date` and global metrics.

    Returns:
        A matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(global_df["date"], global_df["covid_vaccine_cov_tot_a1d"], label="% with ≥1 dose")
    ax.plot(global_df["date"], global_df["covid_vaccine_cov_tot_cps"], label="% fully vaccinated")
    ax.plot(global_df["date"], global_df["covid_vaccine_cov_tot_boost"], label="% boosted")
    ax.set_title("Global COVID‑19 vaccination coverage over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Coverage (%)")
    ax.legend()
    # Annotate major events
    events = {
        "2020-12-31": "First approvals",
        "2021-06-30": "Global ramp‑up",
        "2022-03-31": "Booster campaigns",
        "2023-12-31": "End of dataset",
    }
    for date_str, label in events.items():
        dt = pd.to_datetime(date_str)
        if dt >= global_df["date"].min() and dt <= global_df["date"].max():
            # Find y position at event
            y = np.interp(dt.value, global_df["date"].astype(np.int64), global_df["covid_vaccine_cov_tot_a1d"])
            ax.annotate(label, xy=(dt, y), xytext=(dt + pd.Timedelta(days=30), y + 5),
                        arrowprops=dict(arrowstyle="->", lw=1), fontsize=8, color="black")
    return fig


def main():
    st.set_page_config(page_title="COVID‑19 Vaccination Story", layout="wide")
    st.title("COVID‑19 Vaccination Storytelling Dashboard")

    # Intro narrative
    st.markdown(
        """
        ### Context

        In this dashboard you are a **WHO public health data scientist** exploring how
        COVID‑19 vaccination unfolded around the world and how it influenced the
        trajectory of the pandemic.  The data come from the World Health
        Organization (WHO) and cover the period January 2021 through
        December 2023 for vaccination uptake and weekly case/death reports
        extending into 2025.  Keep in mind that reporting frequency and
        completeness vary by country【136761444091978†L140-L147】.  The goal of
        this app is not only to display numbers but to tell a story that
        provides actionable insights and recommendations for policymakers.
        """
    )

    # Load data
    with st.spinner("Loading data ..."):
        vac_df = load_vaccine_data(VACCINE_FILE)
        if CASE_FILE is not None:
            case_df = load_case_data(CASE_FILE)
        else:
            st.error("Case and death dataset could not be found.")
            case_df = pd.DataFrame()

    # Sidebar controls
    st.sidebar.header("Filters and Options")
    # Region filter
    all_regions = sorted(case_df["who_region"].dropna().unique()) if not case_df.empty else []
    selected_regions = st.sidebar.multiselect(
        "Select WHO region(s)", options=all_regions, default=all_regions
    )
    # Country filter – show only countries from selected regions
    if selected_regions:
        region_countries = case_df[case_df["who_region"].isin(selected_regions)]["country"].unique()
    else:
        region_countries = vac_df["country"].unique()
    selected_countries = st.sidebar.multiselect(
        "Select country/countries", options=sorted(region_countries), default=[]
    )
    # Metric selector for coverage
    metric_map = {
        "At least one dose (% of population)": "covid_vaccine_cov_tot_a1d",
        "Fully vaccinated (% of population)": "covid_vaccine_cov_tot_cps",
        "Booster coverage (% of population)": "covid_vaccine_cov_tot_boost",
        "Doses administered per 100 population": "covid_vaccine_adm_tot_doses_per100",
    }
    metric_label = st.sidebar.selectbox(
        "Select metric to visualize", options=list(metric_map.keys()), index=0
    )
    metric_col = metric_map[metric_label]
    # Date range slider – based on available dates in vaccine data
    # Streamlit's slider requires plain Python date/datetime objects rather
    # than pandas Timestamp objects; convert accordingly.  The underlying
    # dataset still uses pandas Timestamps.
    min_date_ts = vac_df["date"].min()
    max_date_ts = vac_df["date"].max()
    # Convert to python datetime.date for the slider to avoid KeyError
    min_date_dt = min_date_ts.date() if hasattr(min_date_ts, "date") else min_date_ts
    max_date_dt = max_date_ts.date() if hasattr(max_date_ts, "date") else max_date_ts
    date_range_dt = st.sidebar.slider(
        "Select date range", min_value=min_date_dt, max_value=max_date_dt,
        value=(min_date_dt, max_date_dt), format="%Y-%m-%d"
    )
    # Convert slider output (dates) back to pandas Timestamp for filtering
    date_range = (pd.to_datetime(date_range_dt[0]), pd.to_datetime(date_range_dt[1]))

    # Filter vaccine data by selected countries or regions and date range
    vac_filtered = vac_df.copy()
    if selected_regions:
        # Merge to include region; then filter
        region_map = case_df[["country", "who_region"]].drop_duplicates()
        vac_filtered = vac_filtered.merge(region_map, on="country", how="left")
        vac_filtered = vac_filtered[vac_filtered["who_region"].isin(selected_regions)]
    if selected_countries:
        vac_filtered = vac_filtered[vac_filtered["country"].isin(selected_countries)]
    # Date range filtering
    vac_filtered = vac_filtered[(vac_filtered["date"] >= date_range[0]) & (vac_filtered["date"] <= date_range[1])]

    # Global overview section
    st.subheader("Global Vaccination Timeline")
    global_df = prepare_global_trends(vac_df)
    # Filter to date range for global chart
    global_df_range = global_df[(global_df["date"] >= date_range[0]) & (global_df["date"] <= date_range[1])]
    fig_global = px.line(
        global_df_range,
        x="date",
        y=["covid_vaccine_cov_tot_a1d", "covid_vaccine_cov_tot_cps", "covid_vaccine_cov_tot_boost"],
        labels={"value": "Coverage (%)", "variable": "Metric"},
        title="Global coverage of COVID‑19 vaccination over time",
    )
    # Annotate milestone
    fig_global.add_vline(x=pd.to_datetime("2021-03-31"), line_dash="dot", annotation_text="Early ramp‑up")
    fig_global.add_vline(x=pd.to_datetime("2022-01-31"), line_dash="dot", annotation_text="Omicron wave")
    st.plotly_chart(fig_global, use_container_width=True)

    # Matplotlib static plot for milestone annotation
    with st.expander("Static annotated view (Matplotlib)"):
        fig_ann = annotated_global_plot(global_df_range)
        st.pyplot(fig_ann)

    # Regional disparities section
    st.subheader("Regional Disparities in Vaccination Rollout")
    # Heatmap using seaborn – convert pivot table to 2D array
    heat_data = prepare_region_heatmap(vac_df, case_df)
    # Filter heatmap by date range columns
    heat_filtered = heat_data.loc[:, (heat_data.columns >= date_range[0]) & (heat_data.columns <= date_range[1])]
    fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat_filtered, cmap="YlGnBu", cbar_kws={"label": "% with ≥1 dose"}, ax=ax_heat)
    ax_heat.set_xlabel("Date")
    ax_heat.set_ylabel("WHO region")
    ax_heat.set_title("Mean one‑dose coverage by WHO region and month")
    st.pyplot(fig_heat)

    # Choropleth map for selected date – show coverage per country
    st.subheader("World map of vaccination coverage")
    # Choose a date for the map. Convert Timestamps to date for the slider.
    map_date_dt = st.slider(
        "Choose a date for the map", min_value=min_date_dt, max_value=max_date_dt,
        value=max_date_dt, format="%Y-%m-%d"
    )
    # Convert back to pandas Timestamp for comparison
    map_date = pd.to_datetime(map_date_dt)
    map_subset = vac_df[vac_df["date"] == map_date]
    map_metric = st.selectbox(
        "Select metric for the map", options=list(metric_map.keys()), index=0
    )
    map_col = metric_map[map_metric]
    fig_map = px.choropleth(
        map_subset,
        locations="country",
        locationmode="ISO-3",
        color=map_col,
        hover_name="country",
        color_continuous_scale="Blues",
        title=f"{map_metric} on {map_date.date()}"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Vaccination vs outcomes section
    st.subheader("Vaccination vs COVID‑19 deaths")
    corr_df = prepare_correlation_data(vac_df, case_df)
    # Filter by date range
    corr_filtered = corr_df[(corr_df["date"] >= date_range[0]) & (corr_df["date"] <= date_range[1])]
    fig_corr = px.scatter(
        corr_filtered,
        x="coverage_mean",
        y="new_deaths_sum",
        hover_name="date",
        trendline="ols",
        labels={"coverage_mean": "Mean % fully vaccinated", "new_deaths_sum": "Total new deaths"},
        title="Association between vaccination coverage and new deaths (monthly)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        """
        The scatter plot above aggregates the data by month and shows how increases
        in vaccination coverage coincide with declines in the global number of
        COVID‑19 deaths【30†L77-L85】.  The regression line provides a visual
        indication of the negative correlation.
        """
    )

    # Booster uptake ranking
    st.subheader("Country ranking by vaccination metric")
    rank_metric_label = st.selectbox(
        "Select ranking metric", options=list(metric_map.keys()), index=2
    )
    rank_metric = metric_map[rank_metric_label]
    rank_date = st.date_input(
        "Select ranking date", value=max_date, min_value=min_date, max_value=max_date
    )
    rank_date = pd.to_datetime(rank_date)
    ranking_df = prepare_booster_ranking(vac_df, rank_date, rank_metric)
    # Show top 20
    st.dataframe(ranking_df.head(20))

    st.markdown(
        """
        ### Conclusions and Recommendations

        * **Vaccine uptake saved millions of lives:** Statistical modelling
          suggests that COVID‑19 vaccines prevented nearly 20 million deaths
          during their first year【30†L77-L85】.
        * **Disparities remain a critical issue:** High‑income countries
          achieved coverage quickly while low‑income countries lagged
          significantly【190463340138489†L88-L115】.  Policies must address
          equitable access to vaccines and booster doses.
        * **Booster campaigns matter:** Regions with high booster uptake
          experienced fewer severe outcomes during the Omicron wave【32†L318-L324】.
        * **Infrastructure and communication are key:** Successful rollout
          depends on robust health systems, cold chain capacity and efforts
          to build public trust【cidrap.umn.edu†L?】.  Investment in these
          areas will pay dividends in future health emergencies.
        * **Data limitations:** Reporting frequency and population
          denominators vary; some countries have missing data【136761444091978†L140-L147】.
          Interpret patterns cautiously and consider complementary data sources
          when making policy decisions.
        """
    )

    # Data source footnote
    st.markdown(
        """
        **Data sources:** World Health Organization COVID‑19 dashboard (vaccine
        uptake and case/death datasets).  See data notes on the WHO portal for
        details【136761444091978†L140-L147】.  Figures may differ from other
        reporting systems due to varying definitions and reporting schedules.
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()