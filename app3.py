import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

def human_format(num: float) -> str:
    """Convert a large number to a human‑readable string with K, M, B, T suffixes.

    Parameters
    ----------
    num : float
        The number to format.

    Returns
    -------
    str
        Formatted string with appropriate suffix.
    """
    if num is None or np.isnan(num):
        return "N/A"
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 4:
        num /= 1000.0
        magnitude += 1
    suffixes = ['', 'K', 'M', 'B', 'T']
    return f"{num:.1f}{suffixes[magnitude]}" if magnitude else f"{int(num):,}"


def load_data(path: str) -> pd.DataFrame:
    """Load the WHO COVID‑19 dataset from a CSV file.

    The dataset is expected to contain the following columns:
    Date_reported, Country_code, Country, WHO_region,
    New_cases, Cumulative_cases, New_deaths, Cumulative_deaths.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with parsed dates and missing values handled.
    """
    df = pd.read_csv(path)
    # Parse the date column and sort
    df['Date_reported'] = pd.to_datetime(df['Date_reported'], errors='coerce')
    df = df.sort_values('Date_reported')
    # Replace missing values (NaN) in new cases and deaths with zeros
    df['New_cases'] = df['New_cases'].fillna(0)
    df['New_deaths'] = df['New_deaths'].fillna(0)
    # Ensure cumulative columns are numeric; fill missing with zeros
    df['Cumulative_cases'] = pd.to_numeric(df['Cumulative_cases'], errors='coerce').fillna(0)
    df['Cumulative_deaths'] = pd.to_numeric(df['Cumulative_deaths'], errors='coerce').fillna(0)
    return df


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load and cache the global COVID‑19 data."""
    path = "./WHO-COVID-19-global-data.csv"
    return load_data(path)


def main():
    # Page configuration
    st.set_page_config(
        page_title="COVID‑19 Pandemic Insights Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title and description
    st.title("COVID‑19 Pandemic Insights Dashboard")

    # Reflect your role, stakeholders and objective at the very top to meet the activity criteria
    st.markdown(
        """
        **Role:** Public Health Data Analyst at Pan American Health Organization (PAHO)  

        **Stakeholders:** Health ministers and regional coordinators in the Americas  

        **Objective:** Identify when and where COVID‑19 risk was highest and what patterns can help prepare for future waves.
        """
    )

    # Big Idea statement
    st.markdown(
        """
        **Big Idea:** A small number of countries and regions concentrated the majority of COVID‑19 cases and deaths at specific periods in time. By focusing on these hotspots and waves, health authorities can better plan resources and prepare for future outbreaks.
        """
    )

    st.markdown(
        """
        ### Understanding the spread and impact of COVID‑19

        This dashboard presents key metrics and visualisations derived from the World Health Organization's
        **COVID‑19 Global Data** dataset. Use the sidebar to filter by time period, region and country. Each tab below
        is designed as a chapter of the story: **Overview** shows the global timeline and overall burden, **Regional inequalities** highlights
        disparities between regions and countries, **Country deep dive** lets you compare individual country trajectories, and **Seasonality & preparedness**
        explores how the pandemic waxed and waned over months and years.
        """
    )

    # Load data
    df = get_dataset()

    # Sidebar filters
    st.sidebar.header("Filters")

    # --- Date range selection with quick presets and slider ---
    # Convert the min and max reported dates to plain date objects
    min_date = df['Date_reported'].min().date()
    max_date = df['Date_reported'].max().date()

    # Define preset periods for quick selection
    presets = {
        'Full pandemic': (min_date, max_date),
        'First wave (2020)': (date(2020, 1, 1), date(2020, 12, 31)),
        'Pre‑vaccine (2020–2021)': (date(2020, 1, 1), date(2021, 12, 31)),
        'Delta wave (mid‑2021)': (date(2021, 6, 1), date(2021, 10, 31)),
        'Omicron wave (late‑2021 to early‑2022)': (date(2021, 11, 1), date(2022, 3, 31)),
        'Post‑vaccine (2022–2023)': (date(2022, 1, 1), min(date(2023, 12, 31), max_date))
    }

    preset_options = ['Custom range'] + list(presets.keys())
    selected_preset = st.sidebar.selectbox("Time period preset", preset_options, index=0)
    # Determine the initial slider range based on preset selection
    if selected_preset != 'Custom range':
        preset_start, preset_end = presets[selected_preset]
        # Ensure the preset falls within the available data range
        preset_start = max(min_date, preset_start)
        preset_end = min(max_date, preset_end)
        slider_default = (preset_start, preset_end)
    else:
        slider_default = (min_date, max_date)

    # Create date range slider; if the user has selected a preset, the slider
    # defaults to that range but can still be adjusted manually
    date_range = st.sidebar.slider(
        "Select reporting period",
        min_value=min_date,
        max_value=max_date,
        value=slider_default,
        format="YYYY-MM-DD"
    )
    # Normalise to a two‑element tuple
    if isinstance(date_range, date):
        start_date = date_range
        end_date = date_range
    else:
        start_date, end_date = date_range

    # Region selection
    regions = ["All"] + sorted(df['WHO_region'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Select WHO region", regions)

    # Country selection based on region
    if selected_region != "All":
        countries = df[df['WHO_region'] == selected_region]['Country'].dropna().unique()
    else:
        countries = df['Country'].dropna().unique()
    countries = sorted(countries)

    selected_countries = st.sidebar.multiselect(
        "Select countries (leave empty for all)", countries
    )

    # Apply filters to dataset
    mask = (df['Date_reported'] >= pd.to_datetime(start_date)) & (df['Date_reported'] <= pd.to_datetime(end_date))
    if selected_region != "All":
        mask &= df['WHO_region'] == selected_region
    if selected_countries:
        mask &= df['Country'].isin(selected_countries)
    filtered_df = df.loc[mask]

    # Display summary metrics
    # Safely compute metrics even when NaN values exist
    total_new_cases = filtered_df['New_cases'].fillna(0).sum()
    total_new_deaths = filtered_df['New_deaths'].fillna(0).sum()
    if selected_countries:
        total_cum_cases = filtered_df.groupby('Country')['Cumulative_cases'].max().fillna(0).sum()
        total_cum_deaths = filtered_df.groupby('Country')['Cumulative_deaths'].max().fillna(0).sum()
    else:
        total_cum_cases = filtered_df['Cumulative_cases'].fillna(0).max()
        total_cum_deaths = filtered_df['Cumulative_deaths'].fillna(0).max()
    # Calculate peak new cases and deaths within the period
    peak_new_cases = filtered_df['New_cases'].fillna(0).max()
    peak_new_deaths = filtered_df['New_deaths'].fillna(0).max()
    # Case fatality ratio
    cfr = (total_cum_deaths / total_cum_cases * 100) if total_cum_cases > 0 else np.nan

    st.subheader("Key figures for the selected period")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total new cases", human_format(total_new_cases))
    col2.metric("Total new deaths", human_format(total_new_deaths))
    col3.metric("Peak new cases", human_format(peak_new_cases))
    col4.metric("Peak new deaths", human_format(peak_new_deaths))
    col5.metric("Max cumulative cases", human_format(total_cum_cases))
    col6.metric("Max cumulative deaths", human_format(total_cum_deaths))
    # Show case fatality ratio as text below metrics
    st.caption(f"Case fatality ratio (CFR): {cfr:.2f}%" if not np.isnan(cfr) else "Case fatality ratio (CFR): N/A")

    # Tabs for different sections following a narrative flow: Overview → Regional inequalities → Country deep dive → Seasonality
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    tab_overview, tab_regional, tab_country, tab_seasonality = st.tabs([
        "Overview", "Regional inequalities", "Country deep dive", "Seasonality & preparedness"
    ])
    # Overview tab
    with tab_overview:
        # Guiding text for the overview tab
        st.markdown(
            """
            **What this shows:** Daily new cases and deaths aggregated across your selected region and countries, along with 7‑day moving averages and the resulting case fatality ratio over time.  

            **Why it matters:** Understanding when and how quickly infections and deaths rose reveals the timing of each wave and shows how deaths lag behind cases.  

            **What to look at:** Peaks indicate waves; compare the height and timing of peaks between cases and deaths. The moving averages smooth reporting noise. A declining CFR over time suggests improved detection and treatment.
            """
        )

        # Aggregate new cases and deaths by date for the selected filters
        time_series = filtered_df.groupby('Date_reported').agg({
            'New_cases': 'sum',
            'New_deaths': 'sum'
        }).reset_index().sort_values('Date_reported')
        # Compute 7‑day moving averages to smooth out short‑term fluctuations
        time_series['MA_cases'] = time_series['New_cases'].rolling(window=7, min_periods=1).mean()
        time_series['MA_deaths'] = time_series['New_deaths'].rolling(window=7, min_periods=1).mean()
        # Compute cumulative sums and CFR
        time_series['Cum_cases'] = time_series['New_cases'].cumsum()
        time_series['Cum_deaths'] = time_series['New_deaths'].cumsum()
        time_series['CFR'] = np.where(time_series['Cum_cases'] > 0, (time_series['Cum_deaths'] / time_series['Cum_cases']) * 100, 0)
        # Optional: compute weekly growth rate of cases (percentage change relative to 7 days prior)
        time_series['Growth_rate'] = time_series['New_cases'].pct_change(periods=7) * 100

        # Message‑style title summarising peaks (approx by taking top two dates)
        # Identify two highest peaks in new cases
        peaks = time_series.nlargest(2, 'New_cases')['Date_reported'].dt.strftime('%b %Y').tolist()
        if len(peaks) >= 2:
            trend_title = f"Global waves peaked around {peaks[0]} and {peaks[1]}"
        else:
            trend_title = "Epidemic curve"
        # Interactive line chart for new cases/deaths and moving averages
        fig_trend = px.line(
            time_series,
            x='Date_reported',
            y=['New_cases', 'New_deaths', 'MA_cases', 'MA_deaths'],
            labels={'value': 'Count', 'Date_reported': 'Date', 'variable': ''},
            title=trend_title,
            color_discrete_map={
                'New_cases': '#1f77b4',    # blue
                'New_deaths': '#d62728',   # red
                'MA_cases': '#aec7e8',     # light blue
                'MA_deaths': '#ff9896'    # light red
            }
        )
        fig_trend.for_each_trace(lambda t: t.update(name={
            'New_cases': 'New cases',
            'New_deaths': 'New deaths',
            'MA_cases': '7‑day MA (cases)',
            'MA_deaths': '7‑day MA (deaths)'
        }.get(t.name, t.name)))
        fig_trend.update_layout(legend_title_text='')
        st.plotly_chart(fig_trend, use_container_width=True)

        # CFR over time chart
        fig_cfr = px.line(
            time_series,
            x='Date_reported',
            y='CFR',
            labels={'CFR': 'CFR (%)', 'Date_reported': 'Date'},
            title='Case fatality ratio over time',
            color_discrete_sequence=['#9467bd']  # purple
        )
        fig_cfr.update_layout(legend_title_text='', showlegend=False)
        st.plotly_chart(fig_cfr, use_container_width=True)

        # Growth rate chart (optional) – show only if there are enough data points
        if len(time_series) > 14:
            fig_growth = px.line(
                time_series,
                x='Date_reported',
                y='Growth_rate',
                labels={'Growth_rate': 'Growth rate (%)', 'Date_reported': 'Date'},
                title='Weekly growth rate of new cases',
                color_discrete_sequence=['#2ca02c']  # green
            )
            fig_growth.add_hline(y=0, line_dash='dash', line_color='gray')
            st.plotly_chart(fig_growth, use_container_width=True)

    # Regional inequalities tab
    with tab_regional:
        st.markdown(
            """
            **What this shows:** The countries and regions that carried most of the burden of COVID‑19 in your selected period. We look at the top countries by cumulative cases and deaths on the latest reporting day, compare burden across WHO regions, and map where infections and deaths were highest relative to cases.  

            **Why it matters:** Identifying which countries and regions bore the greatest impact highlights inequalities and helps target resources and policy attention.  

            **What to look at:** Notice how a handful of countries account for the majority of cases and deaths; compare how different WHO regions stack up; and look for countries with high death percentages relative to cases as a signal of under‑testing or severe outbreaks.
            """
        )

        # Compute latest date within filtered data and prepare latest_df
        latest_date = filtered_df['Date_reported'].max()
        latest_df = filtered_df[filtered_df['Date_reported'] == latest_date].copy()
        latest_df['Cumulative_cases'] = latest_df['Cumulative_cases'].fillna(0)
        latest_df['Cumulative_deaths'] = latest_df['Cumulative_deaths'].fillna(0)

        # Top 10 countries by cumulative cases and deaths
        top_countries_cases = latest_df.nlargest(10, 'Cumulative_cases')[['Country', 'Cumulative_cases']]
        top_countries_deaths = latest_df.nlargest(10, 'Cumulative_deaths')[['Country', 'Cumulative_deaths']]

        # Share of global cases/deaths for the top 10
        total_cases_all = latest_df['Cumulative_cases'].sum()
        total_deaths_all = latest_df['Cumulative_deaths'].sum()
        share_cases = (top_countries_cases['Cumulative_cases'].sum() / total_cases_all * 100) if total_cases_all > 0 else 0
        share_deaths = (top_countries_deaths['Cumulative_deaths'].sum() / total_deaths_all * 100) if total_deaths_all > 0 else 0

        # Plot horizontal bar charts for top countries
        colA, colB = st.columns(2)
        fig_cases = px.bar(
            top_countries_cases.sort_values('Cumulative_cases'),
            x='Cumulative_cases', y='Country', orientation='h',
            title=f'Top 10 countries by cumulative cases – they account for {share_cases:.1f}% of cases',
            labels={'Cumulative_cases': 'Cumulative cases', 'Country': ''},
            color_discrete_sequence=['#1f77b4']
        )
        fig_cases.update_layout(yaxis=dict(categoryorder='total ascending'), showlegend=False)
        colA.plotly_chart(fig_cases, use_container_width=True)
        fig_deaths = px.bar(
            top_countries_deaths.sort_values('Cumulative_deaths'),
            x='Cumulative_deaths', y='Country', orientation='h',
            title=f'Top 10 countries by cumulative deaths – they account for {share_deaths:.1f}% of deaths',
            labels={'Cumulative_deaths': 'Cumulative deaths', 'Country': ''},
            color_discrete_sequence=['#d62728']
        )
        fig_deaths.update_layout(yaxis=dict(categoryorder='total ascending'), showlegend=False)
        colB.plotly_chart(fig_deaths, use_container_width=True)

        # Regional distribution bar chart
        region_agg = latest_df.groupby('WHO_region').agg({
            'Cumulative_cases': 'sum',
            'Cumulative_deaths': 'sum'
        }).reset_index()
        region_melt = region_agg.melt(id_vars='WHO_region', value_vars=['Cumulative_cases', 'Cumulative_deaths'],
                                      var_name='Metric', value_name='Count')
        fig_region = px.bar(
            region_melt,
            x='Count', y='WHO_region',
            color='Metric', orientation='h',
            labels={'WHO_region': 'WHO region', 'Count': 'Count', 'Metric': ''},
            title='Cumulative cases and deaths by region (latest day)',
            color_discrete_map={'Cumulative_cases': '#1f77b4', 'Cumulative_deaths': '#d62728'}
        )
        fig_region.update_layout(barmode='group', yaxis=dict(categoryorder='total ascending'))
        st.plotly_chart(fig_region, use_container_width=True)

        # Geographical distribution maps
        # Exclude pseudo-entity 'International commercial vessel'
        map_df = filtered_df[filtered_df['Country'] != 'International commercial vessel']
        agg = map_df.groupby('Country').agg({
            'New_cases': 'sum',
            'New_deaths': 'sum'
        }).reset_index()
        agg['Deaths_per_case'] = np.where(agg['New_cases'] > 0, (agg['New_deaths'] / agg['New_cases']) * 100, 0)
        fig_map_cases = px.choropleth(
            agg,
            locations='Country',
            locationmode='country names',
            color='New_cases',
            hover_name='Country',
            color_continuous_scale='Blues',
            title='Total reported cases by country',
        )
        fig_map_cases.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_map_cases, use_container_width=True)
        fig_map_ratio = px.choropleth(
            agg,
            locations='Country',
            locationmode='country names',
            color='Deaths_per_case',
            hover_name='Country',
            color_continuous_scale='Reds',
            title='Death percentage (deaths as % of cases)',
        )
        fig_map_ratio.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_map_ratio, use_container_width=True)

    # Country deep dive tab
    with tab_country:
        st.markdown(
            """
            **What this shows:** Daily new cases (and optionally deaths) for each of your selected countries.  

            **Why it matters:** Comparing individual country trajectories reveals how outbreaks evolved differently across nations and helps identify outliers.  

            **What to look at:** Notice the timing and height of peaks across countries. Selecting multiple countries allows you to compare the pace and scale of their epidemics. You can also observe how policies, demographics or healthcare capacity might influence these patterns.
            """
        )
        if not selected_countries:
            st.info("Please select one or more countries in the sidebar to explore their trajectories.")
        else:
            # Aggregate new cases and deaths by date and country
            country_ts = filtered_df.groupby(['Date_reported', 'Country']).agg({'New_cases': 'sum', 'New_deaths': 'sum'}).reset_index()
            # New cases chart
            fig_country_cases = px.line(
                country_ts,
                x='Date_reported',
                y='New_cases',
                color='Country',
                labels={'New_cases': 'New cases', 'Date_reported': 'Date', 'Country': 'Country'},
                title='Daily new cases by country',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_country_cases, use_container_width=True)
            # New deaths chart
            fig_country_deaths = px.line(
                country_ts,
                x='Date_reported',
                y='New_deaths',
                color='Country',
                labels={'New_deaths': 'New deaths', 'Date_reported': 'Date', 'Country': 'Country'},
                title='Daily new deaths by country',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_country_deaths, use_container_width=True)



    # Seasonality tab: heatmap
    with tab_seasonality:
        st.markdown(
            """
            **What this shows:** Monthly patterns of new cases and deaths over successive years. Each cell represents the total number of cases (or deaths) reported in a given month and year.  

            **Why it matters:** Seasonality can reveal whether certain months consistently see higher transmission, which can inform preparedness efforts and resource planning.  

            **What to look at:** Look for recurring darker cells (higher values) in the same months across years to identify possible seasonal effects. Compare patterns before and after vaccines became widely available.
            """
        )
        # Create year and month fields
        data = filtered_df.copy()
        data['Year'] = data['Date_reported'].dt.year
        data['Month'] = data['Date_reported'].dt.month
        monthly = data.groupby(['Year', 'Month']).agg({'New_cases': 'sum', 'New_deaths': 'sum'}).reset_index()
        # Pivot to year (rows) x month (columns) format for cases and deaths
        cases_pivot = monthly.pivot(index='Year', columns='Month', values='New_cases')
        deaths_pivot = monthly.pivot(index='Year', columns='Month', values='New_deaths')
        cases_pivot = cases_pivot.fillna(0)
        deaths_pivot = deaths_pivot.fillna(0)
        # Interactive heatmaps using Plotly
        fig_cases_heatmap = px.imshow(
            cases_pivot,
            labels=dict(x='Month', y='Year', color='New cases'),
            x=cases_pivot.columns,
            y=cases_pivot.index,
            color_continuous_scale='Blues',
            aspect='auto',
            title='Monthly new cases by year'
        )
        fig_deaths_heatmap = px.imshow(
            deaths_pivot,
            labels=dict(x='Month', y='Year', color='New deaths'),
            x=deaths_pivot.columns,
            y=deaths_pivot.index,
            color_continuous_scale='Reds',
            aspect='auto',
            title='Monthly new deaths by year'
        )
        col_cases, col_deaths = st.columns(2)
        col_cases.plotly_chart(fig_cases_heatmap, use_container_width=True)
        col_deaths.plotly_chart(fig_deaths_heatmap, use_container_width=True)

    # Conclusion panel with key findings, recommendations and limitations
    st.markdown("### Conclusion & recommendations")
    # Summarise key findings using dynamic values where possible
    key_findings = [
        f"Daily new cases and deaths show distinct waves with peaks around **{peaks[0]}** and **{peaks[1]}** (global view).",
        f"The **top 10 countries** account for roughly **{share_cases:.1f}%** of cumulative cases and **{share_deaths:.1f}%** of deaths in the latest reporting day, highlighting a highly unequal burden.",
        "Some countries exhibit **high death percentages relative to reported cases**, which may signal under‑detection of cases or particularly severe outbreaks.",
        "Seasonal patterns suggest that certain months repeatedly see higher transmission, indicating when preparedness efforts should ramp up."
    ]
    st.markdown("**Key findings:**")
    for finding in key_findings:
        st.markdown(f"- {finding}")

    recommendations = [
        "Focus surveillance, testing and resource allocation on the countries and regions with the highest burden (hotspots).",
        "Strengthen early detection and reporting systems to identify waves sooner and act quickly.",
        "Support vaccination and equitable access to healthcare to reduce case fatality ratios.",
        "Plan ahead for periods of increased transmission suggested by seasonal patterns and historical waves."
    ]
    st.markdown("**Recommendations:**")
    for rec in recommendations:
        st.markdown(f"- {rec}")

    limitations = [
        "Data represent official reports submitted to WHO and may lag behind real‑time events or miss cases due to under‑testing.",
        "Reporting practices vary across countries and over time, so comparisons should be interpreted with caution.",
        "Missing values are treated as zeros, which could underestimate true counts.",
        "Population differences are not accounted for; per‑capita metrics could be added in future work.",
        "Cumulative figures reflect the latest reported values within the selected time frame and may differ from global totals due to retrospective updates."
    ]
    st.markdown("**Limitations:**")
    for lim in limitations:
        st.markdown(f"- {lim}")

    # About section
    st.markdown("""
    ### About this dashboard
    This tool was built using [Streamlit](https://streamlit.io/) and leverages **Plotly** for interactive charts and **Matplotlib/Seaborn** for heatmaps.  
    It allows stakeholders to explore COVID‑19 case and death patterns across time, regions and countries. By providing interactive filters and multiple visual perspectives,
    the dashboard supports evidence‑based decision making for public health policy and resource allocation.  
    Updates and improvements incorporate storytelling principles, message‑style titles and user‑centred design to make the narrative clear and actionable.
    """)


if __name__ == "__main__":
    main()