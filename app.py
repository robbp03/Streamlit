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
    st.markdown(
        """
        ### Understanding the spread and impact of COVID‑19

        This dashboard presents key metrics and visualisations derived from the World Health Organization's
        **COVID‑19 Global Data** dataset. The goal is to help stakeholders understand how the pandemic
        evolved over time, identify patterns in case and death counts across countries and regions, and
        draw actionable insights. Use the sidebar to filter by date range, WHO region and country.
        """
    )

    # Load data
    df = get_dataset()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range selection – Streamlit returns a single date when a range is not selected.
    min_date = df['Date_reported'].min().date()
    max_date = df['Date_reported'].max().date()
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    # Normalize date_range to always have start_date and end_date
    if isinstance(date_range, date):
        start_date = date_range
        end_date = date_range
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.sidebar.warning("Please select a valid date or date range.")
        return

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

    # Tabs for different sections
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    tab_trend, tab_top, tab_maps, tab_seasonality = st.tabs([
        "Trend", "Top Countries", "Maps", "Seasonality"
    ])

    # Trend tab: plotly line chart
    with tab_trend:
        st.subheader("New cases and deaths over time")
        time_series = filtered_df.groupby('Date_reported').agg({
            'New_cases': 'sum',
            'New_deaths': 'sum'
        }).reset_index()
        fig_trend = px.line(
            time_series,
            x='Date_reported',
            y=['New_cases', 'New_deaths'],
            labels={'value': 'Count', 'Date_reported': 'Date', 'variable': 'Metric'},
            title='New cases and new deaths over time',
        )
        fig_trend.update_layout(legend_title_text='')
        st.plotly_chart(fig_trend, use_container_width=True)

    # Top countries tab: plotly bar charts
    with tab_top:
        st.subheader("Top 10 countries by cumulative cases and deaths (latest day)")
        latest_date = filtered_df['Date_reported'].max()
        latest_df = filtered_df[filtered_df['Date_reported'] == latest_date].copy()
        latest_df['Cumulative_cases'] = latest_df['Cumulative_cases'].fillna(0)
        latest_df['Cumulative_deaths'] = latest_df['Cumulative_deaths'].fillna(0)
        top_countries_cases = latest_df.nlargest(10, 'Cumulative_cases')[['Country', 'Cumulative_cases']]
        top_countries_deaths = latest_df.nlargest(10, 'Cumulative_deaths')[['Country', 'Cumulative_deaths']]
        # Create two plotly bar charts
        colA, colB = st.columns(2)
        fig_cases = px.bar(
            top_countries_cases.sort_values('Cumulative_cases'),
            x='Cumulative_cases', y='Country', orientation='h',
            title='Top 10 countries by cumulative cases',
            labels={'Cumulative_cases': 'Cumulative cases', 'Country': ''}
        )
        fig_cases.update_layout(yaxis=dict(categoryorder='total ascending'))
        colA.plotly_chart(fig_cases, use_container_width=True)
        fig_deaths = px.bar(
            top_countries_deaths.sort_values('Cumulative_deaths'),
            x='Cumulative_deaths', y='Country', orientation='h',
            title='Top 10 countries by cumulative deaths',
            labels={'Cumulative_deaths': 'Cumulative deaths', 'Country': ''}
        )
        fig_deaths.update_layout(yaxis=dict(categoryorder='total ascending'))
        colB.plotly_chart(fig_deaths, use_container_width=True)

    # Maps tab: plotly choropleth maps
    with tab_maps:
        st.subheader("Geographical distribution")
        # Prepare dataset aggregated by country across selected period
        # Exclude pseudo-entity 'International commercial vessel'
        map_df = filtered_df[filtered_df['Country'] != 'International commercial vessel']
        agg = map_df.groupby('Country').agg({
            'New_cases': 'sum',
            'New_deaths': 'sum'
        }).reset_index()
        # Compute total cases and death ratio
        agg['Deaths_per_case'] = np.where(agg['New_cases'] > 0, (agg['New_deaths'] / agg['New_cases']) * 100, 0)
        # Choropleth for total cases
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
        # Choropleth for death ratio
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

    # Seasonality tab: heatmap
    with tab_seasonality:
        st.subheader("Seasonality: monthly new cases and deaths")
        data = filtered_df.copy()
        data['Year'] = data['Date_reported'].dt.year
        data['Month'] = data['Date_reported'].dt.month
        monthly = data.groupby(['Year', 'Month']).agg({'New_cases': 'sum', 'New_deaths': 'sum'}).reset_index()
        cases_pivot = monthly.pivot(index='Year', columns='Month', values='New_cases')
        deaths_pivot = monthly.pivot(index='Year', columns='Month', values='New_deaths')
        fig4, ax4 = plt.subplots(2, 1, figsize=(10, 8))
        sns.heatmap(cases_pivot, cmap='Blues', ax=ax4[0], annot=False)
        ax4[0].set_title('Monthly new cases by year')
        ax4[0].set_xlabel('Month')
        ax4[0].set_ylabel('Year')
        sns.heatmap(deaths_pivot, cmap='Reds', ax=ax4[1], annot=False)
        ax4[1].set_title('Monthly new deaths by year')
        ax4[1].set_xlabel('Month')
        ax4[1].set_ylabel('Year')
        plt.tight_layout()
        st.pyplot(fig4)

    # Additional insights or notes
    st.markdown("""
    ### Notes and limitations
    - Data represent official reports submitted to WHO and may lag behind real-time events.
    - Missing values are treated as zeros, which could underestimate true counts.
    - Reporting practices vary across countries and over time, so comparisons should be interpreted with caution.
    - Cumulative figures reflect the latest reported values within the selected time frame and may differ from the global totals due to retrospective updates.
    """)

if __name__ == "__main__":
    main()