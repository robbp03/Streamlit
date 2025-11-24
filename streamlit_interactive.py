"""
Interactive Streamlit dashboard for modern renewable energy consumption

This dashboard reads the Our World in Data CSV on modern renewable energy
generation and provides several interactive charts:

* A line chart that lets the viewer pick a country/region and one or more
  renewable sources (hydro, wind, solar, other) to see how generation has
  changed over time.
* A bar chart to explore which countries lead in a chosen energy source for a
  selected year. The user can select the source and year to update the
  ranking dynamically.
* A stacked area chart summarising global renewable generation by source
  across the full time range.

To run this app locally, install Streamlit (``pip install streamlit altair
  pandas``) and then execute ``streamlit run streamlit_interactive.py``.
"""

import streamlit as st
import pandas as pd
import altair as alt


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the renewable energy dataset and prepare it for plotting.

    The function reads the CSV file, drops rows with missing year values
    and standardises the column names to shorter labels. It also converts
    the dataset to a "long" format where each row corresponds to a single
    energy source for a given entity and year. Caching ensures the data is
    loaded only once per session.

    Args:
        path: The relative path to the CSV file.

    Returns:
        A pandas DataFrame in long format with columns ['Entity', 'Year',
        'Source', 'TWh'].
    """
    df = pd.read_csv(path)
    # Rename columns for simplicity
    rename_map = {
        "Hydro generation - TWh": "Hydro",
        "Wind generation - TWh": "Wind",
        "Solar generation - TWh": "Solar",
        "Other renewables (including geothermal and biomass) electricity generation - TWh": "Other",
    }
    df = df.rename(columns=rename_map)
    # Melt to long format
    long_df = df.melt(id_vars=["Entity", "Year"], value_vars=list(rename_map.values()),
                      var_name="Source", value_name="TWh")
    # Ensure numeric and drop NaNs
    long_df['TWh'] = pd.to_numeric(long_df['TWh'], errors='coerce')
    long_df = long_df.dropna(subset=['TWh'])
    return long_df


def line_chart(long_df: pd.DataFrame) -> None:
    """Render an interactive line chart for renewable generation over time.

    This component allows the user to select an entity (country or region) and
    one or more energy sources. The chart updates to show a line for each
    selected source across years. The y-axis uses a linear scale; the user
    can optionally toggle to a logarithmic scale via a checkbox.

    Args:
        long_df: The long-format DataFrame with renewable generation data.
    """
    st.header("Renewable generation over time")
    # Unique entities sorted alphabetically for the selectbox
    entities = sorted(long_df['Entity'].unique())
    default_entity = "World" if "World" in entities else entities[0]
    entity = st.selectbox("Select entity (country or region):", entities, index=entities.index(default_entity))
    sources = ['Hydro', 'Wind', 'Solar', 'Other']
    selected_sources = st.multiselect("Select renewable sources:", sources, default=sources)
    y_log = st.checkbox("Use logarithmic y-axis", value=False)
    # Filter data
    data = long_df[(long_df['Entity'] == entity) & (long_df['Source'].isin(selected_sources))]
    # Build line chart with Altair; interactive tooltips are enabled
    if not data.empty:
        y_scale = alt.Scale(type='log', base=10) if y_log else alt.Scale(type='linear')
        line = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('TWh:Q', scale=y_scale, title='Electricity generation (TWh)'),
            color=alt.Color('Source:N', title='Source'),
            tooltip=['Year:O', 'Source:N', 'TWh:Q']
        ).properties(width=700, height=400)
        st.altair_chart(line, use_container_width=True)
    else:
        st.info("No data available for the selected combination.")


def bar_chart(long_df: pd.DataFrame) -> None:
    """Render an interactive horizontal bar chart ranking top countries.

    Users choose a renewable source and a year. The chart displays the top
    10 countries/entities by generation for that year and source. The bar
    length represents TWh, and the bars are ordered descendingly.

    Args:
        long_df: The long-format DataFrame with renewable generation data.
    """
    st.header("Top renewable generators by source and year")
    sources = ['Hydro', 'Wind', 'Solar', 'Other']
    selected_source = st.selectbox("Select renewable source:", sources, index=sources.index('Solar'))
    # Determine available years; restrict to most recent N years for slider convenience
    years = sorted(long_df['Year'].unique())
    # Use slider to select year; default to the latest year
    selected_year = st.slider("Select year:", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[-1]))
    # Filter by year and source
    subset = long_df[(long_df['Year'] == selected_year) & (long_df['Source'] == selected_source)]
    # Exclude aggregate entities (like World) by requiring a country code (three letters) if available
    # Entities without a space and not labelled as 'World' are typically countries; we can approximate this filter
    # by dropping entries where the entity is in a known aggregate list
    aggregates = ["World", "Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    subset = subset[~subset['Entity'].isin(aggregates)]
    # Get top 10 by generation
    top = subset.nlargest(10, 'TWh').sort_values('TWh', ascending=True)
    if not top.empty:
        bar = alt.Chart(top).mark_bar().encode(
            x=alt.X('TWh:Q', title='Electricity generation (TWh)'),
            y=alt.Y('Entity:N', sort=None, title='Entity'),
            tooltip=['Entity:N', 'TWh:Q']
        ).properties(width=700, height=400)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("No data available for the selected year/source combination.")


def area_chart(long_df: pd.DataFrame) -> None:
    """Render a stacked area chart of global renewable generation by source.

    This summary view aggregates the data across all entities (i.e., global
    totals) and shows how each renewable source contributes to total
    generation over time.

    Args:
        long_df: The long-format DataFrame with renewable generation data.
    """
    st.header("Global renewable generation by source")
    # Aggregate by year and source
    global_df = long_df.groupby(['Year', 'Source'])['TWh'].sum().reset_index()
    area = alt.Chart(global_df).mark_area(opacity=0.7).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('TWh:Q', stack='normalize', title='Share of global renewable generation'),
        color=alt.Color('Source:N', title='Source'),
        tooltip=['Year:O', 'Source:N', 'TWh:Q']
    ).properties(width=700, height=400)
    st.altair_chart(area, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Renewable Energy Dashboard", layout="wide")
    st.title("Modern Renewable Energy Consumption Dashboard")
    st.markdown(
        """
        This interactive dashboard allows you to explore the evolution and
        distribution of modern renewable electricity generation. Use the
        controls below each chart to filter by country/region, energy source
        and year. Data comes from the Our World in Data dataset on modern
        renewable energy consumption (excluding traditional biomass) as of
        2024.
        """
    )
    # Load data
    data_path = "modern-renewable-energy-consumption.csv"
    long_df = load_data(data_path)
    # Render charts in tabs for better organisation
    tab1, tab2, tab3 = st.tabs(["Time Series", "Top Generators", "Global Shares"])
    with tab1:
        line_chart(long_df)
    with tab2:
        bar_chart(long_df)
    with tab3:
        area_chart(long_df)


if __name__ == "__main__":
    main()