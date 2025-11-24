"""
Streamlit Dashboard: Modern Renewable Energy Consumption
-------------------------------------------------------

This script builds an interactive dashboard for exploring the
`modern‑renewable‑energy‑consumption.csv` dataset. The goal of the
dashboard is not only to visualise the data but also to tell a
compelling story by following best practices for data storytelling and
design.  Key design principles drawn from Dr. Sabur Butt’s lecture
materials include:

* **Highlight the important stuff** – draw the eye to the most
  relevant information and remove distractions【301083788703479†L601-L604】.
* **Eliminate distractions and create a clear hierarchy of
  information** – arrange content so that the audience naturally
  follows your narrative【301083788703479†L601-L604】.
* **Be smart with colour, alignment and whitespace** – a clean and
  spacious layout improves comprehension.  Preserving margins and
  avoiding overcrowding makes the dashboard more inviting【301083788703479†L623-L635】.

The dashboard offers the following interactive features:

1. **Entity and year selectors** in the sidebar to filter the
   dataset. Users can choose one or more regions/countries and a
   year range.
2. **Summary KPIs** that update automatically based on the current
   selection, showing total renewable generation for each energy type
   and the percentage change over the selected period.
3. **Line and area charts** created with Altair.  These charts use
   colour to differentiate energy types and provide interactive
   tooltips and brushing so users can focus on particular series.
4. **Bar chart comparison** for the most recent year in the selected
   range, allowing a quick snapshot of how energy sources compare.

To run this dashboard locally, make sure you have installed the
required dependencies (Streamlit, pandas, numpy, and altair).  Then
execute `streamlit run dashboard.py` from the command line.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px

# Define colour scales for each energy type.  Each scale moves from a
# very light tint to the base colour, providing a single‑hue gradient.
COLOR_SCALE_MAP = {
    "Other": ["#eef7ef", "#6CA965", "#2e5032"],
    "Solar": ["#fff6e5", "#E69F00", "#a35f00"],
    "Wind": ["#e9f5fc", "#56B4E9", "#1c73a6"],
    "Hydro": ["#e9eef7", "#4E79A7", "#263f63"],
}


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the renewable energy consumption data.

    The function is cached to avoid reloading the file on every
    interaction.  Rows with missing year values are dropped, and
    column names are cleaned for easier reference.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing renewable generation data.

    Returns
    -------
    DataFrame
        A tidy DataFrame with renamed columns and appropriate data
        types.
    """
    df = pd.read_csv(csv_path)
    # Drop rows where year is NaN and ensure the correct type
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    # Rename columns to shorter, friendly names
    rename_map = {
        "Other renewables (including geothermal and biomass) electricity generation - TWh": "Other",
        "Solar generation - TWh": "Solar",
        "Wind generation - TWh": "Wind",
        "Hydro generation - TWh": "Hydro",
    }
    df = df.rename(columns=rename_map)
    # Fill missing energy values with zero for easier aggregation
    energy_cols = ["Other", "Solar", "Wind", "Hydro"]
    df[energy_cols] = df[energy_cols].fillna(0.0)
    return df


def compute_summary(df: pd.DataFrame, start_year: int, end_year: int, entities: list[str]) -> pd.DataFrame:
    """Aggregate renewable generation over the selected period and entities.

    Parameters
    ----------
    df : DataFrame
        Full dataset with energy generation values.
    start_year : int
        The first year of the selected range.
    end_year : int
        The last year of the selected range.
    entities : list of str
        The regions/countries selected by the user.

    Returns
    -------
    DataFrame
        A one‑row DataFrame containing total generation per energy type
        and the percentage change between the start and end years.
    """
    # Filter data
    mask = (df["Year"] >= start_year) & (df["Year"] <= end_year) & (df["Entity"].isin(entities))
    df_sel = df.loc[mask].copy()
    energy_cols = ["Other", "Solar", "Wind", "Hydro"]
    # Sum over time and entities
    totals = df_sel[energy_cols].sum()
    # Compute growth: (value at end year – value at start year) / start
    growth = {}
    for col in energy_cols:
        # Aggregated value per year across selected entities
        start_val = df_sel.loc[df_sel["Year"] == start_year, col].sum()
        end_val = df_sel.loc[df_sel["Year"] == end_year, col].sum()
        if start_val == 0:
            growth[col] = np.nan
        else:
            growth[col] = (end_val - start_val) / start_val * 100
    summary = pd.DataFrame({
        "Energy": energy_cols,
        "Total (TWh)": totals.values,
        "Growth (%)": [growth[col] for col in energy_cols],
    })
    return summary


def build_line_chart(df: pd.DataFrame, entities: list[str], energy_types: list[str], year_range: tuple[int, int]) -> alt.Chart:
    """Create an interactive line chart showing energy generation over time.

    Users can toggle which energy types to display via the multiselect
    widget.  The chart uses a colour palette to differentiate the
    series and allows interactive selection through legends.

    Parameters
    ----------
    df : DataFrame
        Filtered dataset (already limited to selected entities and year range).
    entities : list of str
        Names of the selected countries/regions.
    energy_types : list of str
        Energy types to include in the line chart.
    year_range : tuple[int, int]
        (start_year, end_year) selected by the user.

    Returns
    -------
    alt.Chart
        An Altair chart object ready to be rendered by Streamlit.
    """
    start_year, end_year = year_range
    # Melt the dataframe to long format for Altair
    energy_cols = ["Other", "Solar", "Wind", "Hydro"]
    df_long = df.melt(id_vars=["Entity", "Code", "Year"], value_vars=energy_cols, var_name="Energy", value_name="Generation")
    df_long = df_long[df_long["Energy"].isin(energy_types)]
    # Create colour scale – use a consistent palette to aid interpretation
    palette = {
        "Other": "#6CA965",  # muted green
        "Solar": "#E69F00",  # warm orange
        "Wind": "#56B4E9",   # sky blue
        "Hydro": "#4E79A7",  # deep blue
    }
    # Selection for interactive legend
    selection = alt.selection_multi(fields=["Energy"], bind="legend")
    base = alt.Chart(df_long).encode(
        x=alt.X("Year:O", title="Year", sort=list(range(start_year, end_year + 1))),
        y=alt.Y("Generation:Q", title="Electricity Generation (TWh)", stack=None),
        color=alt.Color("Energy:N", scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values())), title="Energy Type"),
        tooltip=["Entity", "Energy", "Year", alt.Tooltip("Generation", format=",.1f", title="Generation (TWh)")],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
    )
    line_chart = base.mark_line(opacity=0.85, strokeWidth=2.5).add_selection(selection)
    if len(entities) > 1:
        line_chart = line_chart.facet(
            row=alt.Row("Entity:N", title="Entity", header=alt.Header(labelOrient="left")),
        ).resolve_scale(y="shared", x="shared")
    return line_chart.properties(width=700, height=400)


def build_area_chart(df: pd.DataFrame, entities: list[str], energy_types: list[str], year_range: tuple[int, int]) -> alt.Chart:
    """Create a stacked area chart showing the contribution of each energy source.

    This chart uses a stack to emphasise the cumulative effect of
    renewables over time and highlights how each source contributes to
    the total.  The interactive legend allows viewers to isolate a
    single series.
    """
    start_year, end_year = year_range
    energy_cols = ["Other", "Solar", "Wind", "Hydro"]
    df_long = df.melt(id_vars=["Entity", "Code", "Year"], value_vars=energy_cols, var_name="Energy", value_name="Generation")
    df_long = df_long[df_long["Energy"].isin(energy_types)]
    palette = {
        "Other": "#6CA965",
        "Solar": "#E69F00",
        "Wind": "#56B4E9",
        "Hydro": "#4E79A7",
    }
    selection = alt.selection_multi(fields=["Energy"], bind="legend")
    base = alt.Chart(df_long).encode(
        x=alt.X("Year:O", title="Year", sort=list(range(start_year, end_year + 1))),
        y=alt.Y("sum(Generation):Q", title="Total Generation (TWh)"),
        color=alt.Color("Energy:N", scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values())), title="Energy Type"),
        tooltip=["Entity", "Energy", "Year", alt.Tooltip("Generation", format=",.1f", title="Generation (TWh)")],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
    )
    area_chart = base.mark_area(opacity=0.6).add_selection(selection)
    if len(entities) > 1:
        area_chart = area_chart.facet(
            row=alt.Row("Entity:N", title="Entity", header=alt.Header(labelOrient="left")),
        ).resolve_scale(y="independent", x="shared")
    return area_chart.properties(width=700, height=400)


def build_pie_chart(summary_df: pd.DataFrame) -> px.pie:
    """Create a pie chart showing percentage share of each energy type.

    Parameters
    ----------
    summary_df : DataFrame
        Aggregated totals for the selected period.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly pie chart figure ready for display.
    """
    # Compute percentages
    df = summary_df.copy()
    total_sum = df["Total (TWh)"].sum()
    df["Percentage"] = (df["Total (TWh)"] / total_sum) * 100
    # Colour mapping consistent with our palette
    palette = {
        "Other": "#6CA965",
        "Solar": "#E69F00",
        "Wind": "#56B4E9",
        "Hydro": "#4E79A7",
    }
    fig = px.pie(
        df,
        names="Energy",
        values="Total (TWh)",
        hole=0.4,
        color="Energy",
        color_discrete_map=palette,
    )
    fig.update_traces(textinfo="percent+label", pull=[0.02] * len(df))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def render_kpi_cards(summary_df: pd.DataFrame):
    """Display KPI cards summarising renewable energy generation and growth.

    The function uses Streamlit’s `metric` component to present total
    TWh and growth percentages for each energy type.  Values are
    formatted for readability, and `np.nan` growth values are
    represented with a dash.
    """
    cols = st.columns(len(summary_df))
    for idx, (_, row) in enumerate(summary_df.iterrows()):
        total_formatted = f"{row['Total (TWh)']:.1f} TWh"
        growth = row["Growth (%)"]
        if pd.isna(growth):
            delta = "–"
        else:
            delta = f"{growth:+.1f}%"
        cols[idx].metric(label=row["Energy"], value=total_formatted, delta=delta)


def main():
    # Page configuration
    st.set_page_config(
        page_title="Modern Renewable Energy Consumption Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )

    # Load data
    # Determine where the CSV lives.  Prefer a file in the same directory
    # as this script; if it doesn't exist, fall back to a sibling `share`
    # directory.  This makes it easier to run the script after downloading
    # or relocating the files.
    possible_paths = [
        Path(__file__).parent / "modern-renewable-energy-consumption.csv",
        Path(__file__).parent / "share" / "modern-renewable-energy-consumption.csv",
    ]
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
    if data_path is None:
        st.error("Could not locate the data file 'modern-renewable-energy-consumption.csv'. Please ensure it is placed in the same directory as this script.")
        return
    df = load_data(str(data_path))

    # Sidebar controls
    st.sidebar.header("Filters")
    entities = sorted(df["Entity"].unique())
    default_entities = ["World"] if "World" in entities else [entities[0]]
    selected_entities = st.sidebar.multiselect(
        "Select entities (countries/regions)",
        options=entities,
        default=default_entities,
    )
    year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
    selected_years = st.sidebar.slider(
        "Select year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
    energy_types = ["Other", "Solar", "Wind", "Hydro"]
    default_energy_types = energy_types.copy()
    selected_energy_types = st.sidebar.multiselect(
        "Select energy types", options=energy_types, default=default_energy_types
    )
    st.sidebar.markdown(
        """
        **How to use**
        
        • Choose one or more countries/regions to explore.  By default the
        dashboard shows the **World** aggregate.
        • Adjust the year range to focus on a particular period.
        • Toggle energy types to see only the categories you care about.
        """
    )

    # Main content
    # Header image (optional decorative element)
    # Determine header image path
    image_candidates = [
        Path(__file__).parent / "130447cf-07cb-4761-a4eb-0c5b3147f781.png",
        Path(__file__).parent / "share" / "130447cf-07cb-4761-a4eb-0c5b3147f781.png",
    ]
    header_image = None
    for img in image_candidates:
        if img.exists():
            header_image = img
            break
    if header_image is not None:
        st.image(str(header_image), use_column_width=True)
    st.title("Modern Renewable Energy Consumption")
    st.caption(
        "Explore trends in renewable electricity generation across the world and compare the\n"
        "contribution of different energy sources over time. Use the filters on the left to\n"
        "tailor the view to your interests."
    )

    # Guard against empty selections
    if not selected_entities:
        st.warning("Please select at least one entity from the sidebar.")
        return
    if not selected_energy_types:
        st.warning("Please select at least one energy type from the sidebar.")
        return

    # Filter the dataset according to selections
    start_year, end_year = selected_years
    mask = (
        (df["Entity"].isin(selected_entities))
        & (df["Year"] >= start_year)
        & (df["Year"] <= end_year)
    )
    df_filtered = df.loc[mask]

    # Compute summary metrics
    summary_df = compute_summary(df, start_year, end_year, selected_entities)

    # KPI cards
    render_kpi_cards(summary_df)

    # Layout for charts
    st.markdown("### Trends over time")
    line_chart = build_line_chart(df_filtered, selected_entities, selected_energy_types, selected_years)
    st.altair_chart(line_chart, use_container_width=True)

    # Stacked area chart and bar chart side by side
    st.markdown("### Composition of renewable sources and comparison")
    col_left, col_right = st.columns([2, 1], gap="medium")
    area_chart = build_area_chart(df_filtered, selected_entities, selected_energy_types, selected_years)
    pie_chart = build_pie_chart(summary_df)
    with col_left:
        st.altair_chart(area_chart, use_container_width=True)
    with col_right:
        st.plotly_chart(pie_chart, use_container_width=True)

    # Geographical distribution using a choropleth map
    # Compute aggregated totals for all countries for the end year in selected range
    geo_year = end_year
    geo_df = df[(df["Year"] == geo_year) & df["Code"].notna()].copy()
    # Exclude aggregated regions (codes starting with 'OWID_' or length not equal to 3)
    geo_df = geo_df[geo_df["Code"].str.len() == 3]
    # Sum selected energy types into a single value
    geo_df["Total"] = geo_df[selected_energy_types].sum(axis=1)
    # Choose colour scale based on selected energy type.  If multiple energy types
    # are selected, use the first one as the reference for the scale.
    if selected_energy_types:
        energy_key = selected_energy_types[0]
    else:
        energy_key = "Other"
    color_scale = COLOR_SCALE_MAP.get(energy_key, ["#f0f0f0", "#999999", "#555555"])
    if not geo_df.empty:
        title_energy = energy_key if len(selected_energy_types) == 1 else "Selected Sources"
        st.markdown(f"### Geographical distribution of {title_energy} ({geo_year})")
        fig = px.choropleth(
            geo_df,
            locations="Code",
            color="Total",
            hover_name="Entity",
            projection="natural earth",
            color_continuous_scale=color_scale,
            labels={"Total": "Total generation (TWh)"},
        )
        fig.update_geos(
            showcoastlines=False,
            showland=True,
            landcolor="#f7f7f7",
            showcountries=True,
            countrycolor="white",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="TWh"),
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()