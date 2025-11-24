"""
Enhanced interactive Streamlit dashboard for modern renewable energy consumption

This version of the dashboard refines the design in accordance with key
principles from the "Storytelling with Data" lecture notes (Sabur Butt):

* **Preattentive attributes** (size, color and position) are leveraged to
  guide the viewer’s eye toward important trends and comparisons【149423737840243†L509-L514】. A restrained
  color palette highlights the selected entity or the top performer, while
  neutral tones keep secondary elements in the background.
* **Clarity over clutter**: axes and gridlines are understated, and titles
  communicate the chart’s takeaway rather than simply describing the chart.
* **White space and alignment**: spacing between components and thoughtful
  layout ensure the dashboard feels open and the viewer isn’t overwhelmed【149423737840243†L621-L635】.

The dashboard uses Plotly for interactive charts and includes three
functional panels: a time series explorer, a top generators ranking, and a
global shares view. Use the sidebar to choose the entity, energy sources,
and year. To run locally, install Streamlit and Plotly (``pip install
streamlit pandas plotly``) and run ``streamlit run
streamlit_interactive_plotly.py``.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """Load and transform the renewable energy CSV into long format.

    The dataset contains columns for different renewable sources. This
    function renames the columns to shorter labels, melts them into a
    long-form table and coerces numeric values to floats.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Long-form DataFrame with columns ['Entity','Year','Source','TWh'].
    """
    df = pd.read_csv(csv_path)
    # Rename verbose columns
    rename_map = {
        "Hydro generation - TWh": "Hydro",
        "Wind generation - TWh": "Wind",
        "Solar generation - TWh": "Solar",
        "Other renewables (including geothermal and biomass) electricity generation - TWh": "Other",
    }
    df = df.rename(columns=rename_map)
    long_df = df.melt(id_vars=["Entity", "Year"], value_vars=list(rename_map.values()),
                      var_name="Source", value_name="TWh")
    long_df['TWh'] = pd.to_numeric(long_df['TWh'], errors='coerce')
    long_df = long_df.dropna(subset=['TWh'])
    return long_df


def create_color_map(sources: List[str]) -> dict:
    """Generate a color map for the renewable sources.

    We choose colors that are distinct yet harmonious. The palette is kept
    simple to aid interpretation and is colourblind-friendly. Hydro is
    assigned a cool blue, wind a green, solar a warm orange, and other
    renewables a neutral grey.

    Args:
        sources: List of energy source names.

    Returns:
        A dictionary mapping each source to a hex color.
    """
    palette = {
        'Hydro': '#2C7BB6',    # blue
        'Wind': '#00A676',     # green
        'Solar': '#F79D02',    # orange
        'Other': '#8E8E8E',    # grey
    }
    return {s: palette.get(s, '#333333') for s in sources}


def time_series_chart(long_df: pd.DataFrame, entity: str, sources: List[str], log_scale: bool) -> go.Figure:
    """Create a time series line chart for selected entity and sources.

    Args:
        long_df: The long-form DataFrame.
        entity: Selected entity (country or region).
        sources: List of selected sources to display.
        log_scale: Whether to use a log scale for the y-axis.

    Returns:
        A Plotly figure object.
    """
    dff = long_df[(long_df['Entity'] == entity) & (long_df['Source'].isin(sources))]
    color_map = create_color_map(sources)
    fig = go.Figure()
    for source in sources:
        df_source = dff[dff['Source'] == source]
        fig.add_trace(go.Scatter(
            x=df_source['Year'],
            y=df_source['TWh'],
            mode='lines+markers',
            name=source,
            marker=dict(color=color_map[source]),
            line=dict(color=color_map[source]),
            hovertemplate='%{y:.1f} TWh in %{x}'
        ))
    # Layout customisation to align with design rules
    y_axis_type = 'log' if log_scale else 'linear'
    fig.update_layout(
        title=dict(
            text=f"{entity}: Renewable generation over time",
            x=0.0,  # left aligned for intuitive scanning【149423737840243†L581-L585】
            xanchor='left'
        ),
        xaxis_title='Year',
        yaxis_title='Electricity generation (TWh)',
        yaxis_type=y_axis_type,
        legend_title_text='Source',
        plot_bgcolor='#F7F7F7',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
        margin=dict(l=50, r=30, t=60, b=40),
        height=400
    )
    return fig


def top_generators_chart(long_df: pd.DataFrame, year: int, source: str, n: int = 10) -> go.Figure:
    """Create a horizontal bar chart ranking top entities for a given source/year.

    The top performer is highlighted using a vivid colour, while others are
    displayed in a muted grey to draw the viewer’s attention【149423737840243†L509-L514】.

    Args:
        long_df: The long-form DataFrame.
        year: Year to filter on.
        source: Renewable source to filter on.
        n: Number of top entities to display.

    Returns:
        A Plotly figure object.
    """
    # Filter and remove aggregate regions
    aggregates = ["World", "Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    dff = long_df[(long_df['Year'] == year) & (long_df['Source'] == source) & (~long_df['Entity'].isin(aggregates))]
    top = dff.nlargest(n, 'TWh').sort_values('TWh', ascending=True)
    # Determine highlight colour for the top entity
    highlight_color = create_color_map([source])[source]
    colors = [highlight_color if ent == top.iloc[-1]['Entity'] else '#C0C0C0' for ent in top['Entity']]
    fig = go.Figure(go.Bar(
        x=top['TWh'],
        y=top['Entity'],
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='%{y}: %{x:.1f} TWh'
    ))
    fig.update_layout(
        title=dict(
            text=f"Top {n} {source.lower()} generators in {year}",
            x=0.0,
            xanchor='left'
        ),
        xaxis_title='Electricity generation (TWh)',
        yaxis_title='',
        plot_bgcolor='#F7F7F7',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=120, r=30, t=60, b=40),
        height=400
    )
    return fig


def global_shares_chart(long_df: pd.DataFrame) -> go.Figure:
    """Create a stacked area chart of global renewable generation by source.

    This chart normalises each year to highlight the changing composition of
    renewable generation. The palette leverages distinct colours while
    maintaining harmony and accessibility. A neutral background and subtle
    gridlines improve readability【149423737840243†L621-L635】.

    Args:
        long_df: The long-form DataFrame.

    Returns:
        A Plotly figure object.
    """
    df_global = long_df.groupby(['Year', 'Source'])['TWh'].sum().reset_index()
    df_pivot = df_global.pivot(index='Year', columns='Source', values='TWh').fillna(0)
    # Normalise to fraction of total
    df_frac = df_pivot.divide(df_pivot.sum(axis=1), axis=0).reset_index()
    fig = go.Figure()
    sources = df_global['Source'].unique().tolist()
    color_map = create_color_map(sources)
    for source in sources:
        fig.add_trace(go.Scatter(
            x=df_frac['Year'],
            y=df_frac[source],
            mode='lines',
            stackgroup='one',
            name=source,
            line=dict(width=0.5),
            fillcolor=color_map[source],
            hovertemplate='%{y:.1%} share of global renewable generation in %{x}'
        ))
    fig.update_layout(
        title=dict(
            text="How the mix of renewable generation has changed over time",
            x=0.0,
            xanchor='left'
        ),
        xaxis_title='Year',
        yaxis_title='Share of total (0-100%)',
        yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='lightgrey'),
        xaxis=dict(showgrid=False),
        legend_title_text='Source',
        plot_bgcolor='#F7F7F7',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=50, r=30, t=60, b=40),
        height=400
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Renewable Energy Dashboard", layout="wide")
    st.title("Modern Renewable Energy Consumption Dashboard")
    st.markdown(
        """
        **Big idea:** Renewable electricity generation has undergone a seismic
        shift over the past two decades. Hydropower, once the clear leader,
        now shares the stage with rapidly growing wind and solar technologies.
        Explore how this transition plays out across countries and time.
        """
    )
    # Load data
    data_path = "modern-renewable-energy-consumption.csv"
    long_df = load_data(data_path)
    # Sidebar controls for time series
    with st.sidebar:
        st.header("Time series controls")
        entities = sorted(long_df['Entity'].unique())
        default_entity = "World" if "World" in entities else entities[0]
        entity = st.selectbox("Entity", entities, index=entities.index(default_entity))
        sources = ['Hydro', 'Wind', 'Solar', 'Other']
        selected_sources = st.multiselect("Sources", options=sources, default=sources)
        log_scale = st.checkbox("Logarithmic scale", value=False)
        st.markdown("---")
        st.header("Top generators controls")
        years = sorted(long_df['Year'].unique())
        year = st.slider("Year", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[-1]))
        source_for_bar = st.selectbox("Source", options=sources, index=sources.index('Solar'))
    # Create charts
    ts_fig = time_series_chart(long_df, entity, selected_sources, log_scale)
    bar_fig = top_generators_chart(long_df, year, source_for_bar)
    area_fig = global_shares_chart(long_df)
    # Layout charts in columns
    st.subheader("Explore over time")
    st.plotly_chart(ts_fig, use_container_width=True)
    st.subheader("Leading countries in selected year")
    st.plotly_chart(bar_fig, use_container_width=True)
    st.subheader("Global renewable mix")
    st.plotly_chart(area_fig, use_container_width=True)
    # Narrative summary
    st.markdown(
        """
        ### What the data tells us
        * Hydropower remains a cornerstone of renewable electricity. However, since 2010
          wind and solar have grown exponentially, eating into hydro’s share.
        * In recent years, countries like **China and the United States** dominate
          solar and wind generation, while other nations catch up at different
          speeds.
        * The global mix chart shows that diversity in renewable sources
          is increasing, suggesting a more resilient and sustainable energy future.
        """
    )


if __name__ == '__main__':
    main()