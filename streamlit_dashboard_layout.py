"""
Dashboard layout inspired by the provided financial dashboard image.

This Streamlit app arranges charts and metrics into a grid reminiscent of
executive dashboards: a top row of four key indicators (using donut
charts), a middle row with a bar chart and a line chart, and a bottom row
with a stacked area chart. The color palette has been selected to be
visually appealing, distinct for each renewable source, and suitable for
colourblind audiences. The layout and design adhere to the storytelling
principles described in the course notes (focus attention through colour
and position, minimise clutter, and use white space effectively【149423737840243†L509-L514】【149423737840243†L621-L635】).

To run this app locally: ``pip install streamlit pandas plotly`` then
``streamlit run streamlit_dashboard_layout.py``.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the renewable energy dataset and convert to long format."""
    df = pd.read_csv(path)
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


def color_palette() -> dict:
    """Return a consistent color palette for the renewable sources."""
    return {
        'Hydro': '#4E79A7',    # deep blue
        'Wind': '#59A14F',     # green
        'Solar': '#F28E2B',    # orange
        'Other': '#AF7AA1',    # purple
    }


def make_donut(value: float, color: str, title: str) -> go.Figure:
    """Create a donut chart showing a proportion.

    Args:
        value: Share value between 0 and 1.
        color: Colour for the filled portion.
        title: Title displayed above the donut.

    Returns:
        Plotly Figure representing a donut chart with a central label.
    """
    remainder = 1 - value
    fig = go.Figure(data=[go.Pie(
        values=[value, remainder],
        hole=0.7,
        sort=False,
        direction='clockwise',
        marker=dict(colors=[color, '#E5E5E5']),
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        annotations=[
            dict(text=f"{value * 100:.1f}%", x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color=color)),
            dict(text=title, x=0.5, y=1.1, font_size=14, showarrow=False, xanchor='center')
        ],
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF'
    )
    return fig


def bar_chart(long_df: pd.DataFrame, year: int, source: str, palette: dict, n: int = 10) -> go.Figure:
    """Create a horizontal bar chart ranking top entities for a given source/year."""
    aggregates = ["World", "Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    dff = long_df[(long_df['Year'] == year) & (long_df['Source'] == source) & (~long_df['Entity'].isin(aggregates))]
    top = dff.nlargest(n, 'TWh').sort_values('TWh', ascending=True)
    color = palette[source]
    bars = go.Bar(
        x=top['TWh'],
        y=top['Entity'],
        orientation='h',
        marker=dict(color=color),
        hovertemplate='%{y}: %{x:.1f} TWh'
    )
    fig = go.Figure(bars)
    fig.update_layout(
        title=dict(text=f"Top {n} {source.lower()} generators in {year}", x=0.0, xanchor='left'),
        xaxis_title='Electricity generation (TWh)',
        yaxis_title='',
        plot_bgcolor='#F7F7F7',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=120, r=30, t=40, b=40),
        height=350
    )
    return fig


def line_chart(long_df: pd.DataFrame, entity: str, sources: List[str], palette: dict) -> go.Figure:
    """Create a multi-series line chart for the selected entity and sources."""
    dff = long_df[(long_df['Entity'] == entity) & (long_df['Source'].isin(sources))]
    fig = go.Figure()
    for source in sources:
        df_source = dff[dff['Source'] == source]
        fig.add_trace(go.Scatter(
            x=df_source['Year'],
            y=df_source['TWh'],
            mode='lines+markers',
            name=source,
            marker=dict(color=palette[source]),
            line=dict(color=palette[source]),
            hovertemplate='%{x}: %{y:.1f} TWh'
        ))
    fig.update_layout(
        title=dict(text=f"{entity}: renewable generation trend", x=0.0, xanchor='left'),
        xaxis_title='Year',
        yaxis_title='Electricity generation (TWh)',
        plot_bgcolor='#F7F7F7',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
        margin=dict(l=50, r=30, t=40, b=40),
        height=350
    )
    return fig


def area_chart(long_df: pd.DataFrame, palette: dict) -> go.Figure:
    """Create a normalized stacked area chart of global renewable mix."""
    df_global = long_df.groupby(['Year', 'Source'])['TWh'].sum().reset_index()
    df_pivot = df_global.pivot(index='Year', columns='Source', values='TWh').fillna(0)
    df_frac = df_pivot.divide(df_pivot.sum(axis=1), axis=0).reset_index()
    fig = go.Figure()
    sources = ['Hydro', 'Wind', 'Solar', 'Other']
    for source in sources:
        fig.add_trace(go.Scatter(
            x=df_frac['Year'],
            y=df_frac[source],
            mode='lines',
            stackgroup='one',
            name=source,
            line=dict(width=0.5),
            fillcolor=palette[source],
            hovertemplate='%{x}: %{y:.1%}'
        ))
    fig.update_layout(
        title=dict(text="Changing composition of global renewable generation", x=0.0, xanchor='left'),
        xaxis_title='Year',
        yaxis_title='Share of total',
        yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='lightgrey'),
        xaxis=dict(showgrid=False),
        legend_title_text='Source',
        plot_bgcolor='#F7F7F7',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=50, r=30, t=40, b=40),
        height=350
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Renewable Energy Dashboard", layout="wide")
    st.title("Modern Renewable Energy Dashboard")
    st.markdown(
        """
        This dashboard summarises key metrics and trends in modern renewable
        electricity generation. Use the sidebar to choose a country or region,
        pick a year and a source, and explore how different technologies
        contribute to the energy mix.
        """
    )
    # Load data
    data_path = "modern-renewable-energy-consumption.csv"
    long_df = load_data(data_path)
    palette = color_palette()
    # Sidebar controls
    entities = sorted(long_df['Entity'].unique())
    default_entity = 'World' if 'World' in entities else entities[0]
    entity = st.sidebar.selectbox("Entity", entities, index=entities.index(default_entity))
    years = sorted(long_df['Year'].unique())
    year = st.sidebar.slider("Year", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[-1]))
    sources = ['Hydro', 'Wind', 'Solar', 'Other']
    source_for_bar = st.sidebar.selectbox("Source for ranking", sources, index=sources.index('Solar'))
    selected_sources = st.sidebar.multiselect("Sources for trend", sources, default=sources)
    # Compute metrics for selected entity and year
    entity_year_df = long_df[(long_df['Entity'] == entity) & (long_df['Year'] == year)]
    total_gen = entity_year_df['TWh'].sum()
    # Avoid division by zero
    shares = {}
    for src in sources:
        val = entity_year_df[entity_year_df['Source'] == src]['TWh'].sum()
        shares[src] = (val / total_gen) if total_gen > 0 else 0
    # Top metrics row
    cols = st.columns(4)
    metrics_titles = [f"Total generation\n{year}", "Hydro share", "Wind share", "Solar share"]
    metrics_values = [1.0, shares['Hydro'], shares['Wind'], shares['Solar']]
    metrics_colors = ['#2C7BB6', palette['Hydro'], palette['Wind'], palette['Solar']]
    metrics_display_values = [f"{total_gen:.1f} TWh" if i == 0 else None for i in range(4)]
    for idx, col in enumerate(cols):
        with col:
            if idx == 0:
                # Display total generation as a number without donut
                st.metric(label=metrics_titles[idx], value=metrics_display_values[idx])
            else:
                fig = make_donut(metrics_values[idx], metrics_colors[idx], metrics_titles[idx])
                st.plotly_chart(fig, use_container_width=True)
    # Middle row: bar chart and line chart
    bar_fig = bar_chart(long_df, year, source_for_bar, palette)
    line_fig = line_chart(long_df, entity, selected_sources, palette)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(bar_fig, use_container_width=True)
    with col_right:
        st.plotly_chart(line_fig, use_container_width=True)
    # Bottom row: area chart
    area_fig = area_chart(long_df, palette)
    st.plotly_chart(area_fig, use_container_width=True)
    # Interpretive comments
    st.markdown(
        """
        #### Insights
        * The donut charts reveal the distribution of renewable technologies for the selected entity and year.
        * Use the bar chart to compare leading producers for a chosen technology; note how a small number of countries dominate wind and solar generation.
        * The line chart shows whether the selected entity is keeping pace with the global shift toward wind and solar.
        * The area chart at the bottom illustrates how wind and solar have risen from negligible shares to major components of the global renewable mix.
        """
    )


if __name__ == '__main__':
    main()