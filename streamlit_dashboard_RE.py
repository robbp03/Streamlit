"""
Interactive Streamlit dashboard for the renewable energy data story.

This app presents an end‑to‑end narrative about global renewable energy
trends, leveraging the processed datasets produced by ``data_prep.py``.
It combines time series analysis, composition breakdowns, geographic
distribution, and country comparisons to convey a clear and compelling
story about how renewables have evolved and where opportunities remain.

Running the app:

```
streamlit run streamlit_dashboard.py
```

Ensure the following CSV files exist in the same directory or adjust
the file paths accordingly:

* ``time_series_global.csv``
* ``composition_2024.csv``
* ``country_shares_2024.csv``
* ``top_gen_2024.csv``
* ``slope_data.csv``

The app uses Plotly Express for interactive charts and adheres to
visual design principles taught in the course.  See README or
accompanying report for further details.
"""

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed datasets from CSV files.

    Returns a tuple of DataFrames in the following order:
    (time_series_global, composition, country_shares, top_gen, slope_data).
    """
    ts = pd.read_csv("time_series_global.csv")
    comp = pd.read_csv("composition_2024.csv")
    country = pd.read_csv("country_shares_2024.csv")
    top = pd.read_csv("top_gen_2024.csv")
    slope = pd.read_csv("slope_data.csv")
    return ts, comp, country, top, slope


def page_overview(comp: pd.DataFrame, country: pd.DataFrame) -> None:
    """Display the introductory page with context and key metrics."""
    st.title("Global Renewable Energy Trends")
    st.markdown(
        """
        ## Context & Purpose
        
        **Why renewable energy?** Fossil fuels still dominate global energy supply, accounting for the vast majority of
        electricity and primary energy.  Yet in the last two decades, renewable sources—led by wind and solar—have grown
        rapidly and now play a meaningful role in the global energy mix.  Understanding **how renewables have expanded, what
        they comprise today, and where adoption is strongest** is essential for policymakers, industry stakeholders, and
        anyone interested in climate and sustainability.  This dashboard tells that story.
        
        ### Big Idea
        
        *Global renewable energy generation has grown dramatically since 2000—driven by wind and solar—but adoption remains
        uneven across countries and renewables still supply only a fraction of total energy.  Continued growth is vital to
        meet climate goals.*
        
        ### Key Facts (2024)
        
        - **Global renewable generation:** {total:,.0f} TWh
        - **Hydro share of renewables:** {hydro_share:.1f}%
        - **Top country for renewables generation:** {top_country} ({top_gen:,.0f} TWh)
        - **Highest renewable share of electricity:** {top_share_country} ({top_share:.1f}%)
        
        These metrics set the stage for the deeper exploration that follows.
        """.format(
            total=comp["generation"].sum(),
            hydro_share=comp.loc[comp["source"] == "Hydro", "share_percent"].iloc[0],
            top_country=country.sort_values("renewables_electricity", ascending=False)["country"].iloc[0],
            top_gen=country["renewables_electricity"].max(),
            top_share_country=country.sort_values("renewables_share_elec", ascending=False)["country"].iloc[0],
            top_share=country["renewables_share_elec"].max(),
        )
    )


def page_trends(ts: pd.DataFrame) -> None:
    """Display the trend over time of renewable electricity and its major sources."""
    st.header("Renewable Growth Over Time")
    st.markdown(
        """
        Renewables have expanded quickly in the 21st century.  The line chart below shows total
        global renewable electricity generation and its components (hydro, wind, solar, and other renewables) from 2000 onwards.
        Notice the accelerating growth, particularly after 2010, driven by rapid declines in the cost of wind and solar technologies.
        """
    )
    fig = px.line(
        ts,
        x="year",
        y=[
            "renewables_electricity",
            "hydro_electricity",
            "wind_electricity",
            "solar_electricity",
            "other_renewables_electricity",
        ],
        labels={"value": "Electricity generation (TWh)", "variable": "Source"},
    )
    fig.update_layout(legend_title_text="Source", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def page_composition(comp: pd.DataFrame, country: pd.DataFrame) -> None:
    """Display composition of renewables and geographic distribution."""
    st.header("Composition & Geographic Distribution")
    st.markdown(
        """
        The composition of renewable generation highlights which technologies dominate today, while the map shows
        where renewables are most prevalent.  Use the selector below to view different metrics on the map.  Note the
        use of a single‑hue color scale to convey magnitude (light = low, dark = high) without unnecessary embellishment.
        """
    )
    col1, col2 = st.columns(2)
    # Pie chart on left
    with col1:
        fig_pie = px.pie(
            comp,
            names="source",
            values="generation",
            hole=0,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            title_text="Renewable Generation by Source (2024)",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    # Choropleth map on right
    with col2:
        metric = st.selectbox(
            "Select map metric",
            (
                "Renewables share of electricity (%)",
                "Renewables share of primary energy (%)",
                "Renewable electricity (TWh)",
            ),
        )
        # Map metric selection
        if metric == "Renewables share of electricity (%)":
            color_col = "renewables_share_elec"
            title = "Share of Electricity from Renewables (2024)"
            color_scale = "Greens"
        elif metric == "Renewables share of primary energy (%)":
            color_col = "renewables_share_energy"
            title = "Share of Primary Energy from Renewables (2024)"
            color_scale = "Blues"
        else:
            color_col = "renewables_electricity"
            title = "Renewable Electricity Generation (TWh, 2024)"
            color_scale = "Oranges"
        # Create map
        fig_map = px.choropleth(
            country,
            locations="iso_code",
            color=color_col,
            hover_name="country",
            color_continuous_scale=color_scale,
            projection="natural earth",
        )
        fig_map.update_layout(
            title_text=title,
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_colorbar=dict(title="")
        )
        st.plotly_chart(fig_map, use_container_width=True)


def page_comparison(top: pd.DataFrame, slope: pd.DataFrame) -> None:
    """Display comparison charts (bar chart and slope graph)."""
    st.header("Country Comparisons")
    st.markdown(
        """
        Certain countries contribute disproportionately to global renewable generation, while others have made remarkable
        progress in increasing their renewable share.  The bar chart highlights the top 10 producers of renewable
        electricity in absolute terms, and the slope graph shows the change in renewable share of electricity between 2000
        and 2024 for the five countries with the largest positive change.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            top,
            x="renewables_electricity",
            y="country",
            orientation="h",
            labels={"renewables_electricity": "Renewable electricity (TWh)", "country": "Country"},
        )
        fig_bar.update_layout(title_text="Top 10 Countries by Renewable Electricity Generation (2024)")
        # Sort bars descending
        fig_bar.update_layout(yaxis_categoryorder='total ascending')
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        # Slope graph: we use line chart with markers to connect two years for each country
        fig_slope = px.line(
            slope,
            x="year",
            y="share",
            color="country",
            markers=True,
            labels={"share": "Renewables share of electricity (%)", "year": "Year", "country": "Country"},
        )
        fig_slope.update_layout(title_text="Change in Renewables Share of Electricity (2000→2024)")
        fig_slope.update_xaxes(type='category')
        st.plotly_chart(fig_slope, use_container_width=True)


def page_conclusion() -> None:
    """Display conclusion and call to action."""
    st.header("Conclusion & Call to Action")
    st.markdown(
        """
        ### Summary
        
        Renewable energy has experienced **explosive growth** in the last quarter century.  Wind and solar, virtually
        non‑existent in 2000, now comprise a significant share of total renewable generation.  Hydropower remains the
        single largest renewable source but its dominance is gradually declining as newer technologies scale.  Despite
        this progress, renewables account for only a fraction of total global energy, and adoption varies widely by
        country.

        ### What’s next?
        
        - **Accelerate deployment:** To meet climate goals, countries—especially those lagging in adoption—must continue
          to scale renewables rapidly.  Falling costs and technological improvements make this increasingly feasible.
        - **Diversify the mix:** Solar and wind will lead growth, but investing in a mix of technologies (including
          geothermal, tidal, and bioenergy) can enhance resilience and reduce dependence on any single source.
        - **Support enabling infrastructure:** Expanding grid capacity, improving storage technologies, and creating
          supportive policy frameworks are critical to integrate higher shares of renewables.

        By understanding where we stand and how far we’ve come, policymakers, investors, and citizens alike can make
        informed decisions to accelerate the energy transition.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Renewable Energy Dashboard", layout="wide")
    ts, comp, country, top, slope = load_data()
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Overview", "Trends", "Composition & Geography", "Comparisons", "Conclusion"]
    )
    if page == "Overview":
        page_overview(comp, country)
    elif page == "Trends":
        page_trends(ts)
    elif page == "Composition & Geography":
        page_composition(comp, country)
    elif page == "Comparisons":
        page_comparison(top, slope)
    else:
        page_conclusion()


if __name__ == "__main__":
    main()