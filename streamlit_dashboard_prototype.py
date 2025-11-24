"""
Streamlit Dashboard Prototype for Whirlpool Pricing & Sales Analytics
-------------------------------------------------------------------

This dummy dashboard demonstrates how machine–learning forecasts and
related business insights can be presented to a business user.  The
layout follows best practices for clarity and decision support:

• **Main sections and panels** – A sidebar for filters, a top row of
  headline metrics for the model's predictions (quantity, revenue, DCM),
  a panel summarising model training/performance, and multiple
  supporting charts.

• **Layout** – The page uses a wide layout, with filters on the left
  and results/charts on the right.  Each section is clearly labelled
  and separated with subheaders.

• **Use of colour, placement** – Plotly express is used to create
  distinct charts; Streamlit's default styling keeps colours
  consistent and accessible.  Metric cards emphasise key
  predictions with larger font and bold type.

• **Interactivity** – Users can select a trading partner, product
  type, SKU, proposed price, week of the year and promotion flag.  The
  dummy prediction and charts respond to these selections.  In a
  real implementation, callbacks would feed the selected values into
  your trained ML model and filter the historical data accordingly.

The dummy data used here is randomly generated to demonstrate the
structure.  Replace the generation functions with your real data and
models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def generate_dummy_data():
    """Generate dummy datasets for charts."""
    # Monthly sales data for current and previous year by partner
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    partners = ["Liverpool", "Home Depot", "Chedraui", "Walmart", "Sams", "Soriana"]
    records = []
    for p in partners:
        base = np.random.randint(8000, 15000, size=len(months))
        # Simulate seasonality and noise
        current_year = base + np.random.randint(-2000, 2000, size=len(months))
        last_year = base - np.random.randint(-1500, 1500, size=len(months))
        for m, cur, prev in zip(months, current_year, last_year):
            records.append({
                'Month': m,
                'Trading Partner': p,
                'Units Sold': cur,
                'Units Sold LY': prev
            })
    sales_df = pd.DataFrame(records)

    # Price elasticity dummy data
    price_points = np.linspace(5000, 25000, 20)
    demand = 1000 - (price_points/1000) * 30 + np.random.randn(20) * 50
    elasticity_df = pd.DataFrame({
        'Price (MXN)': price_points,
        'Units Sold': demand
    })

    # Forecast error data
    error_data = np.random.normal(loc=0, scale=20, size=200)
    error_df = pd.DataFrame({
        'Error (units)': error_data
    })

    return sales_df, elasticity_df, error_df


def dummy_predict_qty(price: float, week: int, promotion: bool) -> float:
    """Simple dummy prediction function to simulate ML output.
    Assumes quantity decreases as price increases, with uplift during promotion.
    """
    baseline = 50
    price_effect = -0.002 * (price - 10000)  # Negative slope
    week_effect = 5 * np.sin(2 * np.pi * week / 52)
    promo_effect = 15 if promotion else 0
    noise = np.random.normal(0, 2)
    return max(baseline + price_effect + week_effect + promo_effect + noise, 0)


def dummy_predict_revenue(price: float, quantity: float) -> float:
    """Compute projected revenue from price and quantity."""
    return price * quantity


def dummy_predict_dcm(price: float, quantity: float) -> float:
    """Compute a dummy direct contribution margin.
    Assumes a fixed variable cost per unit (e.g., 60% of price).
    """
    variable_cost_ratio = 0.6
    return (price * (1 - variable_cost_ratio)) * quantity


def main():
    # Configure page
    st.set_page_config(page_title="Whirlpool Pricing & Sales Prototype",
                       layout="wide",
                       initial_sidebar_state="expanded")

    # Title and introduction
    st.title("Whirlpool Pricing & Sales Forecast Prototype")
    st.markdown(
        """This dashboard demonstrates how machine‑learning predictions and
        historical insights can support data‑driven pricing and inventory
        decisions. Use the controls below to explore different scenarios.
        """
    )

    # Generate dummy data
    sales_df, elasticity_df, error_df = generate_dummy_data()

    # Sidebar filters
    st.sidebar.header("Filters & Controls")
    partner_list = sales_df['Trading Partner'].unique().tolist()
    selected_partner = st.sidebar.selectbox("Trading Partner", partner_list)

    product_types = ["Refrigerator", "Washer", "Dryer", "Microwave"]
    selected_type = st.sidebar.selectbox("Product Type", product_types)

    sku_list = ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]
    selected_sku = st.sidebar.selectbox("SKU", sku_list)

    proposed_price = st.sidebar.number_input("Proposed Price (MXN)", min_value=5000.0, max_value=25000.0, value=15000.0, step=500.0)

    week_of_year = st.sidebar.slider("Week of the Year", min_value=1, max_value=52, value=26)

    is_promo = st.sidebar.checkbox("Is Promotional Week?", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Select the parameters above to see how the model responds."
                        "\n\n*The figures shown here are for demonstration only.*")

    # Perform dummy predictions based on selections
    predicted_qty = dummy_predict_qty(proposed_price, week_of_year, is_promo)
    projected_revenue = dummy_predict_revenue(proposed_price, predicted_qty)
    projected_dcm = dummy_predict_dcm(proposed_price, predicted_qty)

    # Top row: display headline metrics
    col_q, col_rev, col_dcm = st.columns(3)
    col_q.metric("Predicted Quantity (units)", f"{predicted_qty:.1f}")
    col_rev.metric("Projected Revenue (MXN)", f"${projected_revenue:,.0f}")
    col_dcm.metric("Projected DCM (MXN)", f"${projected_dcm:,.0f}")

    # Section: Model training & performance
    st.subheader("Model Training & Performance")
    st.write("""
        Our quantity‑forecasting model is trained on weekly sales data for each
        SKU and trading partner. We evaluate models using Root Mean Squared
        Error (RMSE) and R² on a hold‑out test set. Below are example
        performance metrics for dummy models:
    """)
    perf_data = pd.DataFrame({
        'Model': ['Quantity Model', 'Price Model', 'Inventory Model'],
        'Algorithm': ['LightGBM', 'Random Forest', 'XGBoost'],
        'R²': [0.62, 0.85, 0.75],
        'RMSE': [48.7, 385.7, 0.60]
    })
    st.dataframe(perf_data.set_index('Model'))
    st.write("*Lower RMSE indicates better prediction accuracy. R² closer to 1 means the model explains more variance in the data.*")

    # Filter sales data by selected partner for demonstration
    partner_sales = sales_df[sales_df['Trading Partner'] == selected_partner]

    # Sales history chart
    st.subheader("Sales History by Month")
    fig_sales = px.line(partner_sales, x='Month', y=['Units Sold', 'Units Sold LY'],
                        labels={
                            'value': 'Units Sold',
                            'Month': 'Month',
                            'variable': 'Year'
                        },
                        title=f"Monthly Units Sold – {selected_partner}")
    st.plotly_chart(fig_sales, use_container_width=True)

    # Price elasticity chart
    st.subheader("Price vs Units Sold")
    fig_elasticity = px.scatter(elasticity_df, x='Price (MXN)', y='Units Sold', trendline='ols',
                                labels={'Price (MXN)': 'Price (MXN)', 'Units Sold': 'Units Sold'},
                                title="Price Elasticity (Dummy Data)")
    st.plotly_chart(fig_elasticity, use_container_width=True)

    # Trading partner comparison chart (bar)
    st.subheader("Volume by Trading Partner")
    volume_summary = sales_df.groupby('Trading Partner')['Units Sold'].sum().reset_index()
    fig_volume = px.bar(volume_summary, x='Trading Partner', y='Units Sold',
                        labels={'Units Sold': 'Units Sold', 'Trading Partner': 'Trading Partner'},
                        title="Cumulative Units Sold per Trading Partner")
    st.plotly_chart(fig_volume, use_container_width=True)

    # Forecast error distribution
    st.subheader("Forecast Error Distribution")
    fig_error = px.histogram(error_df, x='Error (units)', nbins=30,
                             labels={'Error (units)': 'Prediction Error (units)'},
                             title="Distribution of Forecast Errors (Dummy)")
    st.plotly_chart(fig_error, use_container_width=True)

    # Final note on usage and limitations
    st.markdown("""
    ## Interpretation & Next Steps
    The predicted quantity, revenue and DCM values above are based on a
    simple placeholder formula. In a production version, these would be
    generated by the trained machine‑learning models. Use the additional
    charts to cross‑check the plausibility of the recommendation: for
    example, compare the proposed price against historical price–demand
    patterns and overall volumes by trading partner. Always consider
    business context (promotions, inventory constraints) when
    interpreting the model's output.
    """)


if __name__ == "__main__":
    main()