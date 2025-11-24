"""
Streamlit Dashboard Prototype for Whirlpool Pricing & Sales Analytics
--------------------------------------------------------------------

This updated prototype arranges the first four graphs in two side-by-side rows for a more
compact and comparative layout. It uses Plotly for visualization and a refined color palette.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Custom Color Palette ---
COLOR_PALETTE = ['#2A4D69', '#4B86B4', '#ADCBE3', '#E7EFF6', '#FFCB9A', '#E38471']

def generate_dummy_data():
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    partners = ["Liverpool","Home Depot","Chedraui","Walmart","Sams","Soriana"]
    records = []
    for p in partners:
        base = np.random.randint(8000, 15000, size=len(months))
        current = base + np.random.randint(-2000, 2000, size=len(months))
        last = base - np.random.randint(-1500, 1500, size=len(months))
        for m, cur, prev in zip(months, current, last):
            records.append({"Month": m, "Trading Partner": p, "Units Sold": cur, "Units Sold LY": prev})
    sales_df = pd.DataFrame(records)

    price_points = np.linspace(5000, 25000, 30)
    demand = 1000 - (price_points/1000) * 30 + np.random.randn(30) * 50
    elasticity_df = pd.DataFrame({'Price (MXN)': price_points, 'Units Sold': demand})

    error_data = np.random.normal(loc=0, scale=20, size=200)
    error_df = pd.DataFrame({'Error (units)': error_data})
    return sales_df, elasticity_df, error_df

def dummy_predict_qty(price: float, week: int, promotion: bool) -> float:
    base = 50
    price_effect = -0.002 * (price - 10000)
    week_effect = 5 * np.sin(2 * np.pi * week / 52)
    promo_effect = 15 if promotion else 0
    noise = np.random.normal(0, 2)
    return max(base + price_effect + week_effect + promo_effect + noise, 0)

def dummy_predict_revenue(price: float, qty: float) -> float:
    return price * qty

def dummy_predict_dcm(price: float, qty: float) -> float:
    variable_cost_ratio = 0.6
    return (price * (1 - variable_cost_ratio)) * qty

def main():
    st.set_page_config(page_title="Whirlpool ML Dashboard Prototype", layout="wide")
    st.title("Whirlpool – Price Elasticity Dashboard Prototype")

    sales_df, elasticity_df, error_df = generate_dummy_data()

    st.sidebar.header("User Inputs")
    partner = st.sidebar.selectbox("Trading Partner", sales_df['Trading Partner'].unique())
    product = st.sidebar.selectbox("Product Type", ["Refrigerator", "Washer", "Dryer", "Microwave"])
    sku = st.sidebar.selectbox("SKU", ["SKU-001", "SKU-002", "SKU-003", "SKU-004"])
    price = st.sidebar.number_input("Proposed Price (MXN)", min_value=5000.0, max_value=25000.0, value=15000.0, step=500.0)
    week = st.sidebar.slider("Week of Year", 1, 52, 26)
    promo = st.sidebar.checkbox("Promotional Week?", False)

    pred_qty = dummy_predict_qty(price, week, promo)
    pred_rev = dummy_predict_revenue(price, pred_qty)
    pred_dcm = dummy_predict_dcm(price, pred_qty)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Units", f"{pred_qty:.1f}")
    col2.metric("Projected Revenue", f"${pred_rev:,.0f}")
    col3.metric("Projected DCM", f"${pred_dcm:,.0f}")

    st.subheader("Model Performance (Elasticity Model)")
    best_model_df = pd.DataFrame({
        'Model': ['Elasticity (Best)'],
        'Algorithm': ['LightGBM'],
        'R²': [0.62],
        'RMSE': [48.7]
    })
    st.dataframe(best_model_df.set_index('Model'))

    partner_sales = sales_df[sales_df['Trading Partner'] == partner]

    # --- Display First Four Graphs Side-by-Side ---
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.plotly_chart(px.line(partner_sales, x='Month', y=['Units Sold', 'Units Sold LY'],
                                labels={'value': 'Units', 'Month': 'Month', 'variable': 'Year'},
                                title=f"Monthly Sales History – {partner}",
                                color_discrete_sequence=COLOR_PALETTE),
                        use_container_width=True)
    with col_a2:
        st.plotly_chart(px.scatter(elasticity_df, x='Price (MXN)', y='Units Sold',
                                   trendline='ols',
                                   labels={'Price (MXN)': 'Price (MXN)', 'Units Sold': 'Units'},
                                   title="Price vs Demand Elasticity",
                                   color_discrete_sequence=[COLOR_PALETTE[1]]),
                        use_container_width=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        volume_summary = sales_df.groupby('Trading Partner')['Units Sold'].sum().reset_index()
        st.plotly_chart(px.bar(volume_summary, x='Trading Partner', y='Units Sold',
                               labels={'Units Sold': 'Total Units', 'Trading Partner': 'Partner'},
                               title="Total Volume per Partner",
                               color_discrete_sequence=[COLOR_PALETTE[2]]),
                        use_container_width=True)
    with col_b2:
        st.plotly_chart(px.histogram(error_df, x='Error (units)', nbins=30,
                                     labels={'Error (units)': 'Prediction Error (Units)'},
                                     title="Forecast Error Distribution",
                                     color_discrete_sequence=[COLOR_PALETTE[3]]),
                        use_container_width=True)

    st.subheader("Revenue Curve vs Price")
    price_range = np.linspace(5000, 25000, 40)
    qty_preds = [dummy_predict_qty(p, week, promo) for p in price_range]
    revenues = price_range * qty_preds
    rev_df = pd.DataFrame({'Price': price_range, 'Revenue': revenues})
    st.plotly_chart(px.line(rev_df, x='Price', y='Revenue',
                            labels={'Price': 'Price (MXN)', 'Revenue': 'Revenue (MXN)'},
                            title="Projected Revenue Curve",
                            color_discrete_sequence=[COLOR_PALETTE[4]]),
                    use_container_width=True)

if __name__ == "__main__":
    main()