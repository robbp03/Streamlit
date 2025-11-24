import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data

def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def main() -> None:
    """Build the Streamlit app layout and interactivity."""
    st.set_page_config(page_title="Sellers Dashboard", layout="wide")
    st.title("📊 Sellers Dashboard")
    st.write(
        "Use the controls below to explore the seller dataset. "
        "You can filter the table by region, visualize sales metrics, and "
        "drill down into data for a specific vendor."
    )

    # Load data once at the beginning
    df = load_data("sellers.xlsx")

    # Create a full name column for easier vendor selection
    df["FULLNAME"] = df["NAME"] + " " + df["LASTNAME"]

    # Sidebar for filters and selections
    st.sidebar.header("Filters")

    # Region filter: 'All' option plus unique regions from the dataset
    regions = ["All"] + sorted(df["REGION"].unique().tolist())
    selected_region = st.sidebar.selectbox("Select a region", regions)

    # Metric selection for graphing
    metrics = {
        "Units Sold": "SOLD UNITS",
        "Total Sales": "TOTAL SALES",
        "Average Sales": "SALES AVERAGE",
    }
    selected_metric_label = st.sidebar.radio(
        "Select metric to visualize", list(metrics.keys())
    )
    metric_column = metrics[selected_metric_label]

    sort_order = st.sidebar.checkbox(
        "Sort chart by selected metric", value=True
    )

    # Vendor selection for detailed view
    selected_vendor = st.sidebar.selectbox(
        "Select a vendor", sorted(df["FULLNAME"].unique().tolist())
    )

    # Filter dataframe by region if a specific region is chosen
    if selected_region != "All":
        filtered_df = df[df["REGION"] == selected_region]
    else:
        filtered_df = df.copy()

    # Table Section
    with st.container():
        st.subheader("📋 Data Table")
        st.caption(
            "Below is the seller dataset. Use the region filter to narrow down the rows."
        )
        st.dataframe(
            filtered_df[
                [
                    "REGION",
                    "ID",
                    "FULLNAME",
                    "INCOME",
                    "SOLD UNITS",
                    "TOTAL SALES",
                    "SALES AVERAGE",
                ]
            ],
            use_container_width=True,
        )

    # Graphs Section
    with st.container():
        st.subheader("📈 Metric Visualization")
        st.caption(
            "This bar chart displays the selected metric for each seller. "
            "You can sort the bars by value or leave them in their original order."
        )
        # Optionally sort the data by the selected metric
        if sort_order:
            display_df = filtered_df.sort_values(by=metric_column, ascending=False)
        else:
            display_df = filtered_df.copy()
        # Create bar chart using Plotly Express
        fig = px.bar(
            display_df,
            x="FULLNAME",
            y=metric_column,
            color="REGION",
            labels={
                "FULLNAME": "Vendor",
                metric_column: selected_metric_label,
            },
            title=f"{selected_metric_label} by Vendor",
        )
        # Rotate x-axis labels for better readability
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Vendor Detail Section
    with st.container():
        st.subheader("🔍 Vendor Details")
        vendor_data = df[df["FULLNAME"] == selected_vendor]
        if not vendor_data.empty:
            st.write(f"Showing data for **{selected_vendor}**:")
            st.table(
                vendor_data[
                    [
                        "REGION",
                        "ID",
                        "INCOME",
                        "SOLD UNITS",
                        "TOTAL SALES",
                        "SALES AVERAGE",
                    ]
                ]
            )
        else:
            st.info(
                "No data available for the selected vendor. Please choose another vendor."
            )


if __name__ == "__main__":
    main()