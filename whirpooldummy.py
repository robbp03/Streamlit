"""
Streamlit Dashboard Prototype for Whirlpool Pricing & Sales Analytics
--------------------------------------------------------------------

Este dashboard de ejemplo visualiza cómo se podrían mostrar las predicciones del modelo
de elasticidad junto a métricas de entrenamiento y otros gráficos de soporte.  Se usa
una paleta de colores personalizada y se incluyen gráficos adicionales (historial de
ventas, elasticidad, volumen por socio y curva de ingresos).

Filtros: socio comercial, tipo de producto, SKU, precio propuesto, semana del año y si
es una semana promocional.  Las predicciones simuladas se actualizan con estos
parámetros.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Paleta de colores personalizada inspirada en las presentaciones adjuntas
COLOR_PALETTE = ['#2A4D69', '#4B86B4', '#ADCBE3', '#E7EFF6', '#FFCB9A', '#E38471']

def generate_dummy_data():
    """Genera datos de ejemplo para los gráficos."""
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
    """Predicción de cantidad simplificada."""
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
    st.set_page_config(page_title="Whirlpool ML Dashboard Prototype",
                       layout="wide",
                       initial_sidebar_state="expanded")

    st.title("Whirlpool – Prototipo de Dashboard de Elasticidad de Precio")

    st.markdown("""
    Este panel muestra una vista integrada de las predicciones de nuestro modelo de elasticidad
    junto con información histórica y métricas de entrenamiento. Ajusta los filtros del lateral
    para simular distintos escenarios de precio, semana y promociones.
    """)

    # Datos de ejemplo
    sales_df, elasticity_df, error_df = generate_dummy_data()

    # --- Controles en la barra lateral ---
    st.sidebar.header("Filtros y Controles")
    partner = st.sidebar.selectbox("Trading Partner", sales_df['Trading Partner'].unique())
    product_types = ["Refrigerator", "Washer", "Dryer", "Microwave"]
    product = st.sidebar.selectbox("Tipo de Producto", product_types)
    sku_list = ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]
    sku = st.sidebar.selectbox("SKU", sku_list)
    price = st.sidebar.number_input("Precio Propuesto (MXN)", min_value=5000.0, max_value=25000.0, value=15000.0, step=500.0)
    week = st.sidebar.slider("Semana del Año", 1, 52, value=26)
    promo = st.sidebar.checkbox("¿Semana Promocional?", False)
    st.sidebar.markdown("---")
    st.sidebar.info("Estos valores se alimentan al modelo para obtener las predicciones.")

    # Predicciones simuladas
    pred_qty = dummy_predict_qty(price, week, promo)
    pred_rev = dummy_predict_revenue(price, pred_qty)
    pred_dcm = dummy_predict_dcm(price, pred_qty)

    col1, col2, col3 = st.columns(3)
    col1.metric("Cantidad Predicha (unidades)", f"{pred_qty:.1f}")
    col2.metric("Ingresos Proyectados (MXN)", f"${pred_rev:,.0f}")
    col3.metric("DCM Proyectado (MXN)", f"${pred_dcm:,.0f}")

    # --- Métricas del modelo (solo mejor modelo) ---
    st.subheader("Métricas del Modelo de Elasticidad")
    best_model_df = pd.DataFrame({
        'Modelo': ['Elasticidad (mejor)'],
        'Algoritmo': ['LightGBM'],
        'R²': [0.62],
        'RMSE': [48.7]
    })
    st.dataframe(best_model_df.set_index('Modelo'))
    st.caption("Se muestra solo el mejor modelo seleccionado para las predicciones de elasticidad.")

    # Filtrar historial de ventas para el socio seleccionado
    partner_sales = sales_df[sales_df['Trading Partner'] == partner]

    # --- Gráficos ---
    st.subheader("Historial de Ventas Mensuales")
    fig_sales = px.line(partner_sales, x='Month', y=['Units Sold', 'Units Sold LY'],
                        labels={'value': 'Unidades', 'Month': 'Mes', 'variable': 'Año'},
                        title=f"Unidades Vendidas – {partner}",
                        color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig_sales, use_container_width=True)

    st.subheader("Elasticidad Precio–Unidades")
    fig_elasticity = px.scatter(elasticity_df, x='Price (MXN)', y='Units Sold',
                                trendline='ols',
                                labels={'Price (MXN)': 'Precio (MXN)', 'Units Sold': 'Unidades'},
                                title="Relación entre Precio y Demanda (Ejemplo)",
                                color_discrete_sequence=[COLOR_PALETTE[1]])
    st.plotly_chart(fig_elasticity, use_container_width=True)

    st.subheader("Volumen Total por Socio Comercial")
    volume_summary = sales_df.groupby('Trading Partner')['Units Sold'].sum().reset_index()
    fig_volume = px.bar(volume_summary, x='Trading Partner', y='Units Sold',
                        labels={'Units Sold': 'Unidades Totales', 'Trading Partner': 'Socio'},
                        title="Volumen Acumulado por Socio",
                        color_discrete_sequence=[COLOR_PALETTE[2]])
    st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader("Distribución del Error de Pronóstico")
    fig_error = px.histogram(error_df, x='Error (units)',
                             nbins=30,
                             labels={'Error (units)': 'Error de Predicción (unidades)'},
                             title="Distribución de Errores del Modelo",
                             color_discrete_sequence=[COLOR_PALETTE[3]])
    st.plotly_chart(fig_error, use_container_width=True)

    # Gráfico adicional: curva de ingresos vs precio
    st.subheader("Curva de Ingresos vs Precio")
    price_range = np.linspace(5000, 25000, 40)
    qty_preds = [dummy_predict_qty(p, week, promo) for p in price_range]
    revenues = price_range * qty_preds
    rev_df = pd.DataFrame({'Precio': price_range, 'Ingresos': revenues})
    fig_revenue = px.line(rev_df, x='Precio', y='Ingresos',
                          labels={'Precio': 'Precio (MXN)', 'Ingresos': 'Ingresos (MXN)'},
                          title="Ingresos Proyectados según Precio",
                          color_discrete_sequence=[COLOR_PALETTE[4]])
    st.plotly_chart(fig_revenue, use_container_width=True)

    st.markdown("""
    ### Interpretación y Consejos

    * Utiliza la **Curva de Ingresos** para explorar cómo variaciones en el precio pueden afectar
      tus ingresos esperados según la predicción del modelo.
    * La **Elasticidad** muestra la relación estimada entre precio y demanda; observa la tendencia
      para entender si los clientes son sensibles a cambios de precio.
    * El **Historial de Ventas** te permite comparar el desempeño actual frente al año anterior,
      ayudando a contextualizar la recomendación del modelo.
    * Revisa la **Distribución de Errores** para tener una idea de la variabilidad del modelo;
      recuerda que predicciones con mucho error deben considerarse con precaución.
    """)

if __name__ == "__main__":
    main()
