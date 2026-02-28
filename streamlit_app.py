"""P2P Pricing Mechanism Recommender - Streamlit Web App.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from p2p_pricing import (
    recommend,
    SimulationConfig,
    RecommendationWeights,
    TariffConfig,
)


st.set_page_config(
    page_title="P2P Pricing Recommender",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ P2P Energy Pricing Mechanism Recommender")
st.markdown("""
This tool recommends the best peer-to-peer energy pricing mechanism based on your
community composition and priorities.
""")

# Sidebar - Inputs
st.sidebar.header("Community Composition")

num_consumers = st.sidebar.slider(
    "Consumers (no PV)",
    min_value=0, max_value=20, value=5,
    help="Households without solar panels"
)

num_small_prosumers = st.sidebar.slider(
    "Small Prosumers (40-60% PV)",
    min_value=0, max_value=20, value=3,
    help="PV sized for self-consumption, rarely exports"
)

num_medium_prosumers = st.sidebar.slider(
    "Medium Prosumers (80-110% PV)",
    min_value=0, max_value=20, value=4,
    help="Balanced PV, can export moderate amounts"
)

num_large_prosumers = st.sidebar.slider(
    "Large Prosumers (130-180% PV)",
    min_value=0, max_value=20, value=3,
    help="Large PV systems, significant exporters"
)

total_households = num_consumers + num_small_prosumers + num_medium_prosumers + num_large_prosumers

if total_households == 0:
    st.error("Please select at least one household.")
    st.stop()

# Sidebar - Priority Weights
st.sidebar.header("Priority Weights")
st.sidebar.markdown("Adjust what matters most to your community:")

w_cost = st.sidebar.slider("Cost Savings", 0.0, 1.0, 0.4, 0.1)
w_fairness = st.sidebar.slider("Fairness", 0.0, 1.0, 0.3, 0.1)
w_stability = st.sidebar.slider("Bill Stability", 0.0, 1.0, 0.3, 0.1)

# Sidebar - Tariff Configuration
st.sidebar.header("Grid Tariff Settings")

tariff_mode = st.sidebar.radio(
    "Tariff Mode",
    ["Default (Constant)", "Custom Constant", "Random Range"],
    help="How grid buy prices are assigned to households"
)

tariff_config = None
grid_buy_price = 6.0
grid_sell_price = 3.4

if tariff_mode == "Custom Constant":
    consumer_buy = st.sidebar.number_input("Consumer Buy Price (Rs/kWh)", 3.0, 15.0, 6.0, 0.5)
    prosumer_buy = st.sidebar.number_input("Prosumer Buy Price (Rs/kWh)", 3.0, 15.0, 6.0, 0.5)
    sell_price = st.sidebar.number_input("Grid Sell Price (Rs/kWh)", 1.0, 10.0, 3.4, 0.1)

    tariff_config = TariffConfig(
        consumer_buy_mode="constant",
        prosumer_buy_mode="constant",
        consumer_buy_price=consumer_buy,
        prosumer_buy_price=prosumer_buy,
        grid_sell_price=sell_price
    )
    grid_buy_price = max(consumer_buy, prosumer_buy)
    grid_sell_price = sell_price

elif tariff_mode == "Random Range":
    st.sidebar.markdown("**Consumer Buy Range:**")
    consumer_low = st.sidebar.number_input("Min (Rs/kWh)", 3.0, 15.0, 5.0, 0.5, key="c_low")
    consumer_high = st.sidebar.number_input("Max (Rs/kWh)", 3.0, 15.0, 7.0, 0.5, key="c_high")

    st.sidebar.markdown("**Prosumer Buy Range:**")
    prosumer_low = st.sidebar.number_input("Min (Rs/kWh)", 3.0, 15.0, 5.0, 0.5, key="p_low")
    prosumer_high = st.sidebar.number_input("Max (Rs/kWh)", 3.0, 15.0, 7.0, 0.5, key="p_high")

    sell_price = st.sidebar.number_input("Grid Sell Price (Rs/kWh)", 1.0, 10.0, 3.4, 0.1, key="sell")

    tariff_config = TariffConfig(
        consumer_buy_mode="random",
        prosumer_buy_mode="random",
        consumer_buy_range=(consumer_low, consumer_high),
        prosumer_buy_range=(prosumer_low, prosumer_high),
        grid_sell_price=sell_price
    )
    grid_buy_price = max(consumer_high, prosumer_high)
    grid_sell_price = sell_price

# Create config
config = SimulationConfig(
    grid_buy_price=grid_buy_price,
    grid_sell_price=grid_sell_price,
    tariff_config=tariff_config
)

weights = RecommendationWeights(
    cost_savings=w_cost,
    fairness=w_fairness,
    stability=w_stability
)


@st.cache_data
def get_recommendation(_config_hash, num_c, num_sp, num_mp, num_lp, w_c, w_f, w_s, seed=42):
    """Cached recommendation computation."""
    config = SimulationConfig(
        grid_buy_price=grid_buy_price,
        grid_sell_price=grid_sell_price,
        tariff_config=tariff_config
    )
    weights = RecommendationWeights(cost_savings=w_c, fairness=w_f, stability=w_s)

    return recommend(
        num_consumers=num_c,
        num_small_prosumers=num_sp,
        num_medium_prosumers=num_mp,
        num_large_prosumers=num_lp,
        weights=weights,
        config=config,
        seed=seed
    )


# Generate config hash for caching
config_hash = f"{tariff_mode}_{grid_buy_price}_{grid_sell_price}"
if tariff_config:
    config_hash += f"_{tariff_config.consumer_buy_mode}_{tariff_config.prosumer_buy_mode}"

# Run recommendation
with st.spinner("Running simulation..."):
    result = get_recommendation(
        config_hash,
        num_consumers, num_small_prosumers, num_medium_prosumers, num_large_prosumers,
        w_cost, w_fairness, w_stability
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Recommendation")

    # Recommendation card
    st.success(f"### Recommended: **{result.recommended_mechanism}**")
    st.markdown(result.reasoning)

    # Scores bar chart
    st.subheader("Mechanism Scores")
    scores_df = pd.DataFrame({
        'Mechanism': list(result.scores.keys()),
        'Score': list(result.scores.values())
    }).sort_values('Score', ascending=True)

    fig_scores = px.bar(
        scores_df, x='Score', y='Mechanism',
        orientation='h',
        color='Score',
        color_continuous_scale='Greens'
    )
    fig_scores.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_scores, use_container_width=True)

with col2:
    st.header("Community Summary")
    st.metric("Total Households", total_households)

    # Pie chart of household composition
    comp_data = {
        'Type': ['Consumers', 'Small Prosumers', 'Medium Prosumers', 'Large Prosumers'],
        'Count': [num_consumers, num_small_prosumers, num_medium_prosumers, num_large_prosumers]
    }
    comp_df = pd.DataFrame(comp_data)
    comp_df = comp_df[comp_df['Count'] > 0]

    fig_comp = px.pie(
        comp_df, values='Count', names='Type',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_comp.update_layout(height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

# Detailed Metrics
st.header("Detailed Comparison")

# Create comparison table
metrics_data = []
for name, m in result.metrics.mechanisms.items():
    metrics_data.append({
        'Mechanism': name,
        'Total Cost (Rs)': f"{m.total_cost:,.0f}",
        'Savings vs Conventional': f"{m.cost_savings_pct:.1f}%",
        'Consumer Savings': f"{m.consumer_savings_pct:.1f}%",
        'Prosumer Savings': f"{m.prosumer_savings_pct:.1f}%",
        'Fairness Index': f"{m.fairness_index:.2f}",
        'Daily Volatility (Rs)': f"{m.daily_cost_volatility:.2f}"
    })

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Charts row
col3, col4 = st.columns(2)

with col3:
    st.subheader("Total Costs by Mechanism")
    costs = {name: m.total_cost for name, m in result.metrics.mechanisms.items()}
    costs_df = pd.DataFrame({
        'Mechanism': list(costs.keys()),
        'Total Cost (Rs)': list(costs.values())
    })

    fig_costs = px.bar(
        costs_df, x='Mechanism', y='Total Cost (Rs)',
        color='Mechanism',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_costs.update_layout(showlegend=False)
    st.plotly_chart(fig_costs, use_container_width=True)

with col4:
    st.subheader("Savings by Group")
    p2p_mechanisms = [n for n in result.metrics.mechanisms if n != 'Conventional']

    savings_data = []
    for name in p2p_mechanisms:
        m = result.metrics.mechanisms[name]
        savings_data.append({
            'Mechanism': name,
            'Group': 'Consumers',
            'Savings (%)': m.consumer_savings_pct
        })
        savings_data.append({
            'Mechanism': name,
            'Group': 'Prosumers',
            'Savings (%)': m.prosumer_savings_pct
        })

    savings_df = pd.DataFrame(savings_data)
    fig_savings = px.bar(
        savings_df, x='Mechanism', y='Savings (%)',
        color='Group', barmode='group',
        color_discrete_sequence=['#2ecc71', '#3498db']
    )
    st.plotly_chart(fig_savings, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Mechanisms Explained:**
- **Conventional**: Traditional grid import/export at fixed tariffs
- **MMR (Mid-Market Rate)**: Dynamic P2P price based on supply-demand balance
- **Bill-Sharing**: Energy shared free within community; costs distributed ex-post
- **SDR (Supply-Demand Ratio)**: Incentive-based pricing encouraging supply-demand matching
""")