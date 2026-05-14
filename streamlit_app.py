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
    layout="wide"
)

AXIS_FONT = dict(color='#1a1a1a', size=13)
TITLE_FONT = dict(color='#1a1a1a', size=15)
TICK_FONT = dict(color='#333333', size=11)


def style_fig(fig):
    """Apply dark axis labels and ticks to a plotly figure."""
    layout_update = dict(
        xaxis_title_font=AXIS_FONT,
        yaxis_title_font=AXIS_FONT,
        xaxis_tickfont=TICK_FONT,
        yaxis_tickfont=TICK_FONT,
    )
    if fig.layout.title and fig.layout.title.text:
        layout_update['title_font'] = TITLE_FONT
    fig.update_layout(**layout_update)
    return fig

st.title("P2P Energy Pricing Mechanism Recommender")
st.markdown("""
This tool recommends the best peer-to-peer energy pricing mechanism based on your
community composition and priorities.
""")
st.title("P2P Energy Pricing Dashboard")

# Initialize session state for simulation
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "validation_errors" not in st.session_state:
    st.session_state.validation_errors = []
if "last_run_params" not in st.session_state:
    st.session_state.last_run_params = None

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
st.sidebar.markdown("Adjust what matters most to your community. The three weights must add up to exactly 1.0.")

w_cost = st.sidebar.slider("Cost Savings", 0.0, 1.0, 0.4, 0.01)
w_fairness = st.sidebar.slider("Fairness", 0.0, 1.0, 0.3, 0.01)
w_stability = st.sidebar.slider("Bill Stability", 0.0, 1.0, 0.3, 0.01)

# Display the actual weights and their sum
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Weights:**")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.metric("Cost", f"{w_cost:.2f}")
with col2:
    st.metric("Fairness", f"{w_fairness:.2f}")
with col3:
    st.metric("Stability", f"{w_stability:.2f}")

weight_sum = w_cost + w_fairness + w_stability
if abs(weight_sum - 1.0) < 0.001:
    st.sidebar.success(f"✓ Sum = {weight_sum:.3f}")
else:
    st.sidebar.error(f"✗ Sum = {weight_sum:.3f} (must be 1.0)")

# Sidebar - DSM Settings
st.sidebar.header("DSM Settings")
dsm_alpha = st.sidebar.slider(
    "Discomfort Coefficient (α)",
    min_value=0.1, max_value=50.0, value=0.12, step=0.1,
    help="Higher α = less load shifting (more inconvenience cost). At α>30, DSM behaves like plain SDR. Lower α = more aggressive shifting."
)

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

# Sidebar - Run Simulation Button
st.sidebar.markdown("---")
st.sidebar.header("Run Simulation")

def validate_inputs():
    """Validate all inputs before running simulation."""
    errors = []
    
    # Check minimum community size
    if total_households < 2:
        errors.append("❌ Community must have at least 2 households (minimum 1 consumer + 1 prosumer)")
    
    # Check that we have prosumers
    has_prosumer = (num_small_prosumers + num_medium_prosumers + num_large_prosumers) > 0
    if not has_prosumer:
        errors.append("❌ Community must have at least 1 prosumer")
    
    # Check weights sum to exactly 1.0
    weight_sum = w_cost + w_fairness + w_stability
    if abs(weight_sum - 1.0) > 0.001:
        errors.append(f"❌ Priority weights must sum to exactly 1.0 (currently {weight_sum:.3f})")
    
    return errors

# Run validation to determine button state
validation_errors = validate_inputs()
button_disabled = len(validation_errors) > 0
button_type = "secondary" if button_disabled else "primary"
button_label = "⚠ Fix errors above to run" if button_disabled else "▶ Run Simulation"

if st.sidebar.button(button_label, use_container_width=True, type=button_type, disabled=button_disabled):
    st.session_state.validation_errors = []
    st.session_state.simulation_run = True
    st.session_state.last_run_params = None  # Reset to allow new run with new params

# Display validation errors in sidebar if any
if validation_errors:
    for error in validation_errors:
        st.sidebar.error(error)

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
def get_recommendation(_config_hash, num_c, num_sp, num_mp, num_lp, w_c, w_f, w_s, alpha=0.12, seed=42):
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
        seed=seed,
        dsm_alpha=alpha
    )


# Generate config hash for caching
config_hash = f"{tariff_mode}_{grid_buy_price}_{grid_sell_price}_{dsm_alpha}"
if tariff_config:
    config_hash += f"_{tariff_config.consumer_buy_mode}_{tariff_config.prosumer_buy_mode}"

# Create current parameters snapshot
current_params = {
    "num_consumers": num_consumers,
    "num_small_prosumers": num_small_prosumers,
    "num_medium_prosumers": num_medium_prosumers,
    "num_large_prosumers": num_large_prosumers,
    "w_cost": w_cost,
    "w_fairness": w_fairness,
    "w_stability": w_stability,
    "dsm_alpha": dsm_alpha,
    "tariff_mode": tariff_mode,
    "grid_buy_price": grid_buy_price,
    "grid_sell_price": grid_sell_price
}

# Check if parameters have changed since last run (only if there was a previous run)
if st.session_state.simulation_run and st.session_state.last_run_params is not None:
    if st.session_state.last_run_params != current_params:
        st.session_state.simulation_run = False
        st.session_state.validation_errors = ["⚠️ Parameters changed. Click 'Run Simulation' to update results."]

# Display validation errors or run simulation
if st.session_state.simulation_run:
    # Store parameters as last run params
    st.session_state.last_run_params = current_params
    
    # Run recommendation
    with st.spinner("Running simulation..."):
        result = get_recommendation(
            config_hash,
            num_consumers, num_small_prosumers, num_medium_prosumers, num_large_prosumers,
            w_cost, w_fairness, w_stability, dsm_alpha
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
    st.plotly_chart(style_fig(fig_scores), use_container_width=True)

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
    st.plotly_chart(style_fig(fig_comp), use_container_width=True)

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
    st.plotly_chart(style_fig(fig_costs), use_container_width=True)

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
    st.plotly_chart(style_fig(fig_savings), use_container_width=True)

    # SDR Analysis Section
    st.header("SDR Analysis (with DSM)")
    st.markdown("Supply-Demand Ratio pricing with demand-side management: load reshaping and dynamic P2P pricing.")

    # Access the settlement results for DSM-specific plots
    @st.cache_data
    def get_dsm_data(_config_hash, num_c, num_sp, num_mp, num_lp, alpha=0.12, seed=42):
        """Get raw settlement results for DSM visualizations."""
        from p2p_pricing import generate_all_profiles, run_all_settlements, SimulationConfig
        sim_config = SimulationConfig(
            grid_buy_price=grid_buy_price,
            grid_sell_price=grid_sell_price,
            tariff_config=tariff_config
        )
        profiles = generate_all_profiles(
            num_consumers=num_c,
            num_small_prosumers=num_sp,
            num_medium_prosumers=num_mp,
            num_large_prosumers=num_lp,
            config=sim_config,
            seed=seed
        )
        profiles['dsm_alpha'] = alpha
        settlements = run_all_settlements(profiles)
        return profiles, settlements

    profiles_data, settlements_data = get_dsm_data(
        config_hash, num_consumers, num_small_prosumers,
        num_medium_prosumers, num_large_prosumers, dsm_alpha
    )

    if 'SDR' in settlements_data and settlements_data['SDR'].extra_data:
        dsm_data = settlements_data['SDR'].extra_data

        L_ref = dsm_data['L_ref']
        L_adj = dsm_data['L_adj']
        PV = profiles_data['PV']
        Pr_buy = dsm_data['Pr_buy']
        Pr_sell = dsm_data['Pr_sell']
        SDR_ts = dsm_data['SDR']
        periods_per_day = 96
        days = 30

        def hourly_avg(ts):
            reshaped = ts.reshape(days, periods_per_day)
            hourly = np.zeros((days, 24))
            for h in range(24):
                hourly[:, h] = reshaped[:, h*4:(h+1)*4].mean(axis=1)
            return hourly.mean(axis=0)

        hours = list(range(24))

        dsm_tab1, dsm_tab2, dsm_tab3, dsm_tab4 = st.tabs([
            "Aggregate Profiles", "Load Reshaping", "SDR Dynamics", "P2P Prices"
        ])

        with dsm_tab1:
            L_ref_total = np.sum(L_ref, axis=0)
            L_adj_total = np.sum(L_adj, axis=0)
            PV_total = np.sum(PV, axis=0)

        L_ref_h = hourly_avg(L_ref_total)
        L_adj_h = hourly_avg(L_adj_total)
        PV_h = hourly_avg(PV_total)

        # Compute excess PV and deficit regions for shaded areas
        excess_pv = np.maximum(PV_h - L_adj_h, 0)
        deficit = np.maximum(L_adj_h - PV_h, 0)

        # Smooth curves using interpolation for nicer look
        from scipy.interpolate import make_interp_spline
        hours_fine = np.linspace(0, 23, 200)
        try:
            spl_ref = make_interp_spline(hours, L_ref_h, k=3)
            spl_adj = make_interp_spline(hours, L_adj_h, k=3)
            spl_pv = make_interp_spline(hours, PV_h, k=3)
            L_ref_smooth = spl_ref(hours_fine)
            L_adj_smooth = spl_adj(hours_fine)
            PV_smooth = spl_pv(hours_fine)
        except Exception:
            hours_fine = np.array(hours, dtype=float)
            L_ref_smooth = L_ref_h
            L_adj_smooth = L_adj_h
            PV_smooth = PV_h

        excess_smooth = np.maximum(PV_smooth - L_adj_smooth, 0)
        deficit_smooth = np.maximum(L_adj_smooth - PV_smooth, 0)

        fig_agg = go.Figure()

        # Deficit fill (between adjusted load and PV where load > PV)
        fig_agg.add_trace(go.Scatter(
            x=np.concatenate([hours_fine, hours_fine[::-1]]),
            y=np.concatenate([L_adj_smooth, np.minimum(PV_smooth, L_adj_smooth)[::-1]]),
            fill='toself', fillcolor='rgba(255, 150, 150, 0.4)',
            line=dict(width=0), name='Deficit', showlegend=True
        ))

        # Excess PV fill (between PV and adjusted load where PV > load)
        fig_agg.add_trace(go.Scatter(
            x=np.concatenate([hours_fine, hours_fine[::-1]]),
            y=np.concatenate([PV_smooth, np.minimum(PV_smooth, L_adj_smooth)[::-1]]),
            fill='toself', fillcolor='rgba(144, 238, 144, 0.4)',
            line=dict(width=0), name='Excess PV', showlegend=True
        ))

        # PV Generation line
        fig_agg.add_trace(go.Scatter(
            x=hours_fine, y=PV_smooth, mode='lines', name='Total PV Generation',
            line=dict(color='green', width=2.5)
        ))

        # Reference Load (dashed gray)
        fig_agg.add_trace(go.Scatter(
            x=hours_fine, y=L_ref_smooth, mode='lines', name='Total Reference Load',
            line=dict(color='gray', width=2, dash='dash')
        ))

        # Adjusted Load (solid black)
        fig_agg.add_trace(go.Scatter(
            x=hours_fine, y=L_adj_smooth, mode='lines', name='Total Adjusted Load',
            line=dict(color='black', width=2.5)
        ))

        fig_agg.update_layout(
            title="Community Aggregate Power Profiles",
            xaxis_title="Hour of Day", yaxis_title="Power (kW)",
            height=450,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            xaxis=dict(dtick=5, range=[0, 24]),
            yaxis=dict(rangemode='tozero'),
            plot_bgcolor='white',
            xaxis_showgrid=True, yaxis_showgrid=True,
            xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
        )
        st.plotly_chart(style_fig(fig_agg), use_container_width=True)

            shift_energy = np.sum(np.abs(L_adj_total - L_ref_total)) * 0.25 / 2
            st.metric("Total Energy Shifted (kWh/day avg)", f"{shift_energy/days:.2f}")

        with dsm_tab2:
            N = L_ref.shape[0]
        num_c = profiles_data.get('num_consumers', 0)

        # Consumer profiles
        st.markdown("**Sample Consumers (no PV)**")
        consumer_indices = list(range(min(3, num_c)))
        if consumer_indices:
            cols = st.columns(len(consumer_indices))
            for col_idx, hh_idx in enumerate(consumer_indices):
                ref_h = hourly_avg(L_ref[hh_idx])
                adj_h = hourly_avg(L_adj[hh_idx])

                fig_hh = go.Figure()
                # Fill between reference and adjusted to show shift
                fig_hh.add_trace(go.Scatter(
                    x=hours + hours[::-1],
                    y=list(ref_h) + list(adj_h)[::-1],
                    fill='toself', fillcolor='rgba(99, 110, 250, 0.12)',
                    line=dict(width=0), name='Load Shift', showlegend=True
                ))
                fig_hh.add_trace(go.Scatter(x=hours, y=ref_h, mode='lines', name='Reference Load',
                                             line=dict(color='#AB63FA', width=2, dash='dash')))
                fig_hh.add_trace(go.Scatter(x=hours, y=adj_h, mode='lines', name='Adjusted Load',
                                             line=dict(color='#636EFA', width=2.5)))
                fig_hh.update_layout(
                    title=f"Consumer {col_idx+1}",
                    xaxis_title="Hour", yaxis_title="Power (kW)",
                    height=300, showlegend=(col_idx == 0),
                    plot_bgcolor='#FAFAFA',
                    xaxis_showgrid=True, yaxis_showgrid=True,
                    xaxis_gridcolor='#E0E0E0', yaxis_gridcolor='#E0E0E0',
                    yaxis_rangemode='tozero',
                )
                with cols[col_idx]:
                    st.plotly_chart(style_fig(fig_hh), use_container_width=True)
        else:
            st.info("No consumers in the community.")

        # Prosumer profiles
        st.markdown("**Sample Prosumers (with PV)**")
        prosumer_indices = list(range(num_c, min(num_c + 3, N)))
        if prosumer_indices:
            cols = st.columns(len(prosumer_indices))
            for col_idx, hh_idx in enumerate(prosumer_indices):
                ref_h = hourly_avg(L_ref[hh_idx])
                adj_h = hourly_avg(L_adj[hh_idx])
                pv_h = hourly_avg(PV[hh_idx])

                fig_hh = go.Figure()
                # Fill between reference and adjusted
                fig_hh.add_trace(go.Scatter(
                    x=hours + hours[::-1],
                    y=list(ref_h) + list(adj_h)[::-1],
                    fill='toself', fillcolor='rgba(99, 110, 250, 0.12)',
                    line=dict(width=0), name='Load Shift', showlegend=True
                ))
                fig_hh.add_trace(go.Scatter(x=hours, y=ref_h, mode='lines', name='Reference Load',
                                             line=dict(color='#AB63FA', width=2, dash='dash')))
                fig_hh.add_trace(go.Scatter(x=hours, y=adj_h, mode='lines', name='Adjusted Load',
                                             line=dict(color='#636EFA', width=2.5)))
                fig_hh.add_trace(go.Scatter(x=hours, y=pv_h, mode='lines', name='PV Generation',
                                             line=dict(color='#00CC96', width=2.5)))
                fig_hh.update_layout(
                    title=f"Prosumer {col_idx+1}",
                    xaxis_title="Hour", yaxis_title="Power (kW)",
                    height=300, showlegend=(col_idx == 0),
                    plot_bgcolor='#FAFAFA',
                    xaxis_showgrid=True, yaxis_showgrid=True,
                    xaxis_gridcolor='#E0E0E0', yaxis_gridcolor='#E0E0E0',
                    yaxis_rangemode='tozero',
                )
                with cols[col_idx]:
                    st.plotly_chart(style_fig(fig_hh), use_container_width=True)

    with dsm_tab3:
        sdr_hourly = np.zeros(24)
        count = np.zeros(24)
        for d in range(days):
            for h in range(24):
                for slot in range(4):
                    idx = d * periods_per_day + h * 4 + slot
                    val = SDR_ts[idx]
                    if np.isfinite(val):
                        sdr_hourly[h] += val
                        count[h] += 1
        mask = count > 0
        sdr_hourly[mask] /= count[mask]

        fig_sdr = go.Figure()
        fig_sdr.add_trace(go.Scatter(x=hours, y=sdr_hourly, mode='lines+markers',
                                      name='SDR', line=dict(color='#1565C0', width=2)))
        fig_sdr.add_hline(y=1.0, line_dash="dash", line_color="red",
                          annotation_text="SDR=1 (balanced)")
        fig_sdr.update_layout(
            title="Hourly Supply-Demand Ratio (SDR-DSM)",
            xaxis_title="Hour of Day", yaxis_title="Supply-Demand Ratio",
            height=400
        )
        st.plotly_chart(style_fig(fig_sdr), use_container_width=True)

        peak_sdr = sdr_hourly[mask].max() if mask.any() else 0
        peak_hour = hours[np.argmax(sdr_hourly)] if mask.any() else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak SDR", f"{peak_sdr:.2f}")
        c2.metric("Peak Hour", f"{peak_hour}:00")
        c3.metric("DSM Iterations", str(dsm_data.get('iterations', 'N/A')))

    with dsm_tab4:
        buy_h = hourly_avg(Pr_buy)
        sell_h = hourly_avg(Pr_sell)
        lbuy = profiles_data['lambda_buy_ref']
        lsell = profiles_data['lambda_sell_ref']

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=hours, y=buy_h, mode='lines+markers',
                                        name='P2P Buy Price', line=dict(color='#1565C0', width=2)))
        fig_price.add_trace(go.Scatter(x=hours, y=sell_h, mode='lines+markers',
                                        name='P2P Sell Price', line=dict(color='#C62828', width=2)))
        fig_price.add_hline(y=lbuy, line_dash="dot", line_color="#1565C0",
                            annotation_text=f"Grid Buy (₹{lbuy:.2f})")
        fig_price.add_hline(y=lsell, line_dash="dot", line_color="#C62828",
                            annotation_text=f"Grid Sell (₹{lsell:.2f})")
        fig_price.update_layout(
            title="P2P Market Clearing Prices vs Grid Tariffs",
            xaxis_title="Hour of Day", yaxis_title="Price (₹/kWh)",
            height=400
        )
        st.plotly_chart(style_fig(fig_price), use_container_width=True)

        avg_spread = np.mean(buy_h - sell_h)
        st.metric("Average P2P Spread (₹/kWh)", f"{avg_spread:.4f}")

# Pricing Mechanism Comparison Section
st.header("Pricing Mechanism Comparison")

comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
    "Daily Savings (%)", "Consumer vs Prosumer", "Cumulative Costs", "Fairness & Stability"
])

# Pre-compute daily costs by mechanism and group
daily_costs_by_mech = {}
daily_consumer_costs = {}
daily_prosumer_costs = {}
num_c = profiles_data.get('num_consumers', 0)
for mech_name, settlement in settlements_data.items():
    daily_total = np.zeros(days)
    daily_cons = np.zeros(days)
    daily_pros = np.zeros(days)
    for d in range(days):
        start = d * periods_per_day
        end = (d + 1) * periods_per_day
        daily_total[d] = settlement.cost_ts[:, start:end].sum()
        if num_c > 0:
            daily_cons[d] = settlement.cost_ts[:num_c, start:end].sum()
        daily_pros[d] = settlement.cost_ts[num_c:, start:end].sum()
    daily_costs_by_mech[mech_name] = daily_total
    daily_consumer_costs[mech_name] = daily_cons
    daily_prosumer_costs[mech_name] = daily_pros

colors_map = {'Conventional': '#636363', 'MMR': '#1f77b4', 'Bill-Sharing': '#2ca02c', 'SDR': '#d62728'}

with comp_tab1:
    st.subheader("Daily Savings vs Conventional")
    st.markdown("Shows how much each P2P mechanism saves relative to conventional grid pricing each day.")

    conv_daily = daily_costs_by_mech['Conventional']
    fig_savings_pct = go.Figure()
    for mech_name, daily_costs in daily_costs_by_mech.items():
        if mech_name == 'Conventional':
            continue
        savings_pct = (conv_daily - daily_costs) / conv_daily * 100
        fig_savings_pct.add_trace(go.Scatter(
            x=list(range(1, days + 1)), y=savings_pct,
            mode='lines+markers', name=mech_name,
            line=dict(color=colors_map.get(mech_name, '#7f7f7f'), width=2.5),
            marker=dict(size=5)
        ))
    fig_savings_pct.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_savings_pct.update_layout(
        xaxis_title="Day", yaxis_title="Savings vs Conventional (%)",
        height=400, plot_bgcolor='white',
        xaxis_showgrid=True, yaxis_showgrid=True,
        xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(style_fig(fig_savings_pct), use_container_width=True)

    # Average savings metrics
    avg_cols = st.columns(3)
    for idx, mech_name in enumerate([n for n in daily_costs_by_mech if n != 'Conventional']):
        avg_sav = ((conv_daily - daily_costs_by_mech[mech_name]) / conv_daily * 100).mean()
        with avg_cols[idx]:
            st.metric(f"{mech_name} Avg Savings", f"{avg_sav:.1f}%")

with comp_tab2:
    st.subheader("Consumer vs Prosumer Daily Costs")
    st.markdown("Compare how each mechanism affects consumers (no PV) vs prosumers (with PV) separately.")

    col_cons, col_pros = st.columns(2)

    with col_cons:
        st.markdown("**Consumers**")
        if num_c > 0:
            fig_cons = go.Figure()
            for mech_name, daily_costs in daily_consumer_costs.items():
                fig_cons.add_trace(go.Scatter(
                    x=list(range(1, days + 1)), y=daily_costs / max(num_c, 1),
                    mode='lines', name=mech_name,
                    line=dict(color=colors_map.get(mech_name, '#7f7f7f'), width=2)
                ))
            fig_cons.update_layout(
                xaxis_title="Day", yaxis_title="Avg Cost per Consumer (₹)",
                height=350, plot_bgcolor='white',
                xaxis_showgrid=True, yaxis_showgrid=True,
                xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(style_fig(fig_cons), use_container_width=True)
        else:
            st.info("No consumers in the community.")

    with col_pros:
        st.markdown("**Prosumers**")
        num_p = total_households - num_c
        fig_pros = go.Figure()
        for mech_name, daily_costs in daily_prosumer_costs.items():
            fig_pros.add_trace(go.Scatter(
                x=list(range(1, days + 1)), y=daily_costs / max(num_p, 1),
                mode='lines', name=mech_name,
                line=dict(color=colors_map.get(mech_name, '#7f7f7f'), width=2)
            ))
        fig_pros.update_layout(
            xaxis_title="Day", yaxis_title="Avg Cost per Prosumer (₹)",
            height=350, plot_bgcolor='white',
            xaxis_showgrid=True, yaxis_showgrid=True,
            xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(style_fig(fig_pros), use_container_width=True)

    # Grouped bar: 30-day total per group per mechanism
    st.markdown("**30-Day Average Daily Cost per Household**")
    group_data = []
    for mech_name in daily_costs_by_mech:
        if num_c > 0:
            group_data.append({
                'Mechanism': mech_name, 'Group': 'Consumers',
                'Avg Daily Cost (₹)': daily_consumer_costs[mech_name].mean() / max(num_c, 1)
            })
        group_data.append({
            'Mechanism': mech_name, 'Group': 'Prosumers',
            'Avg Daily Cost (₹)': daily_prosumer_costs[mech_name].mean() / max(num_p, 1)
        })
    group_df = pd.DataFrame(group_data)
    fig_group = px.bar(group_df, x='Mechanism', y='Avg Daily Cost (₹)',
                       color='Group', barmode='group',
                       color_discrete_sequence=['#e74c3c', '#27ae60'])
    fig_group.update_layout(height=350, plot_bgcolor='white')
    st.plotly_chart(style_fig(fig_group), use_container_width=True)

with comp_tab3:
    st.subheader("Cumulative Cost Over Time")
    fig_cum = go.Figure()
    for mech_name, daily_costs in daily_costs_by_mech.items():
        cum_costs = np.cumsum(daily_costs)
        fig_cum.add_trace(go.Scatter(
            x=list(range(1, days + 1)), y=cum_costs,
            mode='lines', name=mech_name,
            line=dict(color=colors_map.get(mech_name, '#7f7f7f'), width=2.5)
        ))
    fig_cum.update_layout(
        xaxis_title="Day", yaxis_title="Cumulative Cost (₹)",
        height=400, plot_bgcolor='white',
        xaxis_showgrid=True, yaxis_showgrid=True,
        xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(style_fig(fig_cum), use_container_width=True)

    # Show final totals
    st.markdown("**30-Day Total Costs:**")
    total_cols = st.columns(len(daily_costs_by_mech))
    for idx, (mech_name, daily_costs) in enumerate(daily_costs_by_mech.items()):
        with total_cols[idx]:
            st.metric(mech_name, f"₹{daily_costs.sum():,.0f}")

with comp_tab4:
    st.subheader("Fairness & Stability Comparison")

    p2p_mechs = [n for n in result.metrics.mechanisms if n != 'Conventional']

    # Radar/spider chart for multi-dimensional comparison
    categories = ['Cost Savings (%)', 'Fairness Index', 'Consumer Savings (%)',
                  'Prosumer Savings (%)', 'Bill Stability']

    fig_radar = go.Figure()
    for mech_name in p2p_mechs:
        m = result.metrics.mechanisms[mech_name]
        stability_score = max(0, 1 - m.daily_cost_volatility / 10) * 100
        values = [
            m.cost_savings_pct,
            m.fairness_index * 100,
            m.consumer_savings_pct,
            m.prosumer_savings_pct,
            stability_score,
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=mech_name,
            line=dict(color=colors_map.get(mech_name, '#7f7f7f'), width=2),
            fill='toself', fillcolor=colors_map.get(mech_name, '#7f7f7f').replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in colors_map.get(mech_name, '') else None,
            opacity=0.8
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=450, showlegend=True
    )
    st.plotly_chart(style_fig(fig_radar), use_container_width=True)

    # Bar chart: fairness index comparison
    col_f, col_s = st.columns(2)
    with col_f:
        fairness_data = []
        for mech_name, m in result.metrics.mechanisms.items():
            fairness_data.append({'Mechanism': mech_name, 'Fairness Index': m.fairness_index})
        fairness_df = pd.DataFrame(fairness_data)
        fig_fair = px.bar(fairness_df, x='Mechanism', y='Fairness Index',
                          color='Mechanism', color_discrete_map=colors_map)
        fig_fair.update_layout(showlegend=False, height=300,
                               yaxis_range=[0, 1], title="Fairness Index (higher = fairer)")
        st.plotly_chart(style_fig(fig_fair), use_container_width=True)

    with col_s:
        vol_data = []
        for mech_name, m in result.metrics.mechanisms.items():
            vol_data.append({'Mechanism': mech_name, 'Daily Volatility (₹)': m.daily_cost_volatility})
        vol_df = pd.DataFrame(vol_data)
        fig_vol = px.bar(vol_df, x='Mechanism', y='Daily Volatility (₹)',
                         color='Mechanism', color_discrete_map=colors_map)
        fig_vol.update_layout(showlegend=False, height=300,
                              title="Daily Cost Volatility (lower = more stable)")
        st.plotly_chart(style_fig(fig_vol), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Mechanisms Explained:**
- **Conventional**: Traditional grid import/export at fixed tariffs
- **MMR (Mid-Market Rate)**: Dynamic P2P price based on supply-demand balance
- **Bill-Sharing**: Energy shared free within community; costs distributed ex-post
- **SDR (Supply-Demand Ratio)**: P2P prices based on real-time supply/demand balance, with demand-side management (controlled by α) allowing participants to shift loads for better prices
""")