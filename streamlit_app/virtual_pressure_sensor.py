# importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import joblib

# setting page configuration
st.set_page_config(
    page_title="OWC Virtual Pressure Sensor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# initializing session state variables
if 'show_success' not in st.session_state:
    st.session_state['show_success'] = False
if st.session_state['show_success']:
    st.toast("Simulation Run Finished", icon="✅")
    st.session_state['show_success'] = False
if 'running' not in st.session_state:
    st.session_state['running'] = False
if 'final_fig' not in st.session_state:
    st.session_state['final_fig'] = None
if 'last_wave' not in st.session_state:
    st.session_state['last_wave'] = 0.0
if 'last_actual' not in st.session_state:
    st.session_state['last_actual'] = 0.0
if 'last_pred' not in st.session_state:
    st.session_state['last_pred'] = 0.0
if 'last_error' not in st.session_state:
    st.session_state['last_error'] = 0.0

# loading model and data with caching
@st.cache_resource
def load_assets():
        # Ensure these paths match your local directory structure
        model = joblib.load('./model_assets/owc_virtual_sensor_model.pkl')
        scaler = joblib.load('./model_assets/scaler.pkl')
        return model, scaler

@st.cache_data
def load_data():
        # Ensure these paths match your local directory structure
        df_high_irr = pd.read_csv('./Processed_Data_Files/owc_high_irregular_test_data.csv')
        df_regular = pd.read_csv('./Processed_Data_Files/owc_normal_test_data.csv')
        return df_high_irr, df_regular

model, scaler = load_assets()
df_high_irr, df_regular = load_data()

# defining function to prepare features for prediction
def prepare_features_live(window_df, scaler):
    features = pd.DataFrame()
    lags = [0, 5, 10, 20, 30, 40, 50]

    for lag in lags:
        features[f'WG1_t-{lag}'] = window_df['WG1'].shift(lag)

    features_clean = features.dropna()

    if features_clean.empty:
        return None, None

    X_scaled = scaler.transform(features_clean)
    return X_scaled, features_clean.index

# defining function to start simulation
def start_simulation():
    if not st.session_state['running']:
        st.session_state['running'] = True
        # Reset any previous final figure when starting new run
        st.session_state['final_fig'] = None 

# setting up sidebar
st.sidebar.title("⚙️ System Configuration")
st.sidebar.markdown("Select the hydrodynamic environment to simulate below.")

scenario_type = st.sidebar.radio(
    "**Input Sea State:**",
    ("Regular Wave (Hs=0.025m)", "Highly Irregular Wave (Hs=0.158m)"),
    index=0,
    disabled=st.session_state['running']
)

# setting up main page
st.title("🌊 OWC Chamber Virtual Pressure Sensing System")
if not st.session_state['running'] and st.session_state['final_fig'] is None:
    st.info("Select parameters below and click **Run Virtual System** to begin real-time simulation of the OWC pressure response.", icon="ℹ️")

# defining variables for simulation based on sea state selection
current_df = None
wave_color = '#999999' 
sim_start_index = 500
regime_color = "gray"

# selecting system properties based on scenario type
if model is not None and df_high_irr is not None and df_regular is not None:
    if "Highly Irregular" in scenario_type:
        current_df = df_high_irr
        wave_color = '#1f77b4'
        sim_start_index = 2000
        regime_color = "red"
    else:
        current_df = df_regular
        wave_color = '#2ca02c'
        sim_start_index = 500
        regime_color = "green"

# setting the graph plot area
chart_spot = st.empty()

# showing the saved final graph when not running
if not st.session_state['running'] and st.session_state['final_fig'] is not None:
    chart_spot.plotly_chart(st.session_state['final_fig'], width='stretch')
else:
    if not st.session_state['running']:
        chart_spot.empty()

# setting the default parameters for simulation
speed = 0.25  # default refresh speed
window_len = 150  # default window length

# setting up simulation controls section
if not st.session_state['running']: 
    st.markdown("---")
    st.markdown("### 🎛️ Simulation Controls")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])

    with col_ctrl1:
        speed = st.slider(
            "Data Refresh Interval (s)",
            0.25,
            1.0,
            0.25,
            0.25,
            disabled=st.session_state['running']
        )

    with col_ctrl2:
        window_len = st.slider(
            "View Window per Refresh (Samples)",
            100,
            1500,
            100,
            disabled=st.session_state['running']
        )

    with col_ctrl3:
        st.write(" ")
        run_btn = st.button(
            "▶️ **Run Virtual System**",
            type="primary",
            disabled=st.session_state['running'] or model is None,
            on_click=start_simulation
        )


# set the sidebar for real-time simulation data
st.sidebar.markdown("---")
st.sidebar.subheader("📟 Real-time Simulation Data")

metric_container = st.sidebar.container()
with metric_container:
    m1 = st.sidebar.empty()
    m2 = st.sidebar.empty()
    m3 = st.sidebar.empty()
    m4 = st.sidebar.empty()

    # rendering the static side bar metrics
    m1.markdown(f"**🌊 Wave Height:** `{st.session_state['last_wave']:.3f} m`")
    m2.markdown(f"**💨 Actual Pressure:** `{st.session_state['last_actual']:.0f} Pa`")
    m3.markdown(f"**🤖 Pred. Pressure:** `{st.session_state['last_pred']:.0f} Pa`")

    err = st.session_state['last_error']
    err_color = "red" if abs(err) > 20 else "green"
    m4.markdown(
        f"**📉 Error:** <span style='color:{err_color}'>{err:+.1f} Pa</span>",
        unsafe_allow_html=True
    )

# setting up the sidebar for simulation progress
st.sidebar.subheader("⏳ Simulation Progress")

progress_bar = st.sidebar.progress(0)
run_status_text = st.sidebar.empty()

# setting the simulation loop
if st.session_state['running'] and model is not None and current_df is not None:
    st.toast("Simulation Run Started", icon="🏁")

    total_steps = 400
    step_size = 5
    sim_data = current_df.iloc[
        sim_start_index : sim_start_index + total_steps + window_len
    ].reset_index(drop=True)

    # initializing the figure outside loop for final save
    fig = None

    # running the simulation loop
    for i in range(0, total_steps, step_size):
        # updating progress bar and status text
        pct_complete = int((i / total_steps) * 100)
        run_status_text.markdown(f"**Running... {pct_complete}%**")
        progress_bar.progress(pct_complete)

        # preparing the window
        window_df = sim_data.iloc[i : i + window_len].copy()
        X_live, valid_indices = prepare_features_live(window_df, scaler)

        if X_live is not None:
            y_pred = model.predict(X_live)

            # extracting data for plotting
            t_valid = window_df.loc[valid_indices, 'Time']
            pressure_true = window_df.loc[valid_indices, 'P_Chamber']
            wave_height = window_df.loc[valid_indices, 'WG1']

            # rendering the plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t_valid,
                    y=wave_height,
                    name="Incident Wave WG1 (m)",
                    line=dict(color=wave_color, width=3),
                    yaxis='y1'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t_valid,
                    y=pressure_true,
                    name="Actual OWC Chamber Press. (Pa)",
                    line=dict(color='black', width=4),
                    opacity=0.2,
                    yaxis='y2'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t_valid,
                    y=y_pred,
                    name="Predicted OWC Chamber Press. (Pa)",
                    line=dict(color='#d62728', width=2, dash='dot'),
                    yaxis='y2'
                )
            )

            fig.update_layout(
                height=350,
                xaxis=dict(
                    title="Time (s)",
                    showgrid=False,
                    range=[t_valid.min(), t_valid.max()]
                ),
                yaxis=dict(
                    title="Wave Elevation (m)",
                    side="left",
                    showgrid=True,
                    title_font=dict(color=wave_color)
                ),
                yaxis2=dict(
                    title="Pneumatic Pressure (Pa)",
                    side="right",
                    overlaying="y",
                    showgrid=False,
                    title_font=dict(color='#d62728')
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.1, x=0),
                hovermode="x unified"
            )
            chart_spot.plotly_chart(fig, width='stretch')

            # updating the session state with current values for sidebar display
            curr_wave = wave_height.iloc[-1]
            curr_pres_act = pressure_true.iloc[-1]
            curr_pres_pred = y_pred[-1]
            error = curr_pres_pred - curr_pres_act

            st.session_state['last_wave'] = curr_wave
            st.session_state['last_actual'] = curr_pres_act
            st.session_state['last_pred'] = curr_pres_pred
            st.session_state['last_error'] = error

            # updating the sidebar metrics with current values
            m1.markdown(f"**🌊 Wave Height:** `{curr_wave:.3f} m`")
            m2.markdown(f"**💨 Actual Press:** `{curr_pres_act:.0f} Pa`")
            m3.markdown(f"**🤖 Pred. Press:** `{curr_pres_pred:.0f} Pa`")
            err_color = "red" if abs(error) > 20 else "green"
            m4.markdown(
                f"**📉 Error:** <span style='color:{err_color}'>{error:+.1f} Pa</span>",
                unsafe_allow_html=True
            )

        time.sleep(speed)

    # saving the final figure to session state for display after run
    st.session_state['final_fig'] = fig
    st.session_state['running'] = False

    # setting the success message state
    st.session_state['show_success'] = True

    # cleaning up the progress bar
    progress_bar.empty()
    run_status_text.empty()
    
    st.rerun()