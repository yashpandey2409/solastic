import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Solastic: Solar Power Maintance Alert System",
    page_icon="logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        color: #2c3e50;
    }
    .metric-card h3 {
        color: #FF6B35;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #2c3e50;
        margin-bottom: 0;
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        color: #d32f2f;
    }
    .alert-card h4 {
        color: #d32f2f;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .alert-card p {
        color: #5d4037;
        margin-bottom: 0;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .success-card h4 {
        color: #2e7d32;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .success-card p {
        color: #4a5d23;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(days=30):
    """Generate realistic solar power plant data"""
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='H')
    
    data = []
    for timestamp in timestamps:
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Simulate solar irradiance (peak at noon, zero at night)
        if 6 <= hour <= 18:
            base_irradiance = 800 * np.sin(np.pi * (hour - 6) / 12)
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * day_of_year / 365)
            irradiance = base_irradiance * seasonal_factor * (0.8 + 0.4 * np.random.random())
        else:
            irradiance = 0
        
        # Weather conditions
        weather_factor = np.random.choice([0.9, 0.7, 0.5], p=[0.7, 0.2, 0.1])  # sunny, cloudy, rainy
        irradiance *= weather_factor
        
        # Temperature (affects efficiency)
        base_temp = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365) + 5 * np.sin(np.pi * hour / 12)
        temperature = base_temp + np.random.normal(0, 3)
        
        # Wind speed
        wind_speed = np.random.exponential(3) + 1
        
        # Panel efficiency (decreases with temperature)
        efficiency = 0.20 - 0.004 * max(0, temperature - 25)
        
        # Power output (simplified)
        panel_area = 1000  # m²
        power_output = irradiance * panel_area * efficiency / 1000  # kW
        
        # Add some random faults
        fault_probability = 0.02
        if np.random.random() < fault_probability:
            power_output *= 0.5  # 50% reduction due to fault
            fault_detected = True
        else:
            fault_detected = False
        
        # Add noise
        power_output += np.random.normal(0, power_output * 0.05)
        power_output = max(0, power_output)
        
        data.append({
            'timestamp': timestamp,
            'irradiance': irradiance,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'power_output': power_output,
            'efficiency': efficiency * 100,
            'fault_detected': fault_detected
        })
    
    return pd.DataFrame(data)

def calculate_performance_metrics(df):
    """Calculate key performance metrics"""
    latest_data = df.tail(24)  # Last 24 hours
    
    metrics = {
        'total_energy_today': latest_data['power_output'].sum(),
        'avg_efficiency': latest_data['efficiency'].mean(),
        'peak_power': latest_data['power_output'].max(),
        'capacity_factor': (latest_data['power_output'].mean() / latest_data['power_output'].max()) * 100 if latest_data['power_output'].max() > 0 else 0,
        'fault_count': latest_data['fault_detected'].sum(),
        'uptime_percentage': (1 - latest_data['fault_detected'].mean()) * 100
    }
    
    return metrics

def detect_anomalies(df):
    """Simple anomaly detection based on statistical methods"""
    recent_data = df.tail(168)  # Last week
    
    # Calculate rolling statistics
    rolling_mean = recent_data['power_output'].rolling(window=24).mean()
    rolling_std = recent_data['power_output'].rolling(window=24).std()
    
    # Define anomaly threshold (2 standard deviations)
    threshold = 2
    anomalies = []
    
    for i in range(len(recent_data)):
        if i >= 24:  # Need enough data for rolling stats
            value = recent_data.iloc[i]['power_output']
            mean_val = rolling_mean.iloc[i]
            std_val = rolling_std.iloc[i]
            
            if abs(value - mean_val) > threshold * std_val:
                anomalies.append({
                    'timestamp': recent_data.iloc[i]['timestamp'],
                    'power_output': value,
                    'expected': mean_val,
                    'deviation': abs(value - mean_val) / std_val if std_val > 0 else 0
                })
    
    return anomalies

def train_power_prediction_model(df):
    """Train a simple ML model for power prediction"""
    # Prepare features
    df_model = df.copy()
    df_model['hour'] = df_model['timestamp'].dt.hour
    df_model['day_of_year'] = df_model['timestamp'].dt.dayofyear
    df_model['month'] = df_model['timestamp'].dt.month
    
    # Features for prediction
    features = ['irradiance', 'temperature', 'wind_speed', 'hour', 'day_of_year', 'month']
    X = df_model[features].fillna(0)
    y = df_model['power_output']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, features

def main():
    # Display logo and header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("logo.jpg", width=200)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">Solatic: Solar Power Maintenance Alert System</h1>', unsafe_allow_html=True)
    
    # Load or generate data
    if 'data' not in st.session_state:
        with st.spinner('Loading solar power data...'):
            st.session_state.data = generate_sample_data(30)
    
    df = st.session_state.data
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("logo.jpg", width=150)
        st.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📊 Data Analysis", "⚠️ Fault Detection", "🔮 Power Prediction", "📈 Performance Metrics"]
    )
    
    if page == "🏠 Dashboard":
        dashboard_page(df)
    elif page == "📊 Data Analysis":
        data_analysis_page(df)
    elif page == "⚠️ Fault Detection":
        fault_detection_page(df)
    elif page == "🔮 Power Prediction":
        power_prediction_page(df)
    elif page == "📈 Performance Metrics":
        performance_metrics_page(df)

def dashboard_page(df):
    """Main dashboard with key metrics and charts"""
    st.header("Real-Time Dashboard")
    
    # Calculate metrics
    metrics = calculate_performance_metrics(df)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Energy Today (kWh)",
            value=f"{metrics['total_energy_today']:.1f}",
            delta=f"{metrics['total_energy_today'] - df.tail(48).head(24)['power_output'].sum():.1f}"
        )
    
    with col2:
        st.metric(
            label="Average Efficiency (%)",
            value=f"{metrics['avg_efficiency']:.1f}%",
            delta=f"{metrics['avg_efficiency'] - df.tail(48).head(24)['efficiency'].mean():.1f}%"
        )
    
    with col3:
        st.metric(
            label="Peak Power (kW)",
            value=f"{metrics['peak_power']:.1f}",
            delta=f"{metrics['peak_power'] - df.tail(48).head(24)['power_output'].max():.1f}"
        )
    
    with col4:
        st.metric(
            label="System Uptime",
            value=f"{metrics['uptime_percentage']:.1f}%",
            delta=f"{metrics['uptime_percentage'] - (1 - df.tail(48).head(24)['fault_detected'].mean()) * 100:.1f}%"
        )
    
    # Real-time power output chart
    st.subheader("Real-Time Power Output")
    recent_data = df.tail(48)  # Last 48 hours
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data['power_output'],
        mode='lines',
        name='Power Output (kW)',
        line=dict(color='#FF6B35', width=2)
    ))
    
    fig.update_layout(
        title="Power Output - Last 48 Hours",
        xaxis_title="Time",
        yaxis_title="Power Output (kW)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Environmental conditions
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp = px.line(recent_data, x='timestamp', y='temperature', 
                          title='Temperature Trend', color_discrete_sequence=['#FF6B35'])
        fig_temp.update_layout(yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        fig_irr = px.line(recent_data, x='timestamp', y='irradiance', 
                         title='Solar Irradiance', color_discrete_sequence=['#FFA500'])
        fig_irr.update_layout(yaxis_title="Irradiance (W/m²)")
        st.plotly_chart(fig_irr, use_container_width=True)
    
    # Alert system
    st.subheader("System Alerts")
    
    if metrics['fault_count'] > 0:
        st.markdown(f"""
        <div class="alert-card">
            <h4>⚠️ Active Faults Detected</h4>
            <p>Number of faults in last 24 hours: <strong>{int(metrics['fault_count'])}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-card">
            <h4>✅ All Systems Operational</h4>
            <p>No faults detected in the last 24 hours</p>
        </div>
        """, unsafe_allow_html=True)

def data_analysis_page(df):
    """Detailed data analysis page"""
    st.header("Data Analysis & Insights")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['timestamp'].min().date())
    with col2:
        end_date = st.date_input("End Date", df['timestamp'].max().date())
    
    # Filter data
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df.loc[mask]
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = filtered_df[['power_output', 'efficiency', 'temperature', 'irradiance']].describe()
    st.dataframe(summary_stats)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr_data = filtered_df[['power_output', 'irradiance', 'temperature', 'wind_speed', 'efficiency']].corr()
    fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto", title="Variable Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Power output distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(filtered_df, x='power_output', nbins=50, 
                               title='Power Output Distribution')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(filtered_df, y='power_output', title='Power Output Box Plot')
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Hourly patterns
    st.subheader("Daily and Hourly Patterns")
    
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    hourly_avg = filtered_df.groupby('hour')['power_output'].mean().reset_index()
    
    fig_hourly = px.line(hourly_avg, x='hour', y='power_output', 
                        title='Average Power Output by Hour')
    fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Power Output (kW)")
    st.plotly_chart(fig_hourly, use_container_width=True)

def fault_detection_page(df):
    """Fault detection and system health monitoring"""
    st.header("Fault Detection & System Health")
    
    # Detect anomalies
    anomalies = detect_anomalies(df)
    
    # Fault summary
    recent_faults = df.tail(168)['fault_detected'].sum()  # Last week
    total_faults = df['fault_detected'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Faults This Week", int(recent_faults))
    with col2:
        st.metric("Total Faults", int(total_faults))
    with col3:
        st.metric("Anomalies Detected", len(anomalies))
    
    # Fault timeline
    st.subheader("Fault Timeline")
    fault_data = df[df['fault_detected'] == True].tail(50)
    
    if not fault_data.empty:
        fig_faults = px.scatter(fault_data, x='timestamp', y='power_output',
                               title='Recent Fault Occurrences',
                               color_discrete_sequence=['red'])
        fig_faults.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_faults, use_container_width=True)
    else:
        st.info("No recent faults detected!")
    
    # Anomaly detection results
    st.subheader("Statistical Anomaly Detection")
    
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        st.dataframe(anomaly_df)
        
        # Plot anomalies
        recent_data = df.tail(168)
        fig_anomaly = go.Figure()
        
        # Normal data
        fig_anomaly.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['power_output'],
            mode='lines',
            name='Normal Operation',
            line=dict(color='blue')
        ))
        
        # Anomalies
        fig_anomaly.add_trace(go.Scatter(
            x=anomaly_df['timestamp'],
            y=anomaly_df['power_output'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        ))
        
        fig_anomaly.update_layout(title="Anomaly Detection Results")
        st.plotly_chart(fig_anomaly, use_container_width=True)
    else:
        st.success("No anomalies detected in recent data!")
    
    # System health indicators
    st.subheader("System Health Indicators")
    
    health_metrics = {
        'Power Output Stability': 85 + np.random.randint(-10, 10),
        'Temperature Control': 92 + np.random.randint(-5, 5),
        'Inverter Performance': 88 + np.random.randint(-8, 8),
        'Connection Quality': 95 + np.random.randint(-3, 3)
    }
    
    for metric, value in health_metrics.items():
        color = 'green' if value >= 90 else 'orange' if value >= 70 else 'red'
        st.progress(value/100)
        st.write(f"{metric}: {value}%")

def power_prediction_page(df):
    """Power prediction using machine learning"""
    st.header("Power Output Prediction")
    
    # Train model
    with st.spinner('Training prediction model...'):
        model, mse, r2, features = train_power_prediction_model(df)
    
    # Model performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model R² Score", f"{r2:.3f}")
    with col2:
        st.metric("Root Mean Square Error", f"{np.sqrt(mse):.2f} kW")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title='Feature Importance in Power Prediction')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction interface
    st.subheader("Make Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_irradiance = st.slider("Solar Irradiance (W/m²)", 0, 1000, 500)
        pred_temperature = st.slider("Temperature (°C)", -10, 50, 25)
    
    with col2:
        pred_wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5)
        pred_hour = st.slider("Hour of Day", 0, 23, 12)
    
    with col3:
        pred_day_of_year = st.slider("Day of Year", 1, 365, 180)
        pred_month = st.slider("Month", 1, 12, 6)
    
    # Make prediction
    if st.button("Predict Power Output"):
        prediction_input = [[pred_irradiance, pred_temperature, pred_wind_speed, 
                           pred_hour, pred_day_of_year, pred_month]]
        predicted_power = model.predict(prediction_input)[0]
        
        st.success(f"Predicted Power Output: {predicted_power:.2f} kW")
    
    # Historical predictions vs actual
    st.subheader("Model Validation")
    
    recent_data = df.tail(168).copy()
    recent_data['hour'] = recent_data['timestamp'].dt.hour
    recent_data['day_of_year'] = recent_data['timestamp'].dt.dayofyear
    recent_data['month'] = recent_data['timestamp'].dt.month
    
    X_recent = recent_data[features].fillna(0)
    predicted_recent = model.predict(X_recent)
    
    fig_validation = go.Figure()
    fig_validation.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data['power_output'],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    fig_validation.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=predicted_recent,
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    fig_validation.update_layout(title="Actual vs Predicted Power Output")
    st.plotly_chart(fig_validation, use_container_width=True)

def performance_metrics_page(df):
    """Detailed performance metrics and KPIs"""
    st.header("Performance Metrics & KPIs")
    
    # Time period selector
    period = st.selectbox("Select Time Period", ["Last 24 Hours", "Last Week", "Last Month", "All Time"])
    
    if period == "Last 24 Hours":
        data_subset = df.tail(24)
    elif period == "Last Week":
        data_subset = df.tail(168)
    elif period == "Last Month":
        data_subset = df.tail(720)
    else:
        data_subset = df
    
    # Key Performance Indicators
    total_energy = data_subset['power_output'].sum()
    avg_efficiency = data_subset['efficiency'].mean()
    capacity_factor = (data_subset['power_output'].mean() / data_subset['power_output'].max()) * 100 if data_subset['power_output'].max() > 0 else 0
    availability = (1 - data_subset['fault_detected'].mean()) * 100
    
    # Display metrics in cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Energy Production</h3>
            <h2>{total_energy:.1f} kWh</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Capacity Factor</h3>
            <h2>{capacity_factor:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Efficiency</h3>
            <h2>{avg_efficiency:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>System Availability</h3>
            <h2>{availability:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.subheader("Performance Trends")
    
    # Daily energy production
    data_subset['date'] = data_subset['timestamp'].dt.date
    daily_energy = data_subset.groupby('date')['power_output'].sum().reset_index()
    
    fig_daily = px.bar(daily_energy, x='date', y='power_output',
                      title='Daily Energy Production')
    fig_daily.update_layout(yaxis_title="Energy Production (kWh)")
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Efficiency vs Temperature
    fig_eff_temp = px.scatter(data_subset, x='temperature', y='efficiency',
                             title='Efficiency vs Temperature')
    st.plotly_chart(fig_eff_temp, use_container_width=True)
    
    # Performance comparison table
    st.subheader("Performance Benchmarks")
    
    benchmarks = pd.DataFrame({
        'Metric': ['Energy Production', 'Efficiency', 'Capacity Factor', 'Availability'],
        'Current Value': [f"{total_energy:.1f} kWh", f"{avg_efficiency:.1f}%", 
                         f"{capacity_factor:.1f}%", f"{availability:.1f}%"],
        'Industry Standard': ["Variable", "18-22%", "15-25%", ">95%"],
        'Status': ["✅ Good", "✅ Good" if avg_efficiency >= 18 else "⚠️ Below Average",
                  "✅ Good" if capacity_factor >= 15 else "⚠️ Below Average",
                  "✅ Good" if availability >= 95 else "⚠️ Below Standard"]
    })
    
    st.dataframe(benchmarks, use_container_width=True)

if __name__ == "__main__":
    main()
