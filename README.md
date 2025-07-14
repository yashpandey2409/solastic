Solastics: Solar Power Maintenance Alert System
Solastics is an interactive Streamlit dashboard designed to monitor, analyze, and predict the performance of solar power plants. It provides real-time insights, detects potential faults, and offers power output predictions using a machine learning model, helping in proactive maintenance and optimal energy generation.

✨ Features
🏠 Real-Time Dashboard: Get an overview of current power output, efficiency, peak power, and system uptime. Includes real-time plots of power output, temperature, and solar irradiance.

📊 Data Analysis: Explore historical data with summary statistics, correlation analysis, power output distribution, and hourly patterns. Filter data by date range for detailed insights.

⚠️ Fault Detection: Identify potential issues and anomalies in power output using statistical methods. Displays recent fault occurrences and system health indicators.

🔮 Power Prediction: Utilizes a trained Machine Learning model (RandomForestRegressor) to predict future power output based on environmental conditions. Provides model performance metrics (R², RMSE) and feature importance.

📈 Performance Metrics: Detailed analysis of key performance indicators (KPIs) like total energy production, average efficiency, capacity factor, and system availability over various time periods. Includes performance trends and industry benchmarks.

🧠 How it Works (Model & Analytics)
This application integrates several analytical components to provide a comprehensive monitoring solution:

Data Generation: Realistic solar power plant data (irradiance, temperature, wind speed, power output, efficiency, and simulated faults) is generated hourly for the past 30 days.

Performance Metrics: Key performance indicators (KPIs) are calculated dynamically based on the latest data, providing immediate insights into the plant's operational status.

Anomaly Detection: A statistical method is employed to identify unusual deviations in power output, flagging potential issues that require attention.

Power Prediction (Machine Learning Model):

A RandomForestRegressor model is trained on historical data, using features like irradiance, temperature, wind speed, hour of day, day of year, and month.

The model's performance (R² score, RMSE) and feature importance are displayed.

Users can input hypothetical environmental conditions via sliders to get real-time power output predictions from the trained model.

Historical actual vs. predicted power output is visualized to demonstrate model accuracy.

🛠️ Technologies Used
Python 

Streamlit: For building the interactive web dashboard.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations and data generation.

Plotly Express & Plotly Graph Objects: For interactive and compelling data visualizations.

Scikit-learn: For the Machine Learning model (RandomForestRegressor) used in power prediction.

🚀 Setup and Installation
To run Solastics locally, follow these steps:

Clone the repository:
git clone https://github.com/yashpandey2409/solastics.git
cd solastics

Create a virtual environment (recommended):
python -m venv venv

Activate the virtual environment:

On Windows:
.\venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run app.py
The application will open in your default web browser.

💡 Usage
Navigate through the different sections using the sidebar on the left:

Dashboard: For a quick overview.
Data Analysis: To delve into historical data trends.
Fault Detection: To check for anomalies and system health.
Power Prediction: To interact with the ML model and get predictions.
Performance Metrics: For detailed KPI analysis.
Interact with sliders, date pickers, and buttons to filter data and generate predictions.

📊 Data Generation
The application uses a generate_sample_data function to create synthetic, yet realistic, solar power plant data. This allows the dashboard to be fully functional out-of-the-box without requiring external data sources.

🚀 Future Enhancements
Integration with real-time sensor data feeds.
More advanced machine learning models for fault detection (e.g., unsupervised learning).
Incorporation of weather API data for more accurate predictions.
Alert notifications


