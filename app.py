
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, time
import plotly.graph_objects as go


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RoadSense AI",
    page_icon="🚦",
    layout="wide"
)


# =====================================================
# 2. CUSTOM STYLING
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    accident_model = joblib.load(
        "models/roadsense_random_forest.pkl"
    )

    traffic_model = joblib.load(
        "models/roadsense_traffic_random_forest.pkl"
    )

    return accident_model, traffic_model


try:
    accident_model, traffic_model = load_models()

except Exception as e:
    st.error(
        "Could not load the trained models. "
        "Check that both .pkl files exist inside the models folder."
    )
    st.exception(e)
    st.stop()


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🚦 RoadSense AI")

st.sidebar.caption(
    "Intelligent Road Analytics Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Accident Severity Prediction",
        "Traffic Volume Prediction",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.success("ML models loaded successfully")


# =====================================================
# 5. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🚦 RoadSense AI")

    st.subheader(
        "Intelligent Road Analytics and Prediction"
    )

    st.write(
        "A Machine Learning application for "
        "accident severity classification and "
        "traffic volume prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🚨 Accident Classification")

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Classification Classes",
            "4"
        )

        st.write(
            "Predicts accident severity using road, "
            "weather, location and time features."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown("### 🚗 Traffic Regression")

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Test R² Score",
            "0.9284"
        )

        st.write(
            "Estimates hourly traffic volume using "
            "weather, holiday and time information."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Available Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Accident Severity Prediction**\n\n"
            "Enter accident conditions and estimate "
            "the severity category."
        )

    with col2:
        st.info(
            "**Traffic Volume Prediction**\n\n"
            "Enter weather and time information to "
            "estimate vehicles per hour."
        )


# =====================================================
# 6. ACCIDENT SEVERITY PREDICTION
# =====================================================

elif page == "Accident Severity Prediction":

    st.title("🚨 Accident Severity Prediction")

    st.write(
        "Enter road, weather, location and time "
        "information to predict accident severity."
    )

    st.warning(
        "This model estimates severity conditional on "
        "an accident having occurred. It does not predict "
        "whether an accident will happen."
    )

    with st.form("accident_form"):

        st.subheader("📍 Location Information")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Start Latitude",
                min_value=24.0,
                max_value=50.0,
                value=37.0,
                step=0.01
            )

        with col2:
            longitude = st.number_input(
                "Start Longitude",
                min_value=-125.0,
                max_value=-66.0,
                value=-95.0,
                step=0.01
            )

        state = st.text_input(
            "US State Code",
            value="CA",
            max_chars=2
        ).strip().upper()

        st.divider()

        st.subheader("🌦️ Weather Information")

        col1, col2 = st.columns(2)

        with col1:

            temperature = st.number_input(
                "Temperature (°F)",
                value=65.0,
                min_value=-90.0,
                max_value=160.0
            )

            humidity = st.number_input(
                "Humidity (%)",
                value=60.0,
                min_value=0.0,
                max_value=100.0
            )

            pressure = st.number_input(
                "Pressure (inches)",
                value=29.9,
                min_value=3.0,
                max_value=39.0
            )

        with col2:

            visibility = st.number_input(
                "Visibility (miles)",
                value=10.0,
                min_value=0.0,
                max_value=80.0
            )

            wind_speed = st.number_input(
                "Wind Speed (mph)",
                value=5.0,
                min_value=0.0,
                max_value=100.0
            )

            precipitation = st.number_input(
                "Precipitation (inches)",
                value=0.0,
                min_value=0.0,
                max_value=10.0
            )

        weather_condition = st.selectbox(
            "Weather Condition",
            [
                "Clear",
                "Mostly Cloudy",
                "Partly Cloudy",
                "Overcast",
                "Cloudy",
                "Fair",
                "Light Rain",
                "Rain",
                "Heavy Rain",
                "Thunderstorm",
                "Snow",
                "Light Snow",
                "Fog",
                "Mist",
                "Haze",
                "Drizzle",
                "Other"
            ]
        )

        wind_direction = st.selectbox(
            "Wind Direction",
            [
                "CALM", "N", "NNE", "NE", "ENE",
                "E", "ESE", "SE", "SSE", "S",
                "SSW", "SW", "WSW", "W",
                "WNW", "NW", "NNW", "VAR"
            ]
        )

        sunrise_sunset = st.selectbox(
            "Sunrise / Sunset",
            ["Day", "Night"]
        )

        st.divider()

        st.subheader("🕒 Accident Time")

        col1, col2 = st.columns(2)

        with col1:

            accident_date = st.date_input(
                "Accident Date",
                value=date(2020, 4, 17)
            )

        with col2:

            accident_time = st.time_input(
                "Accident Time",
                value=time(9, 30)
            )

        hour = accident_time.hour

        day_of_week = accident_date.weekday()

        month = accident_date.month

        is_weekend = int(day_of_week in [5, 6])

        st.divider()

        st.subheader("🛣️ Road Infrastructure")

        st.caption(
            "Select the road features present at the accident location."
        )

        road_columns = [
            "Amenity",
            "Bump",
            "Crossing",
            "Give_Way",
            "Junction",
            "No_Exit",
            "Railway",
            "Roundabout",
            "Station",
            "Stop",
            "Traffic_Calming",
            "Traffic_Signal",
            "Turning_Loop"
        ]

        road_labels = {
            "Amenity": "Amenity",
            "Bump": "Road Bump",
            "Crossing": "Pedestrian Crossing",
            "Give_Way": "Give Way Sign",
            "Junction": "Junction",
            "No_Exit": "No Exit",
            "Railway": "Railway Crossing",
            "Roundabout": "Roundabout",
            "Station": "Station",
            "Stop": "Stop Sign",
            "Traffic_Calming": "Traffic Calming",
            "Traffic_Signal": "Traffic Signal",
            "Turning_Loop": "Turning Loop"
        }

        road_values = {}

        road_col1, road_col2 = st.columns(2)

        for i, feature in enumerate(road_columns):

            target_col = road_col1 if i % 2 == 0 else road_col2

            with target_col:

                road_values[feature] = int(
                    st.checkbox(road_labels[feature])
                )

        submit_accident = st.form_submit_button(
            "Predict Accident Severity",
            use_container_width=True
        )

    if submit_accident:

        accident_input = pd.DataFrame([{

            "Start_Lat": latitude,
            "Start_Lng": longitude,

            "Temperature(F)": temperature,
            "Humidity(%)": humidity,
            "Pressure(in)": pressure,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,

            "start_hour": hour,
            "start_dayofweek": day_of_week,
            "start_month": month,
            "is_weekend": is_weekend,

            "State": state,
            "Weather_Condition": weather_condition,
            "Wind_Direction": wind_direction,
            "Sunrise_Sunset": sunrise_sunset

        } | road_values])

        try:

            prediction = accident_model.predict(
                accident_input
            )[0]

            severity_labels = {
                1: "Severity 1 - Lowest impact category",
                2: "Severity 2",
                3: "Severity 3",
                4: "Severity 4 - Highest impact category"
            }

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Accident Severity: "
                f"{int(prediction)}"
            )

            st.metric(
                "Severity Category",
                severity_labels.get(
                    int(prediction),
                    str(prediction)
                )
            )

            st.caption(
                "This is a model estimate, not a verified "
                "assessment of an accident."
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)


# =====================================================
# 7. TRAFFIC VOLUME PREDICTION
# =====================================================

elif page == "Traffic Volume Prediction":

    st.title("🚗 Traffic Volume Prediction")

    st.write(
        "Enter weather, holiday and time information "
        "to estimate the hourly number of vehicles."
    )

    st.info(
        "The traffic model was trained on observations "
        "from October 2012 through September 2018. "
        "Predictions for later years are extrapolations "
        "and have not been validated."
    )

    with st.form("traffic_form"):

        st.subheader("🌦️ Weather Information")

        col1, col2 = st.columns(2)

        with col1:

            temperature_c = st.number_input(
                "Temperature (°C)",
                value=8.0,
                min_value=-50.0,
                max_value=50.0
            )

            rainfall = st.number_input(
                "Rainfall in Previous Hour (mm)",
                value=0.0,
                min_value=0.0,
                max_value=100.0
            )

        with col2:

            snowfall = st.number_input(
                "Snowfall in Previous Hour (mm)",
                value=0.0,
                min_value=0.0,
                max_value=10.0
            )

            clouds = st.slider(
                "Cloud Coverage (%)",
                min_value=0,
                max_value=100,
                value=50
            )

        weather_main = st.selectbox(
            "Main Weather Condition",
            [
                "Clear",
                "Clouds",
                "Rain",
                "Drizzle",
                "Mist",
                "Snow",
                "Thunderstorm",
                "Fog",
                "Haze",
                "Smoke",
                "Squall"
            ]
        )

        weather_description = st.selectbox(
            "Weather Description",
            [
                "sky is clear",
                "Sky is Clear",
                "few clouds",
                "scattered clouds",
                "broken clouds",
                "overcast clouds",
                "mist",
                "light rain",
                "moderate rain",
                "heavy intensity rain",
                "light snow",
                "snow",
                "fog",
                "haze",
                "thunderstorm",
                "drizzle"
            ]
        )

        st.divider()

        st.subheader("📅 Date and Time")

        selected_date = st.date_input(
            "Select Date",
            value=date(2018, 9, 30)
        )

        selected_time = st.time_input(
            "Select Time",
            value=time(8, 0)
        )

        st.divider()

        st.subheader("🎉 Holiday Information")

        holiday_options = [
            "None",
            "New Years Day",
            "Washingtons Birthday",
            "Memorial Day",
            "Independence Day",
            "State Fair",
            "Labor Day",
            "Columbus Day",
            "Veterans Day",
            "Thanksgiving Day",
            "Christmas Day",
            "Martin Luther King Jr Day"
        ]

        holiday = st.selectbox(
            "Holiday",
            holiday_options
        )

        submit_traffic = st.form_submit_button(
            "Predict Traffic Volume",
            use_container_width=True
        )

    if submit_traffic:

        # Convert Celsius to Kelvin because
        # the training dataset uses Kelvin.
        temperature_kelvin = temperature_c + 273.15

        traffic_input = pd.DataFrame([{

            "temp": temperature_kelvin,
            "rain_1h": rainfall,
            "snow_1h": snowfall,
            "clouds_all": clouds,

            "hour": selected_time.hour,
            "day_of_week": selected_date.weekday(),
            "month": selected_date.month,
            "year": selected_date.year,

            "is_weekend": int(
                selected_date.weekday() in [5, 6]
            ),

            "holiday": holiday,
            "weather_main": weather_main,
            "weather_description": weather_description

        }])

        try:

            prediction = traffic_model.predict(
                traffic_input
            )[0]

            prediction = max(0, prediction)

            st.divider()

            st.subheader("Traffic Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Estimated Traffic Volume",
                    f"{prediction:,.0f} vehicles/hour"
                )

            with col2:

                st.metric(
                    "Estimated Vehicles",
                    f"{prediction:,.0f}"
                )

            st.success(
                "Traffic volume prediction completed!"
            )

            # Visualize prediction
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={
                        "text": "Estimated Vehicles per Hour"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 7280]
                        },
                        "bar": {
                            "color": "#2563eb"
                        },
                        "steps": [
                            {
                                "range": [0, 2000],
                                "color": "#dbeafe"
                            },
                            {
                                "range": [2000, 4500],
                                "color": "#bfdbfe"
                            },
                            {
                                "range": [4500, 7280],
                                "color": "#93c5fd"
                            }
                        ]
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if selected_date.year < 2012 or selected_date.year > 2018:
                st.warning(
                    "This date is outside the model's training "
                    "period. Treat this estimate as an unvalidated "
                    "extrapolation."
                )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)


# =====================================================
# 8. ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("About RoadSense AI")

    st.write("""
    RoadSense AI is a Machine Learning application
    developed to explore intelligent road analytics.

    It contains two independent ML modules.
    """)

    st.subheader("1. Accident Severity Classification")

    st.write("""
    Dataset: US Accidents

    Algorithm: Random Forest Classifier

    Output: Severity categories 1, 2, 3 and 4.
    """)

    st.subheader("2. Traffic Volume Regression")

    st.write("""
    Dataset: Metro Interstate Traffic Volume

    Algorithm: Random Forest Regressor

    Output: Estimated hourly vehicle count.
    """)

    st.divider()

    st.caption(
        "RoadSense AI | Machine Learning Project"
    )
