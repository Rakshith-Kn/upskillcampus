import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/cleaned_traffic_data.csv"

MODEL_FILE = "models/traffic_model.pkl"

FEATURE_FILE = "models/feature_names.pkl"

MODEL_RESULTS_FILE = "outputs/model_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    df["DateTime"] = pd.to_datetime(
        df["DateTime"]
    )

    return df


@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    features = joblib.load(
        FEATURE_FILE
    )

    return model, features


df = load_data()

model, features = load_model()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚦 Smart City Traffic Forecasting"
)

st.markdown(
    """
    ### Machine Learning Based Traffic Analysis & Forecasting

    This system analyzes historical traffic patterns across
    four city junctions and provides traffic forecasts using
    machine learning.
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏙️ Traffic Overview",
        "📊 Traffic Analysis",
        "🔮 Traffic Forecast"
    ]
)


# ============================================================
# PAGE 1 — TRAFFIC OVERVIEW
# ============================================================

if page == "🏙️ Traffic Overview":

    st.header(
        "🏙️ Traffic Overview"
    )

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    junction_avg = (
        df.groupby("Junction")["Vehicles"]
        .mean()
    )

    hourly_avg = (
        df.groupby("Hour")["Vehicles"]
        .mean()
    )

    day_type_avg = (
        df.groupby("DayType")["Vehicles"]
        .mean()
    )


    busiest_junction = (
        junction_avg.idxmax()
    )

    busiest_junction_value = (
        junction_avg.max()
    )

    peak_hour = (
        hourly_avg.idxmax()
    )

    peak_hour_value = (
        hourly_avg.max()
    )

    total_records = len(df)

    average_traffic = (
        df["Vehicles"].mean()
    )


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )


    with col2:

        st.metric(
            "Busiest Junction",
            f"Junction {busiest_junction}"
        )


    with col3:

        st.metric(
            "Peak Hour",
            f"{peak_hour}:00"
        )


    with col4:

        st.metric(
            "Average Traffic",
            f"{average_traffic:.2f}"
        )


    st.divider()


    # --------------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------------

    st.subheader(
        "🔎 Key Traffic Findings"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            f"""
            **Busiest Junction**

            Junction {busiest_junction}
            has the highest average traffic
            with approximately
            **{busiest_junction_value:.2f} vehicles**
            per observation.
            """
        )


    with col2:

        st.warning(
            f"""
            **Peak Traffic Period**

            The highest average traffic
            occurs around
            **{peak_hour}:00**
            with approximately
            **{peak_hour_value:.2f} vehicles**.
            """
        )


    # --------------------------------------------------------
    # DAY TYPE ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "📅 Traffic by Day Type"
    )


    day_order = [
        "Working Day",
        "Weekend",
        "Holiday"
    ]


    day_type_display = (
        day_type_avg
        .reindex(day_order)
        .dropna()
    )


    st.bar_chart(
        day_type_display
    )


    st.markdown(
        """
        **Interpretation:**  
        This comparison helps identify how traffic
        differs between normal working days,
        weekends, and holidays.
        """
    )


    # --------------------------------------------------------
    # JUNCTION TABLE
    # --------------------------------------------------------

    st.subheader(
        "🚦 Junction Traffic Summary"
    )


    junction_table = (
        junction_avg
        .reset_index()
    )


    junction_table.columns = [
        "Junction",
        "Average Vehicles"
    ]


    junction_table[
        "Average Vehicles"
    ] = junction_table[
        "Average Vehicles"
    ].round(2)


    st.dataframe(
        junction_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 2 — TRAFFIC ANALYSIS
# ============================================================

elif page == "📊 Traffic Analysis":

    st.header(
        "📊 Traffic Pattern Analysis"
    )


    st.markdown(
        """
        The following analysis is based on the historical
        traffic dataset used to train the forecasting system.
        """
    )


    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    selected_junction = st.selectbox(
        "Select Junction",
        sorted(
            df["Junction"].unique()
        )
    )


    junction_data = df[
        df["Junction"]
        == selected_junction
    ]


    # --------------------------------------------------------
    # HOURLY TRAFFIC
    # --------------------------------------------------------

    st.subheader(
        f"🕐 Hourly Traffic — Junction {selected_junction}"
    )


    hourly = (
        junction_data
        .groupby("Hour")["Vehicles"]
        .mean()
    )


    st.line_chart(
        hourly
    )


    # --------------------------------------------------------
    # DAY TYPE
    # --------------------------------------------------------

    st.subheader(
        "📅 Working Day vs Weekend vs Holiday"
    )


    day_type = (
        junction_data
        .groupby("DayType")["Vehicles"]
        .mean()
        .reindex(
            [
                "Working Day",
                "Weekend",
                "Holiday"
            ]
        )
        .dropna()
    )


    st.bar_chart(
        day_type
    )


    # --------------------------------------------------------
    # HOLIDAY ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "🎉 Holiday Traffic Pattern"
    )


    holiday_data = junction_data[
        junction_data["IsHoliday"] == 1
    ]


    if len(holiday_data) > 0:

        holiday_hourly = (
            holiday_data
            .groupby("Hour")["Vehicles"]
            .mean()
        )


        st.line_chart(
            holiday_hourly
        )


        holiday_peak = (
            holiday_hourly.idxmax()
        )


        st.success(
            f"""
            Holiday traffic at Junction
            {selected_junction} reaches its
            highest average level around
            **{holiday_peak}:00**.
            """
        )

    else:

        st.info(
            "No holiday records available for this junction."
        )


    # --------------------------------------------------------
    # PEAK HOUR ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "🔥 Peak-Hour Traffic"
    )


    peak_data = junction_data[
        junction_data["IsPeakHour"] == 1
    ]


    if len(peak_data) > 0:

        peak_average = (
            peak_data["Vehicles"].mean()
        )


        normal_data = junction_data[
            junction_data["IsPeakHour"] == 0
        ]


        normal_average = (
            normal_data["Vehicles"].mean()
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Peak-Hour Average",
                f"{peak_average:.2f}"
            )


        with col2:

            st.metric(
                "Non-Peak Average",
                f"{normal_average:.2f}"
            )


# ============================================================
# PAGE 3 — TRAFFIC FORECAST
# ============================================================

elif page == "🔮 Traffic Forecast":

    st.header(
        "🔮 Traffic Forecast"
    )


    st.markdown(
        """
        Enter the traffic conditions below.
        The trained Gradient Boosting model will
        estimate the expected number of vehicles.
        """
    )


    # --------------------------------------------------------
    # BASIC INPUTS
    # --------------------------------------------------------

    st.subheader(
        "📍 Location & Date"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        junction = st.selectbox(
            "Junction",
            sorted(
                df["Junction"].unique()
            )
        )


    with col2:

        year = st.number_input(
            "Year",
            min_value=2015,
            max_value=2035,
            value=2017
        )


    with col3:

        month = st.slider(
            "Month",
            1,
            12,
            6
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        day = st.slider(
            "Day",
            1,
            31,
            15
        )


    with col2:

        hour = st.slider(
            "Hour",
            0,
            23,
            19
        )


    with col3:

        day_of_week = st.slider(
            "Day of Week",
            0,
            6,
            0
        )


    st.caption(
        "Day of Week: 0 = Monday, 6 = Sunday"
    )


    # --------------------------------------------------------
    # DAY TYPE
    # --------------------------------------------------------

    st.subheader(
        "📅 Day Type"
    )


    day_type = st.selectbox(
        "Select Day Type",
        [
            "Working Day",
            "Weekend",
            "Holiday"
        ]
    )


    is_holiday = int(
        day_type == "Holiday"
    )


    is_weekend = int(
        day_type == "Weekend"
    )


    is_working_day = int(
        day_type == "Working Day"
    )


    # --------------------------------------------------------
    # PEAK HOUR
    # --------------------------------------------------------

    peak_hours = [
        7, 8, 9,
        17, 18, 19
    ]


    is_peak_hour = int(
        hour in peak_hours
    )


    # --------------------------------------------------------
    # RECENT TRAFFIC
    # --------------------------------------------------------

    st.subheader(
        "📈 Recent Traffic Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        lag_1 = st.number_input(
            "Previous Traffic",
            min_value=0.0,
            value=20.0
        )


    with col2:

        lag_2 = st.number_input(
            "Traffic 2 Periods Ago",
            min_value=0.0,
            value=20.0
        )


    with col3:

        lag_3 = st.number_input(
            "Traffic 3 Periods Ago",
            min_value=0.0,
            value=20.0
        )


    # Rolling features

    rolling_mean = (
        lag_1 +
        lag_2 +
        lag_3
    ) / 3


    rolling_std = pd.Series(
        [
            lag_1,
            lag_2,
            lag_3
        ]
    ).std()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.divider()


    if st.button(
        "🚦 Predict Traffic",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{

            "Junction": junction,

            "Year": year,

            "Month": month,

            "Day": day,

            "Hour": hour,

            "DayOfWeek": day_of_week,

            "IsWeekend": is_weekend,

            "IsPeakHour": is_peak_hour,

            "IsHoliday": is_holiday,

            "IsWorkingDay": is_working_day,

            "Lag_1": lag_1,

            "Lag_2": lag_2,

            "Lag_3": lag_3,

            "Rolling_Mean_3": rolling_mean,

            "Rolling_Std_3": rolling_std

        }])


        prediction = model.predict(
            input_data[features]
        )[0]


        prediction = max(
            0,
            prediction
        )


        # ----------------------------------------------------
        # TRAFFIC LEVEL
        # ----------------------------------------------------

        if prediction < 15:

            traffic_level = "LOW"

        elif prediction < 30:

            traffic_level = "MODERATE"

        elif prediction < 50:

            traffic_level = "HIGH"

        else:

            traffic_level = "VERY HIGH"


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success(
            "Traffic prediction generated successfully!"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Predicted Vehicles",
                f"{prediction:.0f}"
            )


        with col2:

            st.metric(
                "Traffic Level",
                traffic_level
            )


        # ----------------------------------------------------
        # WARNINGS
        # ----------------------------------------------------

        if is_peak_hour:

            st.warning(
                "⚠️ The selected time falls within "
                "a defined peak traffic period."
            )


        if is_holiday:

            st.info(
                "🎉 Holiday traffic conditions are "
                "being considered by the model."
            )


        if is_working_day:

            st.info(
                "💼 Working-day traffic conditions "
                "are being considered."
            )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "💡 Traffic Management Insight"
        )


        if traffic_level == "VERY HIGH":

            st.error(
                """
                High traffic demand is predicted.
                Traffic authorities may consider
                increased monitoring, traffic
                signal optimization, and congestion
                management measures.
                """
            )

        elif traffic_level == "HIGH":

            st.warning(
                """
                Significant traffic is predicted.
                Additional traffic monitoring may
                be useful during this period.
                """
            )

        elif traffic_level == "MODERATE":

            st.info(
                """
                Moderate traffic is predicted.
                Normal traffic management measures
                should be sufficient.
                """
            )

        else:

            st.success(
                """
                Low traffic is predicted.
                No significant congestion is expected
                under the selected conditions.
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart City Traffic Forecasting | "
    "Machine Learning Internship Project"
)