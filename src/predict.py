import joblib
import pandas as pd


MODEL_FILE = "models/traffic_model.pkl"

model = joblib.load(
    MODEL_FILE
)


features = joblib.load(
    "models/feature_names.pkl"
)


print("\nSMART CITY TRAFFIC FORECASTING")

print("--------------------------------")


junction = int(
    input("Enter Junction Number: ")
)

year = int(
    input("Enter Year: ")
)

month = int(
    input("Enter Month (1-12): ")
)

day = int(
    input("Enter Day: ")
)

hour = int(
    input("Enter Hour (0-23): ")
)

day_of_week = int(
    input(
        "Enter Day of Week (0=Monday, 6=Sunday): "
    )
)


is_weekend = int(
    day_of_week >= 5
)

is_peak = int(
    hour in [7, 8, 9, 17, 18, 19]
)


# For a simple manual prediction,
# lag values are entered by user.

lag_1 = float(
    input(
        "Previous traffic count (Lag 1): "
    )
)

lag_2 = float(
    input(
        "Traffic count two periods ago (Lag 2): "
    )
)

lag_3 = float(
    input(
        "Traffic count three periods ago (Lag 3): "
    )
)

rolling_mean = (
    lag_1 +
    lag_2 +
    lag_3
) / 3


rolling_std = pd.Series(
    [lag_1, lag_2, lag_3]
).std()


input_data = pd.DataFrame([{

    "Junction": junction,

    "Year": year,

    "Month": month,

    "Day": day,

    "Hour": hour,

    "DayOfWeek": day_of_week,

    "IsWeekend": is_weekend,

    "IsPeakHour": is_peak,

    "Lag_1": lag_1,

    "Lag_2": lag_2,

    "Lag_3": lag_3,

    "Rolling_Mean_3": rolling_mean,

    "Rolling_Std_3": rolling_std

}])


prediction = model.predict(
    input_data[features]
)[0]


print("\n--------------------------------")

print(
    f"Predicted Vehicles: {prediction:.0f}"
)

print("--------------------------------")