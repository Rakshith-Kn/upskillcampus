import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SMART CITY TRAFFIC FORECASTING
# MODEL EVALUATION
# ============================================================

DATA_FILE = "data/cleaned_traffic_data.csv"

MODEL_FILE = "models/traffic_model.pkl"

FEATURE_FILE = "models/feature_names.pkl"

OUTPUT_DIR = "outputs/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n==============================================")
print(" SMART CITY TRAFFIC FORECASTING")
print(" MODEL EVALUATION")
print("==============================================")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

df["DateTime"] = pd.to_datetime(df["DateTime"])


# Keep same ordering used during training
df = df.sort_values(
    "DateTime"
).reset_index(drop=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = joblib.load(
    MODEL_FILE
)

features = joblib.load(
    FEATURE_FILE
)


print("Model loaded successfully.")


# ============================================================
# PREPARE DATA
# ============================================================

df = df.dropna(
    subset=features + ["Vehicles"]
).reset_index(drop=True)


X = df[features]

y = df["Vehicles"]


# ============================================================
# SAME TIME-BASED SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)


X_test = X.iloc[split_index:]

y_test = y.iloc[split_index:]

test_dates = df["DateTime"].iloc[split_index:]

test_data = df.iloc[split_index:].copy()


print("\n==============================================")
print(" TEST DATA")
print("==============================================")

print(
    "Testing records:",
    len(X_test)
)

print(
    "Testing period:",
    test_dates.iloc[0],
    "to",
    test_dates.iloc[-1]
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)


# Prevent negative vehicle predictions
predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================================")
print(" MODEL PERFORMANCE")
print("==============================================")


print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results = test_data[
    [
        "DateTime",
        "Junction",
        "Vehicles",
        "IsHoliday",
        "IsWorkingDay",
        "IsPeakHour"
    ]
].copy()


results["Predicted_Vehicles"] = predictions


results["Absolute_Error"] = (
    abs(
        results["Vehicles"]
        -
        results["Predicted_Vehicles"]
    )
)


results.to_csv(
    "outputs/prediction_results.csv",
    index=False
)


print(
    "\nPrediction results saved to:"
)

print(
    "outputs/prediction_results.csv"
)


# ============================================================
# 1. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    results["DateTime"].iloc[:500],
    results["Vehicles"].iloc[:500],
    label="Actual"
)

plt.plot(
    results["DateTime"].iloc[:500],
    results["Predicted_Vehicles"].iloc[:500],
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Traffic"
)

plt.xlabel(
    "Date and Time"
)

plt.ylabel(
    "Number of Vehicles"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/actual_vs_predicted.png"
)

plt.show()


# ============================================================
# 2. PREDICTION ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

sns.histplot(
    results["Absolute_Error"],
    bins=40,
    kde=True
)

plt.title(
    "Prediction Error Distribution"
)

plt.xlabel(
    "Absolute Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/prediction_error_distribution.png"
)

plt.show()


# ============================================================
# 3. ACTUAL VS PREDICTED SCATTER
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.scatterplot(
    x=results["Vehicles"],
    y=results["Predicted_Vehicles"]
)

# Perfect prediction reference line

minimum = min(
    results["Vehicles"].min(),
    results["Predicted_Vehicles"].min()
)

maximum = max(
    results["Vehicles"].max(),
    results["Predicted_Vehicles"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Vehicle Counts"
)

plt.xlabel(
    "Actual Vehicles"
)

plt.ylabel(
    "Predicted Vehicles"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/actual_vs_predicted_scatter.png"
)

plt.show()


# ============================================================
# 4. PEAK-HOUR PERFORMANCE
# ============================================================

peak_data = results[
    results["IsPeakHour"] == 1
]


if len(peak_data) > 0:

    peak_mae = mean_absolute_error(
        peak_data["Vehicles"],
        peak_data["Predicted_Vehicles"]
    )

    peak_rmse = np.sqrt(
        mean_squared_error(
            peak_data["Vehicles"],
            peak_data["Predicted_Vehicles"]
        )
    )

    peak_r2 = r2_score(
        peak_data["Vehicles"],
        peak_data["Predicted_Vehicles"]
    )


    print("\n==============================================")
    print(" PEAK-HOUR PERFORMANCE")
    print("==============================================")

    print(
        f"Peak-hour MAE  : {peak_mae:.4f}"
    )

    print(
        f"Peak-hour RMSE : {peak_rmse:.4f}"
    )

    print(
        f"Peak-hour R²   : {peak_r2:.4f}"
    )


# ============================================================
# 5. HOLIDAY PERFORMANCE
# ============================================================

holiday_data = results[
    results["IsHoliday"] == 1
]


if len(holiday_data) > 0:

    holiday_mae = mean_absolute_error(
        holiday_data["Vehicles"],
        holiday_data["Predicted_Vehicles"]
    )

    holiday_rmse = np.sqrt(
        mean_squared_error(
            holiday_data["Vehicles"],
            holiday_data["Predicted_Vehicles"]
        )
    )

    holiday_r2 = r2_score(
        holiday_data["Vehicles"],
        holiday_data["Predicted_Vehicles"]
    )


    print("\n==============================================")
    print(" HOLIDAY PERFORMANCE")
    print("==============================================")

    print(
        f"Holiday MAE  : {holiday_mae:.4f}"
    )

    print(
        f"Holiday RMSE : {holiday_rmse:.4f}"
    )

    print(
        f"Holiday R²   : {holiday_r2:.4f}"
    )


# ============================================================
# 6. JUNCTION-WISE PERFORMANCE
# ============================================================

print("\n==============================================")
print(" JUNCTION-WISE PERFORMANCE")
print("==============================================")


junction_results = []


for junction in sorted(
    results["Junction"].unique()
):

    junction_data = results[
        results["Junction"] == junction
    ]


    junction_mae = mean_absolute_error(
        junction_data["Vehicles"],
        junction_data["Predicted_Vehicles"]
    )


    junction_rmse = np.sqrt(
        mean_squared_error(
            junction_data["Vehicles"],
            junction_data["Predicted_Vehicles"]
        )
    )


    junction_r2 = r2_score(
        junction_data["Vehicles"],
        junction_data["Predicted_Vehicles"]
    )


    print(
        f"\nJunction {junction}"
    )

    print(
        f"MAE  : {junction_mae:.4f}"
    )

    print(
        f"RMSE : {junction_rmse:.4f}"
    )

    print(
        f"R²   : {junction_r2:.4f}"
    )


    junction_results.append({

        "Junction": junction,

        "MAE": junction_mae,

        "RMSE": junction_rmse,

        "R2": junction_r2

    })


junction_results_df = pd.DataFrame(
    junction_results
)


junction_results_df.to_csv(
    "outputs/junction_model_results.csv",
    index=False
)


# ============================================================
# FINAL SAMPLE PREDICTIONS
# ============================================================

print("\n==============================================")
print(" SAMPLE PREDICTIONS")
print("==============================================")


print(
    results[
        [
            "DateTime",
            "Junction",
            "Vehicles",
            "Predicted_Vehicles",
            "Absolute_Error"
        ]
    ].head(10).to_string(index=False)
)


print("\n==============================================")
print(" MODEL EVALUATION COMPLETED")
print("==============================================")