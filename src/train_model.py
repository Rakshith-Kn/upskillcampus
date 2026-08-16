import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SMART CITY TRAFFIC FORECASTING
# HOLIDAY-AWARE MACHINE LEARNING MODEL
# ============================================================


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

DATA_FILE = "data/cleaned_traffic_data.csv"

MODEL_DIR = "models"

RESULT_FILE = "outputs/model_results.csv"


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n==============================================")
print(" SMART CITY TRAFFIC FORECASTING")
print(" HOLIDAY-AWARE MODEL TRAINING")
print("==============================================")

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_FILE)

df["DateTime"] = pd.to_datetime(df["DateTime"])

print("Dataset loaded successfully.")

print("Total records:", len(df))


# ------------------------------------------------------------
# SORT BY TIME
# ------------------------------------------------------------

# Important for forecasting:
# Earlier observations should be used for training
# and later observations should be used for testing.

df = df.sort_values(
    "DateTime"
).reset_index(drop=True)


# ------------------------------------------------------------
# CHECK REQUIRED FEATURES
# ------------------------------------------------------------

features = [

    # Location
    "Junction",

    # Calendar features
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",

    # Day-type information
    "IsWeekend",
    "IsPeakHour",
    "IsHoliday",
    "IsWorkingDay",

    # Recent traffic history
    "Lag_1",
    "Lag_2",
    "Lag_3",

    # Rolling traffic statistics
    "Rolling_Mean_3",
    "Rolling_Std_3"
]


target = "Vehicles"


missing_features = [
    column
    for column in features
    if column not in df.columns
]


if missing_features:

    print("\nERROR!")

    print(
        "The following required columns are missing:"
    )

    print(missing_features)

    print("\nAvailable columns:")

    print(df.columns.tolist())

    raise SystemExit


# ------------------------------------------------------------
# REMOVE MISSING VALUES
# ------------------------------------------------------------

print("\nChecking missing values in model features...")

print(
    df[features + [target]].isnull().sum()
)


# Lag and rolling features naturally create
# missing values at the beginning of each junction.

df = df.dropna(
    subset=features + [target]
).reset_index(drop=True)


print(
    "\nRecords available for modelling:",
    len(df)
)


# ------------------------------------------------------------
# INPUT AND TARGET
# ------------------------------------------------------------

X = df[features]

y = df[target]


print("\nInput features:")

for feature in features:

    print(" -", feature)


print("\nTarget variable:")

print(" -", target)


# ------------------------------------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# ------------------------------------------------------------

split_index = int(
    len(df) * 0.80
)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\n==============================================")
print(" TIME-BASED DATA SPLIT")
print("==============================================")

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


print(
    "\nTraining period:"
)

print(
    df["DateTime"].iloc[0],
    "to",
    df["DateTime"].iloc[split_index - 1]
)


print(
    "\nTesting period:"
)

print(
    df["DateTime"].iloc[split_index],
    "to",
    df["DateTime"].iloc[-1]
)


# ------------------------------------------------------------
# MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {

    "Linear Regression":

        LinearRegression(),


    "Decision Tree":

        DecisionTreeRegressor(
            max_depth=15,
            random_state=42
        ),


    "Random Forest":

        RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
}


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

results = []

best_model = None

best_model_name = None

best_r2 = -float("inf")


print("\n==============================================")
print(" MODEL TRAINING")
print("==============================================")


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # Train model

    model.fit(
        X_train,
        y_train
    )


    # Generate predictions

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

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


    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )


    # Store results

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })


    # Select best model

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

        best_model_name = name


# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="R2",
    ascending=False
).reset_index(drop=True)


print("\n==============================================")
print(" MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SAVE MODEL RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    RESULT_FILE,
    index=False
)


print(
    "\nModel comparison saved to:"
)

print(
    RESULT_FILE
)


# ------------------------------------------------------------
# SAVE BEST MODEL
# ------------------------------------------------------------

model_path = (
    f"{MODEL_DIR}/traffic_model.pkl"
)


joblib.dump(
    best_model,
    model_path
)


# Save feature names

joblib.dump(
    features,
    f"{MODEL_DIR}/feature_names.pkl"
)


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n==============================================")

print(
    " BEST MODEL"
)

print("==============================================")


print(
    "Model:",
    best_model_name
)


print(
    f"R² Score: {best_r2:.4f}"
)


print(
    "\nSaved model:"
)

print(
    model_path
)


print(
    "\nFeature information saved:"
)

print(
    f"{MODEL_DIR}/feature_names.pkl"
)


print("\n==============================================")

print(
    " MACHINE LEARNING COMPLETED SUCCESSFULLY"
)

print("==============================================")