import pandas as pd
import numpy as np
import os
import holidays


# ==========================================
# FILE PATHS
# ==========================================

INPUT_FILE = "data/train_aWnotuB.csv"

OUTPUT_FILE = "data/cleaned_traffic_data.csv"


# ==========================================
# LOAD DATA
# ==========================================

print("\n======================================")
print(" SMART CITY TRAFFIC FORECASTING")
print(" DATA PREPROCESSING")
print("======================================")

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")

print("Rows:", len(df))

print("Columns:", len(df.columns))


# ==========================================
# CHECK DUPLICATES
# ==========================================

print("\nChecking duplicate records...")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates()

    print("Duplicates removed.")


# ==========================================
# DATETIME CONVERSION
# ==========================================

print("\nConverting DateTime...")

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    errors="coerce"
)


# Remove invalid dates

invalid_dates = df["DateTime"].isnull().sum()

print(
    "Invalid DateTime values:",
    invalid_dates
)

df = df.dropna(
    subset=["DateTime"]
)


# ==========================================
# MISSING VALUES
# ==========================================

print("\nChecking missing values...")

print(
    df.isnull().sum()
)


# Remove missing important records

df = df.dropna(
    subset=[
        "DateTime",
        "Junction",
        "Vehicles"
    ]
)


# ==========================================
# SORT DATA
# ==========================================

df = df.sort_values(
    ["Junction", "DateTime"]
).reset_index(drop=True)


# ==========================================
# BASIC TIME FEATURES
# ==========================================

print("\nCreating time features...")


df["Year"] = (
    df["DateTime"].dt.year
)

df["Month"] = (
    df["DateTime"].dt.month
)

df["Day"] = (
    df["DateTime"].dt.day
)

df["Hour"] = (
    df["DateTime"].dt.hour
)

df["DayOfWeek"] = (
    df["DateTime"].dt.dayofweek
)


# ==========================================
# WEEKEND FEATURE
# ==========================================

df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)


# ==========================================
# PEAK HOUR FEATURE
# ==========================================

peak_hours = [
    7, 8, 9,
    17, 18, 19
]

df["IsPeakHour"] = (
    df["Hour"].isin(peak_hours)
).astype(int)


# ==========================================
# HOLIDAY CALENDAR
# ==========================================

print("\nCreating Indian holiday information...")


start_year = df["Year"].min()

end_year = df["Year"].max()


india_holidays = holidays.country_holidays(
    "IN",
    years=range(
        start_year,
        end_year + 1
    )
)


# Convert date to date-only

df["Date"] = (
    df["DateTime"].dt.date
)


# Holiday indicator

df["IsHoliday"] = (
    df["Date"].isin(
        india_holidays
    )
).astype(int)


# Holiday name

df["HolidayName"] = (
    df["Date"].map(
        lambda x:
        india_holidays.get(x, "Normal Day")
    )
)


# ==========================================
# WORKING DAY
# ==========================================

df["IsWorkingDay"] = (
    (
        (df["DayOfWeek"] < 5)
        &
        (df["IsHoliday"] == 0)
    )
).astype(int)


# ==========================================
# DAY TYPE
# ==========================================

def get_day_type(row):

    if row["IsHoliday"] == 1:

        return "Holiday"

    elif row["IsWeekend"] == 1:

        return "Weekend"

    else:

        return "Working Day"


df["DayType"] = df.apply(
    get_day_type,
    axis=1
)


# ==========================================
# LAG FEATURES
# ==========================================

print("\nCreating traffic history features...")


df["Lag_1"] = (
    df.groupby("Junction")["Vehicles"]
    .shift(1)
)


df["Lag_2"] = (
    df.groupby("Junction")["Vehicles"]
    .shift(2)
)


df["Lag_3"] = (
    df.groupby("Junction")["Vehicles"]
    .shift(3)
)


# ==========================================
# ROLLING TRAFFIC FEATURES
# ==========================================

df["Rolling_Mean_3"] = (

    df.groupby("Junction")["Vehicles"]

    .transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .mean()
    )
)


df["Rolling_Std_3"] = (

    df.groupby("Junction")["Vehicles"]

    .transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .std()
    )
)


# ==========================================
# REMOVE TEMPORARY DATE COLUMN
# ==========================================

df = df.drop(
    columns=["Date"]
)


# ==========================================
# SAVE DATA
# ==========================================

os.makedirs(
    "data",
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n======================================")

print("PREPROCESSING COMPLETED")

print("======================================")


print(
    "\nFinal dataset shape:",
    df.shape
)


print(
    "\nNew features created:"
)

print(
    [
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek",
        "IsWeekend",
        "IsPeakHour",
        "IsHoliday",
        "HolidayName",
        "IsWorkingDay",
        "DayType",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Rolling_Mean_3",
        "Rolling_Std_3"
    ]
)


print(
    "\nDay Type Distribution:"
)

print(
    df["DayType"].value_counts()
)


print(
    "\nHoliday Records:"
)

print(
    df["IsHoliday"].sum()
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nPreprocessing completed successfully!"
)