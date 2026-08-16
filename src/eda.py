import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==========================================
# FILE PATHS
# ==========================================

DATA_FILE = "data/cleaned_traffic_data.csv"

OUTPUT_DIR = "outputs/graphs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

df["DateTime"] = pd.to_datetime(
    df["DateTime"]
)


sns.set_theme(
    style="whitegrid"
)


# ==========================================
# 1. TRAFFIC BY JUNCTION
# ==========================================

junction_traffic = (
    df.groupby("Junction")["Vehicles"]
    .mean()
)


plt.figure(figsize=(8, 5))

junction_traffic.plot(
    kind="bar"
)

plt.title(
    "Average Traffic by Junction"
)

plt.xlabel(
    "Junction"
)

plt.ylabel(
    "Average Vehicles"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/traffic_by_junction.png"
)

plt.show()


# ==========================================
# 2. HOURLY TRAFFIC
# ==========================================

hourly_traffic = (
    df.groupby("Hour")["Vehicles"]
    .mean()
)


plt.figure(figsize=(10, 5))

plt.plot(
    hourly_traffic.index,
    hourly_traffic.values,
    marker="o"
)

plt.title(
    "Average Traffic by Hour"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Average Vehicles"
)

plt.xticks(
    range(24)
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/traffic_by_hour.png"
)

plt.show()


# ==========================================
# 3. MONTHLY TRAFFIC
# ==========================================

monthly_traffic = (
    df.groupby("Month")["Vehicles"]
    .mean()
)


plt.figure(figsize=(9, 5))

plt.plot(
    monthly_traffic.index,
    monthly_traffic.values,
    marker="o"
)

plt.title(
    "Average Traffic by Month"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Average Vehicles"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/traffic_by_month.png"
)

plt.show()


# ==========================================
# 4. DAY OF WEEK
# ==========================================

day_traffic = (
    df.groupby("DayOfWeek")["Vehicles"]
    .mean()
)


day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


plt.figure(figsize=(10, 5))

plt.bar(
    range(7),
    day_traffic.reindex(range(7))
)

plt.xticks(
    range(7),
    day_names,
    rotation=30
)

plt.title(
    "Average Traffic by Day of Week"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Average Vehicles"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/traffic_by_day.png"
)

plt.show()


# ==========================================
# 5. VEHICLE DISTRIBUTION
# ==========================================

plt.figure(figsize=(9, 5))

sns.histplot(
    df["Vehicles"],
    bins=40,
    kde=True
)

plt.title(
    "Distribution of Vehicle Count"
)

plt.xlabel(
    "Vehicles"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/vehicle_distribution.png"
)

plt.show()


# ==========================================
# 6. JUNCTION VS VEHICLES
# ==========================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Junction",
    y="Vehicles"
)

plt.title(
    "Vehicle Count by Junction"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/junction_boxplot.png"
)

plt.show()


# ==========================================
# 7. CORRELATION HEATMAP
# ==========================================

correlation_features = [
    "Junction",
    "Vehicles",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "IsPeakHour",
    "IsHoliday",
    "IsWorkingDay"
]


correlation = (
    df[correlation_features]
    .corr()
)


plt.figure(
    figsize=(12, 9)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Traffic Feature Correlation"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png"
)

plt.show()


# ==========================================
# 8. WORKING DAY vs WEEKEND vs HOLIDAY
# ==========================================

day_type_traffic = (
    df.groupby("DayType")["Vehicles"]
    .mean()
)


order = [
    "Working Day",
    "Weekend",
    "Holiday"
]


plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="DayType",
    y="Vehicles",
    order=order
)

plt.title(
    "Average Traffic: Working Day vs Weekend vs Holiday"
)

plt.xlabel(
    "Day Type"
)

plt.ylabel(
    "Average Vehicles"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/day_type_comparison.png"
)

plt.show()


# ==========================================
# 9. HOLIDAY vs WORKING DAY BY JUNCTION
# ==========================================

comparison = (
    df[
        df["DayType"].isin(
            ["Working Day", "Holiday"]
        )
    ]
    .groupby(
        ["Junction", "DayType"]
    )["Vehicles"]
    .mean()
    .reset_index()
)


plt.figure(figsize=(10, 5))

sns.barplot(
    data=comparison,
    x="Junction",
    y="Vehicles",
    hue="DayType"
)

plt.title(
    "Holiday vs Working-Day Traffic by Junction"
)

plt.xlabel(
    "Junction"
)

plt.ylabel(
    "Average Vehicles"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/holiday_vs_working_junction.png"
)

plt.show()


# ==========================================
# 10. HOLIDAY TRAFFIC BY HOUR
# ==========================================

holiday_hourly = (
    df[
        df["IsHoliday"] == 1
    ]
    .groupby("Hour")["Vehicles"]
    .mean()
)


plt.figure(figsize=(10, 5))

plt.plot(
    holiday_hourly.index,
    holiday_hourly.values,
    marker="o"
)

plt.title(
    "Traffic Pattern by Hour on Holidays"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Average Vehicles"
)

plt.xticks(
    range(24)
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/holiday_hourly_traffic.png"
)

plt.show()


# ==========================================
# 11. WORKING DAY vs HOLIDAY BY HOUR
# ==========================================

hour_comparison = (
    df[
        df["DayType"].isin(
            ["Working Day", "Holiday"]
        )
    ]
    .groupby(
        ["Hour", "DayType"]
    )["Vehicles"]
    .mean()
    .reset_index()
)


plt.figure(figsize=(11, 5))

sns.lineplot(
    data=hour_comparison,
    x="Hour",
    y="Vehicles",
    hue="DayType",
    marker="o"
)

plt.title(
    "Hourly Traffic: Working Days vs Holidays"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Average Vehicles"
)

plt.xticks(
    range(24)
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/working_vs_holiday_hourly.png"
)

plt.show()


# ==========================================
# PRINT IMPORTANT FINDINGS
# ==========================================

print("\n======================================")
print(" TRAFFIC ANALYSIS RESULTS")
print("======================================")


print("\nAverage Traffic by Junction:")

print(
    junction_traffic.round(2)
)


print(
    "\nBusiest Junction:"
)

print(
    junction_traffic.idxmax()
)


print(
    "\nHighest Traffic Hour:"
)

print(
    hourly_traffic.idxmax()
)


print(
    "\nTraffic by Day Type:"
)

print(
    day_type_traffic.round(2)
)


# Holiday busiest junction

holiday_junction = (
    df[
        df["IsHoliday"] == 1
    ]
    .groupby("Junction")["Vehicles"]
    .mean()
)


print(
    "\nBusiest Junction on Holidays:"
)

print(
    holiday_junction.idxmax()
)


# Holiday peak hour

if len(holiday_hourly) > 0:

    print(
        "\nPeak Holiday Traffic Hour:"
    )

    print(
        holiday_hourly.idxmax()
    )


# ==========================================
# COMPLETED
# ==========================================

print(
    "\nEDA completed successfully!"
)

print(
    f"Graphs saved in: {OUTPUT_DIR}"
)