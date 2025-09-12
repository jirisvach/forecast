import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# Load
# Nainstalovat openpyxl pres pip
df = pd.read_excel("W:\Python\HelloWorld\spend_trends.xlsx")

# Sort
df = df.sort_values("rep_date")

# Nove sloupce
# 0 -> Pondeli, 6 -> Nedele
df["day_of_week"] = df["rep_date"].dt.weekday      # 0=Mon, 6=Sun
df["day_of_month"] = df["rep_date"].dt.day
df["month"] = df["rep_date"].dt.month

# Lagy pro time - series model, rolling avg pro uhlazeni dat a kratkodoby trend
df["lag_1"] = df["Spend"].shift(1)
df["lag_7"] = df["Spend"].shift(7)
df["lag_30"] = df["Spend"].shift(30)
df["rolling_7"] = df["Spend"].rolling(window=7).mean()

# Lagy vytvorily NA hodnoty
df = df.dropna()

# Features (X), target (y)
X = df[["day_of_week", "day_of_month", "month",
        "lag_1", "lag_7", "lag_30", "rolling_7"]]
y = df["Spend"]

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# Forecast na 15 dni
future_preds = []
last_known = df.copy()

for i in range(1, 16):
    last_date = last_known["rep_date"].max()
    next_date = last_date + timedelta(days=1)

    # Vypoocet indexu - po-pa = 1, vikend 0,75
    # Upravit logiku nebo pouzit jiz hotove indexy
    day_index = 1 if next_date.weekday() < 5 else 0.75

    # Prepare features for next_date
    features = {
        "rep_date": next_date,
        "day_of_week": next_date.weekday(),
        "day_of_month": next_date.day,
        "month": next_date.month,
        "lag_1": last_known.iloc[-1]["Spend"],
        "lag_7": last_known.iloc[-7]["Spend"],
        "lag_30": last_known.iloc[-30]["Spend"] if len(last_known) >= 30 else last_known.iloc[-1]["Spend"],
        "rolling_7": last_known["Spend"].tail(7).mean(),
        "day_index": day_index
    }

    feat_df = pd.DataFrame([features])
    pred = model.predict(feat_df[["day_of_week", "day_of_month",
                         "month", "lag_1", "lag_7", "lag_30", "rolling_7"]])[0]

    # Roznasobeni forecastu s dennim indexem
    features["Spend"] = pred * day_index

    # Append k forecast listu
    last_known = pd.concat(
        [last_known, pd.DataFrame([features])], ignore_index=True)
    future_preds.append(features)

# Preklopeni listu do dataframu
future_df = pd.DataFrame(future_preds)

# Slouceni puvodniho df s forecastem a save do excelu
output = pd.concat([df, future_df], ignore_index=True)
output.to_excel("W:\Python\HelloWorld\spend_trends_forecast.xlsx", index=False)

print("Done spend_trends_forecast.xlsx")


# verze s jiz vyplnenymi indexy
# je zde predpoklad, ze date a index column je vyplnena na dalsich X dni, spend je null
# Pouzit stejny import jako nahore

# Load
df = pd.read_excel("spend_trends_null_spend.xlsx")

# Sort
df = df.sort_values("rep_date").reset_index(drop=True)

# Nove sloupce
# 0 -> Pondeli, 6 -> Nedele
df["day_of_week"] = df["rep_date"].dt.weekday
df["day_of_month"] = df["rep_date"].dt.day
df["month"] = df["rep_date"].dt.month

# Lagy pro time - series model, rolling avg pro uhlazeni dat a kratkodoby trend
df["lag_1"] = df["Spend"].shift(1)
df["lag_7"] = df["Spend"].shift(7)
df["lag_30"] = df["Spend"].shift(30)
df["rolling_7"] = df["Spend"].rolling(window=7).mean()

# Urceni hodnot
# Pro train pouzijeme jen data, kde je vyplneny spend
train_df = df.dropna(subset=["Spend"]).dropna()
X_train = train_df[["day_of_week", "day_of_month",
                    "month", "lag_1", "lag_7", "lag_30", "rolling_7"]]
y_train = train_df["Spend"]

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Dynamicky forecast - lagy se pocitaji po kazdem loopu
# Neni specifikovany pocet loopu, ridi se poctem hodnot v rep_Date sloupci
for i in df.index:
    if pd.isna(df.loc[i, "Spend"]):  # Pojede tak dlouho, dokud budou ve spendu NA hodnoty
        # Update lagu po kazdem loopu
        df.loc[i, "lag_1"] = df.loc[i-1, "Spend"]
        df.loc[i, "lag_7"] = df.loc[i-7,
                                    "Spend"] if i >= 7 else df.loc[i-1, "Spend"]
        df.loc[i, "lag_30"] = df.loc[i-30,
                                     "Spend"] if i >= 30 else df.loc[i-1, "Spend"]
        df.loc[i, "rolling_7"] = df.loc[max(0, i-7):i-1, "Spend"].mean()

        # Build feature row
        features = df.loc[i, ["day_of_week", "day_of_month", "month",
                              "lag_1", "lag_7", "lag_30", "rolling_7"]].to_frame().T

        # Predict raw spend
        raw_pred = model.predict(features)[0]

        # Pouziti sloupce s jiz vyplnenym indexem
        final_pred = raw_pred * df.loc[i, "day_index"]

        # update hodnot v hlavnim DF
        df.loc[i, "Raw_Prediction"] = raw_pred
        df.loc[i, "Spend"] = final_pred

# Export do Excelu
df.to_excel("spend_forecast_with_custom_index_dynamic.xlsx", index=False)

print("Forecast saved to spend_forecast_with_custom_index_dynamic.xlsx")
