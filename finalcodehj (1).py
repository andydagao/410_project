#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if not hasattr(pd, "Int64Index"):
    pd.Int64Index = pd.Index
if not hasattr(pd, "Float64Index"):
    pd.Float64Index = pd.Index

import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression 

#load data
survey_df = pd.read_csv("maindataset.csv")
unemp = pd.read_excel("Unemployment Rate.xlsx")
cpi = pd.read_excel("cpi food at home.xlsx")
fi = pd.read_excel("food_insecurity_rates_usa.xlsx")

#build yearly macro level table: unemployment, inflation,and food insecurity rate
unemp_yearly = unemp[["Year", "avg_unemployment_rate"]].copy()

cpi_yearly = (
    cpi[["Year", "cpi_avg"]]
    .sort_values("Year")
    .assign(inflation_rate=lambda d: d["cpi_avg"].pct_change() * 100)
)

fi_yearly = (
    fi[["Year", "Food insecurity in households with children"]]
    .rename(
        columns={
            "Food insecurity in households with children": "food_insecurity_rate"
        }
    )
)

macro_df = (
    unemp_yearly
    .merge(cpi_yearly[["Year", "inflation_rate"]], on="Year")
    .merge(fi_yearly, on="Year")
    .query("Year >= 2015")
    .dropna()
)

#correlation heat map
corr = macro_df.drop(columns="Year").corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, interpolation="nearest")
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
plt.colorbar(label="Pearson r")
plt.title("unemployment, inflation,and food Insecurity Correlation (2015-2024)")
plt.tight_layout()
plt.show()

print("correlation matrix:\n", corr.round(3), "\n")

#time aware train/test split + OLS regression on yearly macro data
train_df = macro_df[macro_df["Year"] <= 2019]
test_df = macro_df[macro_df["Year"] >= 2020]

X_train = train_df[["avg_unemployment_rate", "inflation_rate"]]
y_train = train_df["food_insecurity_rate"]

X_test = test_df[["avg_unemployment_rate", "inflation_rate"]]
y_test = test_df["food_insecurity_rate"]

X_train_const = X_train.copy()
X_train_const.insert(0, "const", 1.0)

X_test_const = X_test.copy()
X_test_const.insert(0, "const", 1.0)

ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())

y_pred = ols_model.predict(X_test_const)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f"\nTest RMSE: {rmse:.2f}")

#plot predicted vs actual
years = test_df["Year"].to_numpy()
actual = y_test.to_numpy()
pred = np.asarray(y_pred)

plt.figure(figsize=(6, 4))
plt.plot(years, actual, marker="o", label="Actual")
plt.plot(years, pred, marker="o", linestyle="--", label="Predicted")
plt.xlabel("Year")
plt.ylabel("Food-insecurity rate (%)")
plt.title("Linear regression — test-set fit (2020-2024)")
plt.legend()
plt.tight_layout()
plt.show()

#  binary target 
secure_counts = (
    survey_df["Enough of the kinds of food wanted"]
    + survey_df["Enough Food, but not always the kinds wanted"]
)
insecure_counts = (
    survey_df["Sometimes not enough to eat"]
    + survey_df["Often not enough to eat"]
)
share_insecure = insecure_counts / (secure_counts + insecure_counts)

threshold = share_insecure.median(skipna=True)
survey_df["food_insecure"] = (
    share_insecure.fillna(threshold) >= threshold
).astype(int)

#drop columns we don't want as features
drop_cols = [
    "Enough of the kinds of food wanted",
    "Enough Food, but not always the kinds wanted",
    "Sometimes not enough to eat",
    "Often not enough to eat",
    "Did not report",
    "week_name",
    "Year",
    "Location",
]

existing_drop_cols = [c for c in drop_cols if c in survey_df.columns]
survey_df = survey_df.drop(columns=existing_drop_cols)

#basic missing value handling and label encoding categoricals
for col in survey_df.columns:
    if survey_df[col].dtype == "O":
        survey_df[col] = survey_df[col].fillna("Unknown")
    else:
        survey_df[col] = pd.to_numeric(survey_df[col], errors="coerce")
        survey_df[col] = survey_df[col].fillna(survey_df[col].median())

label_encoders = {}
for col in survey_df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    survey_df[col] = le.fit_transform(survey_df[col].astype(str))
    label_encoders[col] = le

# 4) Features / target for RF
X = survey_df.drop(columns=["food_insecure"])
y = survey_df["food_insecure"]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#random Forest model
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

clf.fit(X_train_rf, y_train_rf)

y_pred_rf = clf.predict(X_test_rf)

print("classification report:\n")
print(classification_report(y_test_rf, y_pred_rf))

print("confusion matrix:")
print(confusion_matrix(y_test_rf, y_pred_rf), "\n")

#feature importances plot
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("feature importances for predicting food insecurity")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_features, rotation=45, ha="right")
plt.tight_layout()
plt.show()
