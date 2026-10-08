import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


data = {
    "temperature_c": [22, 24, 26, 28, 25, 23, 27, 29, 21, 30, 19, 31, 20, 32, 18, 27],
    "relative_humidity_pct": [72, 75, 78, 90, 80, 74, 88, 93, 70, 95, 68, 96, 69, 97, 66, 89],
    "rainfall_rate_mmph": [8, 10, 14, 28, 16, 9, 24, 32, 6, 35, 5, 36, 7, 38, 4, 25],
    "wind_speed_kmph": [18, 20, 24, 35, 22, 19, 31, 38, 17, 41, 15, 42, 16, 45, 14, 30],
    "surface_pressure_hpa": [1008, 1006, 1004, 996, 1002, 1007, 998, 994, 1010, 992, 1012, 991, 1011, 989, 1013, 997],
    "pressure_drop_3h_hpa": [1.2, 1.8, 2.4, 5.6, 2.8, 1.5, 4.9, 6.2, 1.0, 6.8, 0.9, 7.0, 1.1, 7.4, 0.8, 5.1],
    "cloudburst": [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}

df = pd.DataFrame(data)

feature_columns = [
    "temperature_c",
    "relative_humidity_pct",
    "rainfall_rate_mmph",
    "wind_speed_kmph",
    "surface_pressure_hpa",
    "pressure_drop_3h_hpa",
]

X = df[feature_columns]
y = df["cloudburst"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=3))

new_conditions = pd.DataFrame(
    [
        {
            "temperature_c": 28,
            "relative_humidity_pct": 92,
            "rainfall_rate_mmph": 30,
            "wind_speed_kmph": 39,
            "surface_pressure_hpa": 993,
            "pressure_drop_3h_hpa": 6.3,
        }
    ]
)

prediction = model.predict(new_conditions)[0]
probability = model.predict_proba(new_conditions)[0][1]

print("New-condition prediction:", "Cloudburst likely" if prediction == 1 else "Cloudburst unlikely")
print("Predicted cloudburst probability:", round(float(probability), 3))
