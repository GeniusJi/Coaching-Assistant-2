import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "filtered_data.csv")
features = df.select_dtypes("number").columns
X_train, X_test, y_train, y_test = train_test_split(
df[features], df.Position, test_size=0.2, random_state=42, stratify=df.Position)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

results = df.loc[X_test.index, ["Player Name", "Position"]].copy()
results["Predicted"] = pred
results["Correct"] = results.Position == results.Predicted

print(f"\nAccuracy: {accuracy_score(y_test, pred):.2%}\n")
print(classification_report(y_test, pred))
print("Test results:\n", results.to_string(index=False))

reliance = pd.Series(model.feature_importances_ * 100, index=features).sort_values(ascending=False)
print("\nParameter reliance (%):\n", reliance.round(2).to_string())