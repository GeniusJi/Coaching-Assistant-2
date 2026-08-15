import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "filtered_data.csv")
features = df.select_dtypes("number").columns

x = df[features]
y = df["Position"]

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(x, y)

with open("random_forest.pkl", "wb") as f:
    pickle.dump(
        {"model": model, "features": features.tolist()},
        f
    )