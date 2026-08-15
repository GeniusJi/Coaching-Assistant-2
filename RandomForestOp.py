import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "filtered_data.csv")
features = df.select_dtypes("number").columns
X_train, X_test, y_train, y_test = train_test_split(
df[features], df.Position, test_size=0.2, random_state=42, stratify=df.Position)

best_accuracy = 0
best_settings = None

for trees in [100, 300, 500, 700, 1000]:
    for depth in [5, 10, 15, 20, None]:
        for min_split in [2, 5, 10]:
            for min_leaf in [1, 2, 4]:

                model = RandomForestClassifier(
                    n_estimators=trees,
                    max_depth=depth,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_settings = (trees, depth, min_split, min_leaf)

                print(f"Trees: {trees}, Depth: {depth}, Min Split: {min_split}, Min Leaf: {min_leaf} => Accuracy: {accuracy:.2%}")

print("Best accuracy:", f"{best_accuracy:.2%}")
print("Best settings:")
print("Trees:", best_settings[0])
print("Depth:", best_settings[1])
print("Minimum split:", best_settings[2])
print("Minimum leaf:", best_settings[3])