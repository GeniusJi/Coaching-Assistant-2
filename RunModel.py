import pickle
import pandas as pd

with open("random_forest.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
features = saved["features"]

minutes = float(input("Minutes: "))

raw = {
    "Touches": float(input("Touches: ")),
    "Shots": float(input("Shots: ")),
    "Progressive Carries": float(input("Progressive Carries: ")),
    "Interceptions": float(input("Interceptions: ")),
    "fThird Passes": float(input("Final-third passes: ")),
    "Successful fThird Passes": float(input("Successful final-third passes: ")),
    "Crosses": float(input("Crosses: ")),
    "Successful Crosses": float(input("Successful crosses: ")),
    "Ground Duels": float(input("Ground duels: ")),
    "gDuels Won": float(input("Ground duels won: "))
}

def rate(successful, attempted):
    return successful / attempted if attempted else 0

player = {
    "Touches": raw["Touches"] / minutes * 90,
    "Shots": raw["Shots"] / minutes * 90,
    "Progressive Carries": raw["Progressive Carries"] / minutes * 90,
    "Interceptions": raw["Interceptions"] / minutes * 90,
    "fThird Passes": raw["fThird Passes"] / minutes * 90,
    "Successful fThird Passes%": rate(
        raw["Successful fThird Passes"],
        raw["fThird Passes"]
    ),
    "Crosses": raw["Crosses"] / minutes * 90,
    "Successful Crosses%": rate(
        raw["Successful Crosses"],
        raw["Crosses"]
    ),
    "Ground Duels": raw["Ground Duels"] / minutes * 90,
    "gDuels Won%": rate(
        raw["gDuels Won"],
        raw["Ground Duels"]
    )
}

X = pd.DataFrame([player])[features]

prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print("\nPredicted position:", prediction)
print("\nProbabilities:")

for position, probability in sorted(
    zip(model.classes_, probabilities),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{position}: {probability:.2%}")