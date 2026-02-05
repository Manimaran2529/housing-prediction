import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
house = pd.read_csv("Housing.csv")

# Handle missing values
house[["bathrooms", "stories"]] = house[["bathrooms", "stories"]].fillna(2)
house["airconditioning"] = house["airconditioning"].fillna("yes")

# Encode yes/no categorical columns
cat_yes_no = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]
house[cat_yes_no] = house[cat_yes_no].replace({"yes": 1, "no": 0})

# One-hot encode furnishingstatus
house = pd.get_dummies(house, columns=["furnishingstatus"], dtype=int)

# Features and target
X = house.drop(columns=["price"])
y = house["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model (tuned)
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42
)

# Train model
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
