import pandas as pd

# Load the restaurant dataset
data = pd.read_csv("restaurant_data.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# Display column names
print("\nColumns in the dataset:")
print(data.columns.tolist())

# Check missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Handle missing values in Cuisines
data["Cuisines"] = data["Cuisines"].fillna("Unknown")

print("\nMissing values after preprocessing:")
print(data.isnull().sum())

# Select features for prediction
features = [
    "Average Cost for two",
    "Price range",
    "Votes",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now"
]

target = "Aggregate rating"

X = data[features]
y = data[target]

print("\nFeatures selected:")
print(features)

print("\nTarget:")
print(target)

print("\nFeature data:")
print(X.head())

print("\nTarget data:")
print(y.head())

# Convert Yes/No values into numbers
binary_columns = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now"
]

for column in binary_columns:
    X[column] = X[column].map({"Yes": 1, "No": 0})

print("\nFeatures after converting Yes/No to numbers:")
print(X.head())

#split the data(traing and testing data)
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", X_train.shape[0])
print("Testing data size:", X_test.shape[0])

from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Make predictions on the testing data
y_pred = model.predict(X_test)

print("\nPredictions made successfully!")

# Display the first 5 actual and predicted ratings
print("\nActual vs Predicted Ratings:")
for actual, predicted in zip(y_test.head(5), y_pred[:5]):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

from sklearn.metrics import mean_squared_error, r2_score

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 4))
print("R² Score:", round(r2, 4))

# Calculate feature importance
importance = model.feature_importances_

# Create a table of features and their importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)