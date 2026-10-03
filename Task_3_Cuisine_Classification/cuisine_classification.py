import pandas as pd

# 1. LOAD DATASET
data = pd.read_csv("restaurant_data.csv")

print("Dataset loaded successfully!")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# 2. DISPLAY CUISINE INFORMATION
print("\nCuisine column:")
print(data["Cuisines"].head(10))

print("\nNumber of missing cuisine values:")
print(data["Cuisines"].isnull().sum())

print("\nMost common cuisine entries:")
print(data["Cuisines"].value_counts().head(10))

# 3. PREPROCESS CUISINE DATA
# Remove rows where cuisine information is missing
data = data.dropna(subset=["Cuisines"]).copy()

# Remove extra spaces from cuisine names
data["Cuisines"] = data["Cuisines"].str.strip()

print("\nMissing values after preprocessing:")
print(data["Cuisines"].isnull().sum())

print("\nDataset size after preprocessing:")
print(data.shape)

# Count unique cuisine combinations
print("\nNumber of unique cuisine combinations:")
print(data["Cuisines"].nunique())

# 4. EXTRACT THE MAIN CUISINE
# Use the first cuisine as the main cuisine
data["Cuisine"] = data["Cuisines"].str.split(",").str[0].str.strip()

print("\nNumber of unique main cuisines:")
print(data["Cuisine"].nunique())

print("\nMost common main cuisines:")
print(data["Cuisine"].value_counts().head(15))

# 5. FILTER RARE CUISINE CATEGORIES
# Count restaurants in each main cuisine
cuisine_counts = data["Cuisine"].value_counts()

# Keep cuisines that have at least 20 restaurants
common_cuisines = cuisine_counts[cuisine_counts >= 20].index

data = data[data["Cuisine"].isin(common_cuisines)].copy()

print("\nDataset size after filtering rare cuisines:")
print(data.shape)

print("\nNumber of cuisine categories for classification:")
print(data["Cuisine"].nunique())

print("\nCuisine categories:")
print(data["Cuisine"].value_counts())

# 6. SELECT FEATURES AND TARGET
# Input features used to predict cuisine
features = [
    "Average Cost for two",
    "Price range",
    "Votes",
    "Aggregate rating",
    "Has Table booking",
    "Has Online delivery"
]

X = data[features].copy()
y = data["Cuisine"].copy()

# Convert Yes/No values into numbers
X["Has Table booking"] = X["Has Table booking"].map(
    {"Yes": 1, "No": 0}
)

X["Has Online delivery"] = X["Has Online delivery"].map(
    {"Yes": 1, "No": 0}
)

# Fill missing input values with the median
X = X.fillna(X.median())

# 7. SPLIT TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Convert cuisine names into numerical labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of input features:", X_train.shape[1])
print("Number of cuisine categories:", len(encoder.classes_))

# 8. TRAIN THE CLASSIFICATION MODEL
from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced_subsample"
)

# Train the model
model.fit(X_train, y_train)

print("\nCuisine classification model trained successfully!")

# 9. EVALUATE THE MODEL

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Predict cuisines for the testing data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)

# Display results
print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)