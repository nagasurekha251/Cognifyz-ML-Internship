import pandas as pd

# 1. LOAD DATASET
data = pd.read_csv("restaurant_data.csv")

print("Dataset loaded successfully!")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# 2. DISPLAY REQUIRED COLUMNS
print("\nRestaurant Name:")
print(data["Restaurant Name"].head())

print("\nCuisines:")
print(data["Cuisines"].head())

print("\nPrice Range:")
print(data["Price range"].head())

# 3. HANDLE MISSING VALUES
print("\nMissing values before preprocessing:")
print(data[["Restaurant Name", "Cuisines", "Price range"]].isnull().sum())

# Replace missing cuisine values
data["Cuisines"] = data["Cuisines"].fillna("Unknown")

print("\nMissing values after preprocessing:")
print(data[["Restaurant Name", "Cuisines", "Price range"]].isnull().sum())

# 4. CONVERT CUISINES INTO NUMERICAL FEATURES
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert cuisine information into numerical form
cuisine_matrix = vectorizer.fit_transform(data["Cuisines"])

print("\nCuisine features created successfully!")
print("Cuisine matrix shape:", cuisine_matrix.shape)

# 5. CALCULATE RESTAURANT SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between restaurants
similarity_matrix = cosine_similarity(cuisine_matrix)

print("\nSimilarity calculation completed!")
print("Similarity matrix shape:", similarity_matrix.shape)


# 6. CREATE RECOMMENDATION FUNCTION

def recommend_restaurants(restaurant_name, number_of_recommendations=5):

    # Find the restaurant
    restaurant_index = data[
        data["Restaurant Name"].str.lower() == restaurant_name.lower()
    ].index

    if len(restaurant_index) == 0:
        print("\nRestaurant not found!")
        return

    index = restaurant_index[0]
    selected_price = data.loc[index, "Price range"]

    # Get cuisine similarity scores
    similarity_scores = list(enumerate(similarity_matrix[index]))

    combined_scores = []

    for restaurant_idx, cuisine_score in similarity_scores:

        # Skip the selected restaurant itself
        if restaurant_idx == index:
            continue

        restaurant_price = data.iloc[restaurant_idx]["Price range"]

        # Give a bonus for matching price ranges
        price_score = 1 if restaurant_price == selected_price else 0

        # Combine cuisine and price scores
        final_score = (0.8 * cuisine_score) + (0.2 * price_score)

        combined_scores.append((restaurant_idx, final_score))

    # Sort by final score, highest first
    combined_scores.sort(key=lambda x: x[1], reverse=True)

    recommendations = combined_scores[:number_of_recommendations]

    print(f"\nRestaurants similar to '{restaurant_name}':")
    print(f"Selected Price Range: {selected_price}")

    for restaurant_idx, score in recommendations:
        restaurant = data.iloc[restaurant_idx]

        print(
            f"{restaurant['Restaurant Name']} | "
            f"Cuisine: {restaurant['Cuisines']} | "
            f"Price Range: {restaurant['Price range']} | "
            f"Similarity Score: {score:.2f}"
        )


# 7. TEST RECOMMENDATION SYSTEM
recommend_restaurants("Le Petit Souffle", 5)