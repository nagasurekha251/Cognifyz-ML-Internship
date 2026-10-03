# Cognifyz Technologies – Machine Learning Internship

This repository contains three machine learning projects completed as part of the Cognifyz Technologies Machine Learning Internship.

## Projects

### Task 1: Restaurant Rating Prediction

**Folder:** `Task_1_Restaurant_Ratings`

* Predicts restaurant aggregate ratings using restaurant-related features.
* Uses a Random Forest Regressor.
* Evaluates performance using Mean Squared Error (MSE) and R² score.

**Results:**

* MSE: 0.133
* R² Score: 0.9416

### Task 2: Restaurant Recommendation System

**Folder:** `Task_2_Restaurant_Recommendation`

* Recommends restaurants based on cuisine similarity.
* Uses TF-IDF vectorization and cosine similarity.
* Considers price range when calculating recommendation scores.

### Task 3: Cuisine Classification

**Folder:** `Task_3_Cuisine_Classification`

* Predicts the main cuisine category of a restaurant.
* Uses a Random Forest Classifier.
* Evaluates performance using accuracy, weighted precision, and weighted recall.

**Results:**

* Accuracy: 19.95%
* Weighted Precision: 29.19%
* Weighted Recall: 19.95%

The classification model is an initial experiment and requires further improvement.

## Dataset

The project uses `restaurant_data.csv`, which contains restaurant information such as cuisines, average cost, price range, ratings, votes, and booking options.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* TF-IDF Vectorization
* Cosine Similarity
* Random Forest Regression and Classification

## Project Structure

```text
ML/
├── restaurant_data.csv
├── Task_1_Restaurant_Ratings/
│   └── rating_prediction.py
├── Task_2_Restaurant_Recommendation/
│   └── restaurant_recommendation.py
├── Task_3_Cuisine_Classification/
│   └── cuisine_classification.py
└── README.md
```

## Installation and Execution

1. Install Python.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment if needed.
4. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

5. Run a task from the main project folder:

```bash
python Task_1_Restaurant_Ratings/rating_prediction.py
python Task_2_Restaurant_Recommendation/restaurant_recommendation.py
python Task_3_Cuisine_Classification/cuisine_classification.py
```

## Conclusion

These projects demonstrate regression, content-based recommendation, and classification techniques using Python and machine learning.
