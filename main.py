"""
Dog Breed Recommendation System
-------------------------------
This program analyzes a dog breed dataset to:
1. Cluster similar dog breeds together
2. Predict high-quality dog breeds using machine learning
3. Provide personalized recommendations based on user preferences

The analysis uses multiple methods (clustering, classification, recommendation)
to solve the business problem of matching potential dog owners with suitable breeds.
"""

# Import necessary libraries for data manipulation, visualization and modeling
import pandas as pd  # Data manipulation
import numpy as np   # Numerical operations
import matplotlib.pyplot as plt  # Plotting and visualization
import seaborn as sns  # Enhanced visualizations
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # ML validation tools
from sklearn.ensemble import RandomForestClassifier  # Tree-based ensemble model
from sklearn.linear_model import LogisticRegression  # Linear classification model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Model evaluation
from sklearn.preprocessing import StandardScaler  # Feature normalization
from sklearn.cluster import KMeans  # Clustering algorithm
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.feature_selection import SelectFromModel  # Feature selection technique

#----------------------------------------------
# PART 1: DATA LOADING AND EXPLORATION
#----------------------------------------------

# Load the dog breed dataset from CSV file
print("Loading dataset...")
df = pd.read_csv("dogs-ranking-dataset.csv")

# Display basic dataset information to understand its structure
print("Dataset shape:", df.shape)
print("Summary statistics:")
print(df.describe())

# Visualize missing values to identify data quality issues
print("Checking for missing values...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values in Dataset")
plt.savefig('missing_values.png')  # Save visualization for the report

# Examine categorical values in key columns to understand the data
print("Unique intelligence values:")
print(df['intelligence'].unique())
print("\nUnique grooming frequency values:")
print(df['GROOMING FREQUNCY'].unique())

# Placeholder for mapping categorical values to numerical scores
# These would be filled after examining the actual values in the dataset
intelligence_map = {}  # Intelligence category to numerical score mapping
grooming_map = {}      # Grooming frequency to numerical score mapping

#----------------------------------------------
# PART 2: DATA PREPROCESSING
#----------------------------------------------

# Clean and convert monetary columns by removing currency symbols and commas
print("Preprocessing numeric columns...")
df['LIFETIME_COST'] = pd.to_numeric(df['$LIFETIME COST'].replace(r'[\$,%,]', '', regex=True), errors='coerce')
df['FOOD_COSTS_PER_YEAR'] = pd.to_numeric(df['FOOD COSTS PER YEAR'].replace(r'[\$,%,]', '', regex=True), errors='coerce')
df['PURCHASE_PRICE'] = pd.to_numeric(df['PURCHASE PRICE'].replace(r'[\$,%,]', '', regex=True), errors='coerce')
df['INTELLIGENCE_PERCENT'] = pd.to_numeric(df['INTELLIGENCE %'].replace(r'[\$,%,]', '', regex=True), errors='coerce')

# Define potentially useful features for our analysis
valid_features = ['score', 'score for kids', 'LONGEVITY(YEARS)', 'LIFETIME_COST', 
                 'FOOD_COSTS_PER_YEAR', 'PURCHASE_PRICE', 'SUITABILITY FOR CHILDREN']

# Filter to only include features that exist and have no missing values
# This ensures our models won't encounter NaN values during training
cluster_features = []
for feature in valid_features:
    if feature in df.columns and df[feature].isna().sum() == 0:
        cluster_features.append(feature)

print(f"\nUsing these features for clustering: {cluster_features}")

#----------------------------------------------
# PART 3: CLUSTERING ANALYSIS - Finding groups of similar dog breeds
#----------------------------------------------

# Standardize the data to give equal weight to all features
# This is important for distance-based algorithms like K-means
print("Performing K-means clustering...")
cluster_data = df[cluster_features]
scaler = StandardScaler()  # Will transform features to mean=0, std=1
scaled_data = scaler.fit_transform(cluster_data)

# Apply K-means clustering to group similar dog breeds
k = 3  # Number of clusters - can be adjusted based on domain knowledge or silhouette score
kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducibility
df['cluster'] = kmeans.fit_predict(scaled_data)  # Add cluster assignments to dataframe

# Analyze the characteristics of each cluster
print("\nMethod 1: K-means clustering of dog breeds")
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Transform back to original scale
for i in range(k):
    # Display the center (average characteristics) of each cluster
    print(f"Cluster {i} center: {dict(zip(cluster_features, cluster_centers[i]))}")
    # Show representative dog breeds in each cluster
    cluster_dogs = df[df['cluster'] == i]['Breed'].head(5).tolist()
    print(f"Top dogs in cluster {i}: {', '.join(cluster_dogs)}")

#----------------------------------------------
# PART 4: CLASSIFICATION MODELS - Predicting high-quality dog breeds
#----------------------------------------------

# Create a binary target variable: high quality (1) vs. average/low quality (0)
# Using the median score as the threshold
print("Creating classification models...")
df['high_quality'] = (df['score'] > df['score'].median()).astype(int)

# Prepare features (X) and target (y) for classification
X = df[cluster_features]  # Features from our validated list
y = df['high_quality']    # Target variable

# Split data into training and testing sets to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter tuning for Random Forest using Grid Search
# This helps find the optimal model configuration
print("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100],          # Number of trees in the forest
    'max_depth': [None, 10, 20],        # Maximum depth of trees
    'min_samples_split': [2, 5]         # Minimum samples required to split a node
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_  # Store the optimized model

# Train Random Forest model (using default parameters for comparison)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Train Logistic Regression model as a simpler alternative
lr = LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter to ensure convergence
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train SVM model as a third approach
# SVM can handle non-linear relationships between features
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Apply cross-validation to get more robust performance estimates
# This helps detect overfitting by testing on multiple data splits
print("Performing cross-validation...")
cv_rf = cross_val_score(rf, X, y, cv=5)  # 5-fold cross-validation
cv_lr = cross_val_score(lr, X, y, cv=5)

# Display cross-validation results
print(f"Random Forest CV Scores: {cv_rf}")
print(f"Random Forest CV Mean: {cv_rf.mean():.4f}, Std: {cv_rf.std():.4f}")
print(f"Logistic Regression CV Scores: {cv_lr}")
print(f"Logistic Regression CV Mean: {cv_lr.mean():.4f}, Std: {cv_lr.std():.4f}")

# Evaluate model performance on the test set
print("\nMethod 2: Classification models to predict high-quality dogs")
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Analyze feature importance to understand what drives dog quality
print("\nFeature importance for predicting high-quality dogs:")
importances = rf.feature_importances_
feature_df = pd.DataFrame({'Feature': cluster_features, 'Importance': importances})
feature_df = feature_df.sort_values('Importance', ascending=False)
print(feature_df)

# Perform feature selection to identify the most predictive attributes
# This can simplify the model while maintaining performance
print("Selecting the most important features...")
selector = SelectFromModel(rf, threshold="mean")
selector.fit(X, y)
selected_features = [feature for feature, selected in zip(cluster_features, selector.get_support()) if selected]
print("Selected features:", selected_features)

#----------------------------------------------
# PART 5: RECOMMENDATION SYSTEM - Matching owners to suitable breeds
#----------------------------------------------

# Define a function to recommend dog breeds based on user criteria
def recommend_breeds(df, criteria, top_n=5):
    """
    Recommend dog breeds based on user preferences.
    
    Parameters:
    - df: DataFrame containing dog breeds data
    - criteria: Dictionary with columns as keys and either:
                - A minimum value, or
                - A tuple of (min_value, max_value) for ranges
    - top_n: Number of top recommendations to return
    
    Returns:
    - DataFrame with the top N recommended breeds
    """
    # Start with all rows
    result = df.copy()
    
    # Apply each criterion as a filter
    for col, condition in criteria.items():
        if isinstance(condition, tuple):
            # Range condition (min, max)
            min_val, max_val = condition
            result = result[(result[col] >= min_val) & (result[col] <= max_val)]
        else:
            # Minimum value condition
            result = result[result[col] >= condition]
    
    # If no dogs match all criteria, return top breeds by score
    if len(result) == 0:
        print("No breeds match all criteria. Showing top dogs by overall score.")
        return df.sort_values('score', ascending=False).head(top_n)
    
    # Otherwise, return top matching breeds
    return result.sort_values('score', ascending=False).head(top_n)

# Define different user personas for recommendation examples
print("Generating breed recommendations for different user personas...")

# Family-oriented criteria - focus on child-friendliness
family_criteria = {
    'score for kids': 4.0,              # High score for kid-friendliness
    'SUITABILITY FOR CHILDREN': 1       # Suitable for children (assuming 1=suitable)
}

# Budget-conscious criteria - focus on cost factors
budget_criteria = {
    'LIFETIME_COST': (0, 15000),        # Lower lifetime cost range
    'PURCHASE_PRICE': (0, 1000)         # Lower initial purchase price
}

# Generate and display recommendations for each persona
print("\nTop dog breeds for families:")
family_dogs = recommend_breeds(df, family_criteria)
print(family_dogs[['Breed', 'score', 'score for kids', 'SUITABILITY FOR CHILDREN']])

print("\nTop dog breeds for budget-conscious owners:")
budget_dogs = recommend_breeds(df, budget_criteria)
print(budget_dogs[['Breed', 'score', 'LIFETIME_COST', 'PURCHASE_PRICE']])

#----------------------------------------------
# PART 6: VISUALIZATION - Creating figures for the report
#----------------------------------------------

# Create a multi-panel figure for displaying key insights
print("Generating visualizations for the report...")
plt.figure(figsize=(15, 10))

# Plot 1: Scatter plot showing clusters of dog breeds
plt.subplot(2, 2, 1)
plt.scatter(df['score'], df['LONGEVITY(YEARS)'], c=df['cluster'], cmap='viridis')
plt.xlabel('Overall Score')
plt.ylabel('Longevity (Years)')
plt.title('Dog Breeds Clusters')
plt.colorbar(label='Cluster')

# Plot 2: Bar chart of top-rated dog breeds
plt.subplot(2, 2, 2)
top_breeds = df.sort_values('score', ascending=False).head(10)
sns.barplot(x='score', y='Breed', data=top_breeds)
plt.title('Top 10 Dog Breeds by Overall Score')

# Plot 3: Feature importance visualization
plt.subplot(2, 2, 3)
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance for High-Quality Dogs')

# Plot 4: Scatter plot showing relationship between price and longevity
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='PURCHASE_PRICE', y='LONGEVITY(YEARS)', hue='high_quality')
plt.title('Purchase Price vs. Longevity')
plt.legend(title='High Quality')

# Save and display the visualization panel
plt.tight_layout()
plt.savefig('dog_analysis_visualizations.png')
plt.show()

# Create a confusion matrix to visualize classification performance
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Error handling template for production code
try:
    # Potentially error-prone code would go here
    # This is a placeholder for actual error-prone operations
    pass
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
    
print("Analysis complete! View the generated images for visualization results.")