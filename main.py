
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel

print("Loading dataset...")
df = pd.read_csv("dogs-ranking-dataset.csv")

print("Dataset shape:", df.shape)
print("Summary statistics:")
print(df.describe())

print("Checking for missing values...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values in Dataset")
plt.savefig('missing_values.png')

print("Unique intelligence values:")
print(df['intelligence'].unique())
print("\nUnique grooming frequency values:")
print(df['GROOMING FREQUNCY'].unique())

intelligence_map = {} 
grooming_map = {} 

print("Preprocessing numeric columns...")
df['LIFETIME_COST'] = pd.to_numeric(df['$LIFETIME COST'].replace(r'[\$,%,]', '', regex=True), errors='coerce')
df['FOOD_COSTS_PER_YEAR'] = pd.to_numeric(df['FOOD COSTS PER YEAR'].replace(r'[\$,%,]', '', regex=True), errors='coerce')
df['PURCHASE_PRICE'] = pd.to_numeric(df['PURCHASE PRICE'].replace(r'[\$,%,]', '', regex=True), errors='coerce')
df['INTELLIGENCE_PERCENT'] = pd.to_numeric(df['INTELLIGENCE %'].replace(r'[\$,%,]', '', regex=True), errors='coerce')

valid_features = ['score', 'score for kids', 'LONGEVITY(YEARS)', 'LIFETIME_COST', 
                 'FOOD_COSTS_PER_YEAR', 'PURCHASE_PRICE', 'SUITABILITY FOR CHILDREN']

cluster_features = []
for feature in valid_features:
    if feature in df.columns and df[feature].isna().sum() == 0:
        cluster_features.append(feature)

print(f"\nUsing these features for clustering: {cluster_features}")

print("Performing K-means clustering...")
cluster_data = df[cluster_features]
scaler = StandardScaler() 
scaled_data = scaler.fit_transform(cluster_data)

k = 3  
kmeans = KMeans(n_clusters=k, random_state=42)  
df['cluster'] = kmeans.fit_predict(scaled_data)  


print("\nMethod 1: K-means clustering of dog breeds")
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  
for i in range(k):
    
    print(f"Cluster {i} center: {dict(zip(cluster_features, cluster_centers[i]))}")
    
    cluster_dogs = df[df['cluster'] == i]['Breed'].head(5).tolist()
    print(f"Top dogs in cluster {i}: {', '.join(cluster_dogs)}")


print("Creating classification models...")
df['high_quality'] = (df['score'] > df['score'].median()).astype(int)


X = df[cluster_features]  
y = df['high_quality']    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5] 
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_ 


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


lr = LogisticRegression(random_state=42, max_iter=1000)  
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


print("Performing cross-validation...")
cv_rf = cross_val_score(rf, X, y, cv=5)
cv_lr = cross_val_score(lr, X, y, cv=5)


print(f"Random Forest CV Scores: {cv_rf}")
print(f"Random Forest CV Mean: {cv_rf.mean():.4f}, Std: {cv_rf.std():.4f}")
print(f"Logistic Regression CV Scores: {cv_lr}")
print(f"Logistic Regression CV Mean: {cv_lr.mean():.4f}, Std: {cv_lr.std():.4f}")


print("\nMethod 2: Classification models to predict high-quality dogs")
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))


print("\nFeature importance for predicting high-quality dogs:")
importances = rf.feature_importances_
feature_df = pd.DataFrame({'Feature': cluster_features, 'Importance': importances})
feature_df = feature_df.sort_values('Importance', ascending=False)
print(feature_df)


print("Selecting the most important features...")
selector = SelectFromModel(rf, threshold="mean")
selector.fit(X, y)
selected_features = [feature for feature, selected in zip(cluster_features, selector.get_support()) if selected]
print("Selected features:", selected_features)

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
 
    result = df.copy()
    
 
    for col, condition in criteria.items():
        if isinstance(condition, tuple):
           
            min_val, max_val = condition
            result = result[(result[col] >= min_val) & (result[col] <= max_val)]
        else:
            
            result = result[result[col] >= condition]
    

    if len(result) == 0:
        print("No breeds match all criteria. Showing top dogs by overall score.")
        return df.sort_values('score', ascending=False).head(top_n)
    
    return result.sort_values('score', ascending=False).head(top_n)


print("Generating breed recommendations for different user personas...")

family_criteria = {
    'score for kids': 4.0,
    'SUITABILITY FOR CHILDREN': 1
}

budget_criteria = {
    'LIFETIME_COST': (0, 15000),
    'PURCHASE_PRICE': (0, 1000)
}


print("\nTop dog breeds for families:")
family_dogs = recommend_breeds(df, family_criteria)
print(family_dogs[['Breed', 'score', 'score for kids', 'SUITABILITY FOR CHILDREN']])

print("\nTop dog breeds for budget-conscious owners:")
budget_dogs = recommend_breeds(df, budget_criteria)
print(budget_dogs[['Breed', 'score', 'LIFETIME_COST', 'PURCHASE_PRICE']])

print("Generating visualizations for the report...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(df['score'], df['LONGEVITY(YEARS)'], c=df['cluster'], cmap='viridis')
plt.xlabel('Overall Score')
plt.ylabel('Longevity (Years)')
plt.title('Dog Breeds Clusters')
plt.colorbar(label='Cluster')

plt.subplot(2, 2, 2)
top_breeds = df.sort_values('score', ascending=False).head(10)
sns.barplot(x='score', y='Breed', data=top_breeds)
plt.title('Top 10 Dog Breeds by Overall Score')

plt.subplot(2, 2, 3)
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance for High-Quality Dogs')

plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='PURCHASE_PRICE', y='LONGEVITY(YEARS)', hue='high_quality')
plt.title('Purchase Price vs. Longevity')
plt.legend(title='High Quality')

plt.tight_layout()
plt.savefig('dog_analysis_visualizations.png')
plt.show()

print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix')
plt.savefig('confusion_matrix.png')

try:
    pass
except Exception as e:
    print(f"Error: {e}")
    
print("Analysis complete! View the generated images for visualization results.")