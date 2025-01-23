# 1. Importing library

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pylab as plt


"""## 2. Data exploration and data cleaning"""

# Read data
df = pd.read_csv('D:/BA/Portfolio/Portfolio/mobile-price-analysis/Mobile_Price_Data.csv')
print(df.head())
print(len(df))

# Count and remove empty value
print(df.isna().sum())
df = df.dropna()
print(len(df))
print(df.info())

"""# 2. Exploring relationships between "price_range" and other features"""

correlation_matrix = df.corr()
correlation_with_price = correlation_matrix['price_range'].sort_values(ascending=False)
# Display the correlations
print("Correlations with price_range:")
print(correlation_with_price)

# Plot a heatmap of correlations with price_range
plt.figure(figsize=(10, 6))
sns.heatmap(
    df.corr()[['price_range']].sort_values(by='price_range', ascending=False),
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1
)
plt.title("Correlation Heatmap with Price Range")
plt.show()

"""
The variables that are helpful for predicting the price range are:
- Battery power: 0.20265
- RAM: 0.91

Because these 2 variables have the highest correlation coefficients
"""

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,   # Display the correlation values on the heatmap
    fmt=".2f",    # Format for the numbers
    cmap='coolwarm',  # Color scheme
    vmin=-1, vmax=1  # Range of correlation values
)
plt.title("Feature Correlation Heatmap")
plt.show()

"""#3. Split and train model"""

# Train : test size = 8 : 2
train , test = train_test_split(df, test_size=0.2, random_state=42)
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

# Train model:
# Model: Applying training/testing data on the most 2 correlated features:
scaler = StandardScaler()
X_train = train[['battery_power', 'ram']]
y_train = train['price_range']
X_test = test[['battery_power', 'ram']]
y_test = test['price_range']

# Fit scaler on training data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver='lbfgs')  # Increased max_iter to address ConvergenceWarning
model.fit(X_train_scaled, y_train)

# Calculate accuracy for Logistic Regression
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print("Logistic Regression Training Accuracy:", train_accuracy)
print("Logistic Regression Testing Accuracy:", test_accuracy)

"""## Comment on result

The model assumes two factors: Battery power and RAM will be used to predict price range. From the result, training accuracy is 0.82 which is considered high. The training accuracy shows the proportion of correct predictions made by the model on the training dataset. 0.82 suggests that the model is effective and can predict the price range at 82% of the instances in the training set.Testing accuracy is 0.84 reflects that it can accurately predict the price rate 84% of the instances in the test set.

A slightly higher in testing accuracy comparing to training accuracy suggests that there can be unusual scenario. This can be caused due to the low correlated coefficient of "battery power".

# 4. Train KNN model
"""

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict and calculate accuracy for KNN
train_preds = knn.predict(X_train_scaled)
train_accuracy_knn = accuracy_score(y_train, train_preds)

test_preds = knn.predict(X_test_scaled)
test_accuracy_knn = accuracy_score(y_test, test_preds)

print("KNN Training Accuracy:", train_accuracy_knn)
print("KNN Testing Accuracy:", test_accuracy_knn)

"""# 5. Tune the hyper parameter K in KNN"""

param_grid = {'n_neighbors': np.arange(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], grid_search.cv_results_['mean_test_score'], marker='o')
plt.title('Grid Search Results')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Mean Cross-Validated Accuracy')
plt.grid(True)
plt.show()

print("Best K:", grid_search.best_params_['n_neighbors'])
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

best_knn = grid_search.best_estimator_
best_test_preds = best_knn.predict(X_test_scaled)
best_test_accuracy = accuracy_score(y_test, best_test_preds)
print("Accuracy on test set (best model):", best_test_accuracy)

"""
## Explain how K influences the prediction performance.

Low K:
- Pros: More flexible decision boundary, sensitive to local patterns.
- Cons: Higher variance, susceptible to noise and outliers, potential overfitting.

High K:
- Pros: Smoother decision boundary, less sensitive to individual data points.
- Cons: Lower variance, potential underfitting, may miss local patterns.
"""