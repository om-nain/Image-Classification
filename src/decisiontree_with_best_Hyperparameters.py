# Importing necessary libraries
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Fetching the CIFAR-10 dataset
cifar_10 = fetch_openml('CIFAR_10', version=1, as_frame=False)

# Preprocessing the data
X_data = cifar_10.data / 255.0  # Scaling the pixel values to the range [0, 1]
y_target = cifar_10.target.astype(int)  # Converting target labels to integers

# Defining the hyperparameter grid for GridSearchCV
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Splitting the data into training and testing sets
X_traini, X_testi, y_traini, y_testi = train_test_split(X_data, y_target, test_size=0.2, random_state=42)

# Creating a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Performing Grid Search Cross Validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_traini, y_traini)

# Printing the best hyperparameters found by GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)

# Getting the best estimator from the grid search
best_dt_classifier = grid_search.best_estimator_

# Making predictions on the test set
y_predicted = best_dt_classifier.predict(X_test)

# Calculating and printing the accuracy score
accuracy_result = accuracy_score(y_testi, y_predicted)
print("from this i got Accuracy:", accuracy_result)

# Generating and printing the classification report
report_result = classification_report(y_testi, y_predicted)
print("Classification Report i got :")
print(report_result)
