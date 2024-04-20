from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocessing import process

if __name__ == "__main__":
    # Load the data using the process module
    x_train, y_train, x_test, y_test = process.load_data()
    
    # Decision Tree as base estimator for AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # AdaBoost Classifier with specified parameters
    adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=42)

    # Fit AdaBoost model to the training data
    adaboost.fit(x_train, y_train)

    # Predict on the test data using AdaBoost
    y_pred_boost = adaboost.predict(x_test)

    # Print the classification report for AdaBoost
    print("\nAdaBoost Classification Report:")
    print(classification_report(y_test, y_pred_boost))
