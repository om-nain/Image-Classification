from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from preprocessing import process

if __name__ == "__main__":
    # Load the data using the process module
    x_train, y_train, x_test, y_test = process.load_data()
    
    # Random Forest Classifier
    random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, random_state=42)

    # Fit Random Forest model
    random_forest.fit(x_train, y_train)

    # Predict on the test data using Random Forest
    y_pred_rf = random_forest.predict(x_test)

    # Print the classification report for Random Forest
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
