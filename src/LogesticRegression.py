from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import process

if __name__ == "__main__":
    # Load the data using the process module
    x_train, y_train, x_test, y_test = process.load_data()
    
    # Initialize Logistic Regression model with specified parameters
    logreg = LogisticRegression(max_iter=1000, solver='lbfgs')

    # Fit the model on the training data
    logreg.fit(x_train, y_train)

    # Predict on the test data
    y_pred_logreg = logreg.predict(x_test)

    # Print the classification report
    print(classification_report(y_test, y_pred_logreg))
