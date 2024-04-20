from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from preprocessing import process

if __name__ == "__main__":
    # Load the data using the process module
    x_train, y_train, x_test, y_test = process.load_data()
    
    # Initialize and configure the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 512, 256), activation='relu', max_iter=500, solver='adam', alpha=0.0001)

    # Fit the model on the training data
    mlp.fit(x_train, y_train)

    # Predict on the test data
    y_pred = mlp.predict(x_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))
