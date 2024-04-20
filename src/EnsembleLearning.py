from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from preprocessing import process
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    # Load the data using the process module
    x_train, y_train, x_test, y_test = process.load_data()

    # Initialize the base models
    base_models = [
        ('logreg', LogisticRegression(max_iter=1000, solver='lbfgs')),
        ('mlp', MLPClassifier(hidden_layer_sizes=(1024, 512, 256), activation='relu', max_iter=500, solver='adam', alpha=0.0001)),
        ('adaboost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=42)),
        ('random_forest', RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, random_state=42))
    ]

    # Initialize the VotingClassifier
    voting_classifier = VotingClassifier(estimators=base_models, voting='hard')

    # Fit the VotingClassifier to the training data
    voting_classifier.fit(x_train, y_train)

    # Predict on the test data using the VotingClassifier
    y_pred_voting = voting_classifier.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_voting)
    print("Ensemble Accuracy:", accuracy)
