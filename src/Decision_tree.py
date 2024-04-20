import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value if the node is a leaf (class label)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the decision tree

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))  # Number of unique classes
        self.num_features = X.shape[1]  # Number of features
        self.tree = self._grow_tree(X, y)  # Grow the decision tree

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]  # Count samples per class
        predicted_class = np.argmax(num_samples_per_class)  # Predicted class based on majority vote

        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return Node(value=predicted_class)  # If only one class or reached max depth, create leaf node

        feature_indices = np.random.choice(self.num_features, self.num_features, replace=False)  # Randomly select feature indices
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices, num_samples_per_class)  # Find best split
        left_indices, right_indices = self._split_data(X[:, best_feature], best_threshold)  # Split data

        left_subtree = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)  # Grow left subtree recursively
        right_subtree = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)  # Grow right subtree recursively

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y, feature_indices, num_samples_per_class):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_index in feature_indices:
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))  # Sort feature values and corresponding class labels
            num_left = [0] * self.num_classes  # Initialize counts for left child
            num_right = num_samples_per_class.copy()  # Initialize counts for right child

            for i in range(1, len(y)):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (len(y) - i)) ** 2 for x in range(self.num_classes))
                gini = (i * gini_left + (len(y) - i) * gini_right) / len(y)

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_feature, best_threshold

    def _split_data(self, feature_values, threshold):
        left_indices = np.where(feature_values <= threshold)[0]  # Indices of samples with feature values less than or equal to threshold
        right_indices = np.where(feature_values > threshold)[0]  # Indices of samples with feature values greater than threshold
        return left_indices, right_indices

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)  # Recursively traverse left subtree
        else:
            return self._predict_single(x, node.right)  # Recursively traverse right subtree

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components  # Number of principal components
        self.components = None  # Principal components
        self.mean = None  # Mean of the data

    def fit(self, X):
        self.mean = np.mean(X, axis=0)  # Compute mean of the data
        X_centered = X - self.mean  # Center the data
        covariance_matrix = np.cov(X_centered, rowvar=False)  # Compute covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # Perform eigen decomposition
        idx = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
        eigenvectors = eigenvectors[:, idx]  # Reorder eigenvectors
        eigenvalues = eigenvalues[idx]  # Reorder eigenvalues
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]  # Select top n_components eigenvectors
        else:
            self.components = eigenvectors

    def transform(self, X):
        X_centered = X - self.mean  # Center the data
        return np.dot(X_centered, self.components)  # Project data onto principal components

(x_training, y_training), (x_testing, y_testing) = tf.keras.datasets.cifar10.load_data()  # Load CIFAR-10 dataset
print("Data loaded successfully")

X_training = x_training.reshape(x_training.shape[0], -1)  # Reshape training data
X_testing = x_testing.reshape(x_testing.shape[0], -1)  # Reshape test data
y_training = y_training.ravel()  # Flatten training labels
y_testing = y_testing.ravel()  # Flatten test labels

X_training = X_training / 255.0  # Normalize training data
X_testing = X_testing / 255.0  # Normalize test data

pca = PCA(n_components=217)  # Initialize PCA with desired number of components
pca.fit(X_training)  # Fit PCA to training data
X_training_pca = pca.transform(X_training)  # Transform training data
X_testing_pca = pca.transform(X_testing)  # Transform test data

print("Number of components obtained from PCA:", pca.components.shape[1])  # Print number of components obtained

plt.figure(figsize=(8, 6))  # Create a new figure
plt.plot(np.cumsum(np.var(X_training_pca, axis=0) / np.sum(np.var(X_training_pca, axis=0))))  # Plot explained variance ratio
plt.xlabel('Number of Components')  # Set x-axis label
plt.ylabel('Explained Variance Ratio')  # Set y-axis label
plt.title('Explained Variance Ratio by Number of Components')  # Set title
plt.grid(True)  # Enable grid
plt.show()  # Show plot

clf_scratch = DecisionTreeClassifier(max_depth=5)  # Initialize decision tree classifier
clf_scratch.fit(X_training_pca, y_training)  # Train decision tree classifier

y_prediction_scratch = clf_scratch.predict(X_testing_pca)  # Predict labels for test data
accuracy_scratch = accuracy_score(y_testing, y_prediction_scratch)  # Compute accuracy
report_scratch = classification_report(y_testing, y_prediction_scratch)  # Generate classification report
print("from this i got Accuracy (Scratch):", accuracy_scratch)  # Print accuracy
print("Classification Report (Scratch):")  # Print classification report
print(report_scratch)
