import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from collections import Counter # Import Counter from collections

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images for feature extraction
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Implement KNN classifier
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(train_images, train_labels, test_image, k):
    distances = []
    for i in range(len(train_images)):
        dist = euclidean_distance(test_image, train_images[i])
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return Counter([tuple(label) for dist, label in neighbors]).most_common(1)[0][0]

# Select the value of k
k = 3

# Classify test images
predictions = []
for i in range(len(x_test_flat)):
    prediction = knn_predict(x_train_flat, y_train, x_test_flat[i], k)
    predictions.append(prediction)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Qualitative analysis: Visualize misclassified images
misclassified_indices = [i for i, (pred, actual) in enumerate(zip(predictions, y_test)) if pred != actual]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_test[misclassified_indices[i]])
    ax.set_title(f"Predicted: {predictions[misclassified_indices[i]]}, Actual: {y_test[misclassified_indices[i]]}")
    ax.axis('off')
plt.show()

# Failure case analysis: Cross-validation for hyperparameter tuning
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train_flat, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best k: {best_k}")