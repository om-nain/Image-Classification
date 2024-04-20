import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



# Flattening the images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Standardizing the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Applying PCA
n_components = 119
pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# Applying LDA on PCA-transformed data
lda = LinearDiscriminantAnalysis()
x_train_lda = lda.fit_transform(x_train_pca, y_train)
x_test_lda = lda.transform(x_test_pca)

# Fitting models and comparing their test accuracies
models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

results = {}
for name, model in models.items():
    model.fit(x_train_lda, y_train)
    y_pred = model.predict(x_test_lda)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Printing the results
for name, accuracy in results.items():
    print(f"{name}: Test Accuracy = {accuracy:.4f}")
