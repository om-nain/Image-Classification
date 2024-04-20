from joblib import load
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("data loaded")


# Converting train/test set images to grayscale
x_train_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train]
x_test_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test]
print("data grayscaled")


# Initializing HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)


# Computing HOG features for test set
x_test_hog = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, visualize=False) for img in x_test_gray]
print("HOG features computed")




# Initializing PCA
pca = PCA(n_components=119)

# Fit PCA on training data and transform both training and test data
x_train_hog = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, visualize=False) for img in x_train_gray]
x_train_pca = pca.fit_transform(x_train_hog)
x_test_pca = pca.transform(x_test_hog)

print("data reduced")

#Normalization
# Initialize StandardScaler
scaler = StandardScaler()

# Fit StandardScaler on training data and transform both training and test data
x_train_scaled = scaler.fit_transform(x_train_pca)
x_test_scaled = scaler.transform(x_test_pca)

print("data normalized")



# Load SVM classifiers
svm_linear_loaded = load('Linear_svm_model.pkl')
svm_poly_loaded = load('Polynomial_svm_model.pkl')
svm_rbf_loaded = load('RBF_svm_model.pkl')

# List of loaded SVM classifiers
loaded_svms = [svm_linear_loaded, svm_poly_loaded, svm_rbf_loaded]

# List of kernel names
kernels = ['Linear', 'Polynomial', 'RBF']

# Predict using loaded models
for svm, kernel in zip(loaded_svms, kernels):
    # Predict on test set
    y_pred = svm.predict(x_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy with {kernel} kernel (loaded model): {accuracy}")
