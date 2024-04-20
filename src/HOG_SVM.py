import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("data loaded")



#grayscaling the images
# Converting train set images to grayscale
x_train_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train]

# Converting test set images to grayscale
x_test_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test]
print("Train and Test images grayscaled")


#Applying HOG

# Defining HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Computing HOG features for train set
x_train_hog = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, visualize=False) for img in x_train_gray]

# Computing HOG features for test set
x_test_hog = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, visualize=False) for img in x_test_gray]
print("HOG features computed")



#Applying PCA to reduce the dimensionality of the HOG features


# Fit PCA on training set without specifying n_components
pca = PCA()
pca.fit(x_train_hog)

# Calculating the cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1

print(f"Number of components for 95% explained variance: {n_components_95}")




# Definining PCA
pca = PCA(n_components=n_components_95)

# Fit on training set and transform train, validation, and test sets
x_train_pca = pca.fit_transform(x_train_hog)
# x_val_pca = pca.transform(x_val_hog)
x_test_pca = pca.transform(x_test_hog)
print("data reduced")


#Normalization
# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training set and transform train, validation, and test sets
x_train_scaled = scaler.fit_transform(x_train_pca)
# x_val_scaled = scaler.transform(x_val_hog)
x_test_scaled = scaler.transform(x_test_pca)
print("data normalized")


# Initialize SVM classifiers with different kernels
svm_linear = SVC(kernel='linear')
svm_poly = SVC(kernel='poly')
svm_rbf = SVC(kernel='rbf', C=10)

# List of SVM classifiers
svms = [svm_linear, svm_poly, svm_rbf]

# List of kernel names
kernels = ['Linear', 'Polynomial', 'RBF']

# Fit on training set and predict on test set for each SVM classifier
for svm, kernel in zip(svms, kernels):
    # Fit on training set
    svm.fit(x_train_scaled, y_train.ravel())
    
    # Predict on test set
    y_pred = svm.predict(x_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy with {kernel} kernel: {accuracy}")


# Saving the models
from joblib import dump

for svm, kernel in zip(svms, kernels):
    dump(svm, f'{kernel}_svm_model.pkl')







