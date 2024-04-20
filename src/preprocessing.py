import tensorflow as tf
import cv2
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to load CIFAR-10 dataset
def load_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("data loaded")
    return x_train, y_train, x_test, y_test

# Function to convert images to grayscale
def grayscale(x_train, x_test):
    # Convert train set images to grayscale
    x_train_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train]

    # Convert test set images to grayscale
    x_test_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test]
    print("data grayscaled")
    return x_train_gray, x_test_gray

# Function to compute HOG features
def hog_features(x_train_gray, x_test_gray):
    # Define HOG parameters
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Compute HOG features for train set
    x_train_hog = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=False) for img in x_train_gray]

    # Compute HOG features for test set
    x_test_hog = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=False) for img in x_test_gray]
    print("HOG features computed")
    return x_train_hog, x_test_hog

# Function to apply PCA to reduce dimensionality
def pca(x_train_hog, x_test_hog):
    # Define PCA
    pca = PCA(n_components=64)

    # Fit on training set and transform train, validation, and test sets
    x_train_pca = pca.fit_transform(x_train_hog)
    # x_val_pca = pca.transform(x_val_hog)
    x_test_pca = pca.transform(x_test_hog)
    print("data reduced")
    return x_train_pca, x_test_pca

# Function to normalize data
def normalize(x_train_pca, x_test_pca):
    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit on training set and transform train, validation, and test sets
    x_train_scaled = scaler.fit_transform(x_train_pca)
    # x_val_scaled = scaler.transform(x_val_hog)
    x_test_scaled = scaler.transform(x_test_pca)
    print("data normalized")
    return x_train_scaled, x_test_scaled

# Main processing function
def process():
    # Load data
    x_train, y_train, x_test, y_test = load_data()
    # Convert images to grayscale
    x_train_gray, x_test_gray = grayscale(x_train, x_test)
    # Compute HOG features
    x_train_hog, x_test_hog = hog_features(x_train_gray, x_test_gray)
    # Perform PCA
    x_train_pca, x_test_pca = pca(x_train_hog, x_test_hog)
    # Normalize data
    x_train_scaled, x_test_scaled = normalize(x_train_pca, x_test_pca)
    return x_train_scaled, y_train, x_test_scaled, y_test

