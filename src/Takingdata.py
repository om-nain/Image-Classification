import tensorflow as tf
from sklearn.model_selection import train_test_split


# Load CIFAR-10 dataset
(x_training, y_training), (x_testing, y_testing) = tf.keras.datasets.cifar10.load_data()
print("data loaded successfully ")
import matplotlib.pyplot as plt

# Printing some of the  loaded data
print("x_train shape:", x_training.shape)
print("y_train shape:", y_training.shape)

# Displaying some images from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_training[i], cmap=plt.cm.binary)
    plt.xlabel(y_training[i][0])
plt.show()
