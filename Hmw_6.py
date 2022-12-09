#Import the required libraries and the CIFAR-10 data set from TensorFlow.
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

#Load the data set and split it into training and test sets.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Convert the data to floating point numbers and normalize the input data by dividing by 255. This will scale the data so that each pixel has a value between 0 and 1.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#Use the TensorFlow function tf.one_hot to convert the target vectors to one-hot encoded vectors.
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

#Cache the preprocessed data in memory to speed up future access.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.cache()

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.cache()

#Visualize and sample the data to check if it has been preprocessed correctly.
import matplotlib.pyplot as plt

# Visualize a sample from the training set
sample_idx = 0
sample_image = x_train[sample_idx]
sample_label = y_train[sample_idx]

plt.imshow(sample_image)
plt.title(sample_label)
plt.show()

# Sample a batch of data from the training set
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
for images, labels in train_dataset.take(1):
  print(images.shape, labels.shape)
