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

# Sample a batch of data from the training set
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Import the necessary layers and optimizers
from tensorflow.keras import layers, optimizers
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

  
# Import the necessary layers and optimizers
from tensorflow.keras import layers, optimizers

# Use the ResNet50 architecture with weight regularization, batch normalization, and dropout
model = tf.keras.Sequential([
    layers.Reshape((32, 32, 3), input_shape=(32, 32, 3)),
    layers.Conv2D(32, 3, padding='same', 
kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, 3, padding='same',                                                                                              
kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding='same', 
kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, 3, padding='same', 
kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Conv2D(128, 3, padding='same', 
kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, 3, padding='same', 
kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# defining the training loop function
def train_loop(model, dataset, epochs, optimizer):
  for epoch in range(epochs):
    # Iterate over the dataset
    for x, y in dataset:
      # Open a GradientTape
      with tf.GradientTape() as tape:
        # Forward pass
        logits = model(x)
        # Compute the loss
        loss_value = loss(y, logits)
        # Compute the gradients
        gradients = tape.gradient(loss_value, model.trainable_variables)
        # Apply the gradients to the model
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
 #To define the hyperparameters, you can specify the learning rate, batch size, and number of epochs to train for. For example:
learning_rate = 0.001
batch_size = 32
epochs = 15

#define loss function
loss = tf.keras.losses.CategoricalCrossentropy()

#define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

#store loss and accuracy
# Create metric objects for tracking the loss and accuracy
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.Mean()

# In the training loop
for x, y in train_dataset:
  # Forward pass and loss computation
  logits = model(x)
  loss_value = loss(y, logits)
  
  # Update the metrics
  train_loss.update_state(loss_value)
  train_accuracy.update_state(y, logits)
  
# After the training loop
print(f'Training loss: {train_loss.result():.4f}')
print(f'Training accuracy: {train_accuracy.result():.4f}')





