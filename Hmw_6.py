# Import the required libraries and the CIFAR-10 data set from TensorFlow.
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load the data set and split it into training and test sets.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert the data to floating point numbers and normalize the input data by dividing by 255. 
# This will scale the data so that each pixel has a value between 0 and 1.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Use the TensorFlow function tf.one_hot to convert the target vectors to one-hot encoded vectors.
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Cache the preprocessed data in memory to speed up future access.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.cache()

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.cache()

# Visualize a sample from the training set
sample_idx = 0
sample_image = x_train[sample_idx]
sample_label = y_train[sample_idx]

plt.imshow(sample_image)
plt.title(sample_label)
plt.show()

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

#To train the network and define a training loop function,
batch_size = 128
learning_rate = 1e-3
num_epochs = 15

#Create an optimizer object using the Adam optimizer with the specified learning rate.
optimizer = optimizers.Adam(learning_rate)

#define loss function
loss_fn = tf.losses.CategoricalCrossentropy()

#define metrick to track the accuracy of the modle on the test set
train_acc_metric = tf.metrics.CategoricalAccuracy()
test_acc_metric = tf.metrics.CategoricalAccuracy()

#define the function to that we will use to test the model:
def train_loop(model, optimizer, loss_fn, train_dataset, test_dataset):
  for epoch in range(num_epochs):
    # Reset the metrics at the start of each epoch
    train_acc_metric.reset_states()
    test_acc_metric.reset_states()

    # Train the model on the training set
    for x_batch, y_batch in train_dataset:
      with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss_value = loss_fn(y_batch, logits)
      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
      train_acc_metric.update_state(y_batch, logits)

    # Evaluate the model on the test set
    for x_batch, y_batch in test_dataset:
      logits = model(x_batch, training=False)
      test_acc_metric.update_state(y_batch, logits)

    # Print the loss and accuracy on the training and test sets
    print('Epoch: {}, Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
        epoch, loss_value, train_acc_metric.result(), test_acc_metric.result()))

#call the function to train the model 
train_loop(model, optimizer, loss_fn, train_dataset, test_dataset)
    
    
    
import matplotlib.pyplot as plt

#Define a function to visualize the performance of the model

def visualize_performance(history):

#Get the accuracy values from the history object

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

#Get the loss values from the history object

loss = history.history['loss']
val_loss = history.history['val_loss']

#Get the number of epochs

epochs = range(1, len(acc) + 1)

#Plot the training and validation accuracy

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#Plot the training and validation loss

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Train the model and get the training history

history = train_loop(model, optimizer, loss_fn, train_dataset, test_dataset)

#Visualize the model's performance

visualize_performance(history)


