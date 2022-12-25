import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add a channel dimension to the input data
x_train = x_train[:, :, :, tf.newaxis]
x_test = x_test[:, :, :, tf.newaxis]

# Create sequence data by alternating the signs of the targets and computing the cumulative sum
def create_sequence_data(x, y):
    y = y.astype(int)
    y = y * (-1 if i % 2 else 1)
    y = y.cumsum()
    return x, y

# Modify the loop variable i to avoid an undefined variable error
x_train_seq = []
y_train_seq = []
for i in range(len(x_train)):
    x_train[i], y_train[i] = create_sequence_data(x_train[i], y_train[i])

for i in range(len(x_test)):
    x_test[i], y_test[i] = create_sequence_data(x_test[i], y_test[i])

# Create the train and test datasets
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Define the CNN and LSTM model


model = tf.keras.Sequential([
    # CNN part
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    # LSTM part
    tf.keras.layers.Reshape((batch_size, -1)),
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=5)

# Evaluate the model
model.evaluate(test_dataset)
