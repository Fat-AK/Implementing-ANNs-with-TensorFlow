import tensorflow as tf
import numpy as np
import tensorflow_text as text
import sentencepiece as sp
import io
import tqdm

# load the dataset
with open('bible.txt', 'r') as f:
    bible = f.read()

# train the tokenizer and save it to a file
sp.SentencePieceTrainer.train(
    input='bible.txt', model_prefix='tokenizer_model', model_type="unigram", vocab_size=4096)

# load the trained tokenizer from file
trained_tokenizer_model = io.open('tokenizer_model.model', mode='rb').read()
tokenizer = text.SentencepieceTokenizer(model=trained_tokenizer_model, out_type=tf.int32)

# encode the text using the tokenizer
tokenized_text = tokenizer.tokenize(bible)

# split the text into windows
window_size = 64
data = text.sliding_window(tokenized_text, window_size + 1, axis=0)
inputs = data[:, :window_size]
targets = data[:, 1:window_size + 1]

# create a dataset
batch_size = 64
buffer_size = 10000
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# define the model architecture
embedding_size = 128

class Embedder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(4096, embedding_size)
        self.position_encoding = tf.keras.layers.Embedding(window_size, embedding_size)

    def call(self, input):
        positions = tf.range(0, window_size)
        embedded_seq = self.token_embedding(input)
        embedded_pos = self.position_encoding(positions)
        result = embedded_seq + embedded_pos
        return result

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=embedding_size)
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embedding_size)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.batchnorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.batchnorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

    def call(self, input, training):
        x = self.mha(input, input, use_causal_mask=True)
        x = self.dropout1(x, training=training)
        x = input + x
        ln_out = self.batchnorm1(x)
        x = self.dense1(ln_out)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = x + ln_out
        result = self.batchnorm2(x)
        return result

# define the model
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, num_transformer_blocks):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.transformer_blocks = [TransformerBlock() for _ in range(num_transformer_blocks)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, input, training):
        embedded = self.token_embedding(input)
        for block in self.transformer_blocks:
            embedded = block(embedded, training=training)
        logits = self.dense(embedded)
        return logits

# define hyperparameters
vocab_size = 4096
embedding_size = 128
num_transformer_blocks = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# create the model
model = Model(vocab_size, embedding_size, num_transformer_blocks)

# define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# define the metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

# define the file writers for TensorBoard
train_log_dir = 'logs/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_log_dir = 'logs/val'
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

# define the training loop
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = loss_fn(targets, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(targets, logits)

@tf.function
def val_step(inputs, targets):
    logits = model(inputs, training=False)
    loss = loss_fn(targets, logits)
    val_loss(loss)
    val_acc(targets, logits)

# define the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((inputs_val, targets_val)).batch(batch_size, drop_remainder=True)

# define the training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    for step, (inputs, targets) in enumerate(train_dataset):
        train_step(inputs, targets)
        if step % 100 == 0:
            print(f'Step {step}, Loss: {train_loss.result()}, Accuracy: {train_acc.result()}')
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
    for inputs, targets in val_dataset:
        val_step(inputs, targets)
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', val_acc.result(), step=epoch)
    print(f'Validation Loss: {val_loss.result()}, Validation Accuracy: {val_acc.result()}')

# save the model
model.save('model.h5')
