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

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.tokenizer = tokenizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # define the metrics
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.top_k_acc_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(10, name="top_10_acc")

       
