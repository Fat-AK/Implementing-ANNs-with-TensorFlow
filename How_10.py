import tensorflow as tf
from tensorflow import keras
import tensorflow_text as text

# Load the bible text
bible_text = open("bible.txt", "r").read()
bible_text = bible_text.lower()

# Tokenize the text into words
word_tokenizer = text.tokenizers.WordTokenizer()
bible_words = word_tokenizer.tokenize(bible_text)

# Get the top 10000 most common words
word_counts = {}
for word in bible_words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
top_words = [word[0] for word in sorted_word_counts[:10000]]

# Create the input-target pairs
window_size = 4
input_target_pairs = []
for i, word in enumerate(bible_words):
    if word in top_words:
        for j in range(i-window_size, i+window_size+1):
            if j != i and j >= 0 and j < len(bible_words) and bible_words[j] in top_words:
                input_target_pairs.append((word, bible_words[j]))

# Create a dataset from the input-target pairs
dataset = tf.data.Dataset.from_generator(lambda: input_target_pairs, (tf.string, tf.string))
dataset = dataset.batch(32).shuffle(len(input_target_pairs))

# Implement the SkipGram model
vocab_size = len(top_words)
embedding_size = 128

class SkipGram(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def build(self, batch_input_shape):
        self.embeddings = self.add_weight(
            shape=(self.vocab_size, self.embedding_size),
            initializer="random_normal",
            trainable=True,
        )
        self.biases = self.add_weight(
            shape=(self.vocab_size,),
            initializer="zeros",
            trainable=True,
        )
