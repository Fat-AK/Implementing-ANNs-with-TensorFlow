import tensorflow as tf
import datetime
import tqdm
import io
import nltk
     


!pip install tensorflow_text
import tensorflow_text as tf_text

!pip install sentencepiece
import sentencepiece as sp


import requests
from bs4 import BeautifulSoup
     


VOCAB_SIZE = 4000
WINDOW_SIZE = 120
BATCH_SIZE = 250
EMBEDDING_DIM = 64

#Data set: preprocessing and tokenization
# Getting the HTML 
r = requests.get('https://archive.org/details/sq_20211001')

# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, 'html.parser')

# Getting the text out of the soup
text = soup.get_text()

song_of_achilles_text = text[71261:643308]

#cleaning
text = song_of_achilles_text.lower()
words = text.split()
words = [word.strip('.,!;()[]') for word in words]
words = [word.replace("'s", '') for word in words]

#finding unique
unique = []
for word in words:
    if word not in unique:
        unique.append(word)

print(len(unique))
     
achilles  = open('Song_of_Achilles.txt', 'w+')
achilles.write(achilles_text)
achilles.close

sp.SentencePieceTrainer.train(
    input= 'Song_of_Achilles.txt', 
    model_prefix='tokenizer_model', 
    model_type="unigram", 
    vocab_size=VOCAB_SIZE,
    pad_id=0,                
    unk_id=1,
    bos_id=2,
    eos_id=3)
     
trained_tokenizer_model = tf.io.gfile.GFile('tokenizer_model.model', "rb").read()

# load the model as a tokenizer 
tokenizer = tf_text.SentencepieceTokenizer(
    model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1, reverse=False,
    add_bos=False, add_eos=False, return_nbest=False, name=None

def preprocessing_pipeline(text):
    """Apply preproccesing pipeline to the given dataset.
    
    :param data: data to be preprocessed
    :type data: tensorflow 'Dataset'
    :return: preprocessed dataset
    :rtype: tensorflow 'Dataset'
    """
    # tokenize text
    tokens = tokenizer.tokenize(text)
    # create input sequence with target as last element of the sequence
    data = tf_text.sliding_window(data = tokens, width = WINDOW_SIZE+1, axis = 0)
    # create tf dataset and split it into seq and target pairs
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.map(lambda data: tf.split(data, [WINDOW_SIZE, 1], -1))
    # cache the dataset
    data = data.cache()
    # shuffle, batch and prefetch the dataset
    data = data.shuffle(1000)
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(100)
    return data
  
  train_data = preprocessing_pipeline(song_of_achilles_text)
  
  #Model 
  from tensorflow.keras.layers import Layer

class EmbeddingBlock(Layer):
    def __init__(self):
        super(EmbeddingBlock, self).__init__()

        self.token_embedding = tf.keras.layers.Embedding(input_dim = VOCAB_SIZE, 
                                                   output_dim = EMBEDDING_DIM)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim = WINDOW_SIZE,
                                                  output_dim = EMBEDDING_DIM)
        

    def call(self, seq):
        temp = tf.range(0, WINDOW_SIZE)

        x = self.token_embedding(seq)

        pos = self.pos_embedding(temp)

        x += pos

        return x
 class TransformerBlock(Layer):
    def __init__(self, num_heads):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads= self.num_heads,
                                                      key_dim = EMBEDDING_DIM)
        self.dense1 = tf.keras.layers.Dense(units = 256, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = EMBEDDING_DIM)

        self.dropOut1 = tf.keras.layers.Dropout(rate = 0.1)
        self.dropOut2 = tf.keras.layers.Dropout(rate = 0.1)
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon = 0.000001)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon = 0.000001)

    def call(self, input):
        
        tmp = self.mha(input, input)
        tmp = self.dropOut1(tmp)

        tmp += input

        ln_out = self.norm1(tmp)

        x = self.dense1(ln_out)
        x = self.dense2(x)
        x = self.dropOut2(x)

        x += ln_out

        x = self.norm2(x)       

        return x
  
  class MyModel(tf.keras.Model):

  def __init__(self, tokenizer, num_heads, optimizer, loss_func):
    super(MyModel, self).__init__()

    self.tokenizer = tokenizer
    self.num_heads = num_heads
    
    self.optimizer = optimizer
    self.loss_function = loss_func
    
    self.metrics_list = [
                    tf.keras.metrics.Mean(name="loss"),
                    tf.keras.metrics.CategoricalAccuracy(name="acc"),
                    tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                    ]
    
    self.layerList = [
                      EmbeddingBlock(),
                      TransformerBlock(self.num_heads),
                      tf.keras.layers.GlobalAveragePooling1D(),
                      tf.keras.layers.Dense(units=VOCAB_SIZE)
                      ]
    
  def call(self, inputs):
    """Compute the feed-forward pass through all dense layers.
    
    :param inputs: network input
    :type inputs: tf.Tensor
    """
    x = inputs
    for layer in self.layerList.layers:
        x = layer(x)
    return x
  
  def reset_metrics(self):
    for metric in self.metrics:
      metric.reset_states()

  @tf.function
  def train_step(self, data):
    
    x, targets = data
    
    with tf.GradientTape() as tape:
        predictions = self(x, training=True)
        
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
    
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    # update loss metric
    self.metrics[0].update_state(loss)
    
    # for all metrics except loss, update states (accuracy etc.)
    for metric in self.metrics[1:]:
        metric.update_state(targets,predictions)

    # Return a dictionary mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

  def generate_text(self, prompt, length):

    tokenseq = tokenizer.tokenize(prompt)

    while len(tokenseq) <= length:
      #add batch dimesnion
      seq = tf.expand_dims(tokenseq, axis = 0)
      #crate padding
      temp = WINDOW_SIZE- len(tokenseq)
      paddings = tf.constant([[0, 0, ], [temp, 0 ]])
      #padd input
      seq = tf.pad(seq, paddings, 'CONSTANT')
      # run seq through model
      logit = self.call(seq)

      # top_k
      logits, indices = tf.math.top_k(logit, k= 100, sorted=True)
      top_k = tf.random.categorical(logits, 1)
      top_k = tf.squeeze(top_k)
      top_k = top_k.numpy()
      indices = tf.squeeze(indices)

      token = indices[top_k]
      token = tf.expand_dims(token, 0)
      tokenseq = tf.concat([tokenseq, token], -1)
    out = tokenizer.detokenize(tokenseq)

    return out
  
  # Initialize the loss-function
cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Initialize the optimizer
optimizer = tf.keras.optimizers.Adam(0.001)
# Initialize Model
model = MyModel(tokenizer, 3, optimizer, cross_entropy_loss)
     
#training loop
!rm -rf ./logs/
# load tensorboard extension
%load_ext tensorboard

# Define where to save the log
hyperparameter_string= "Test1"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{hyperparameter_string}/{current_time}/train"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)
import tqdm.notebook as notebook
temp = 5
for epoch in range(100):
    
    print(f"Epoch {epoch}:")
    
    # Training:
    
    for data in notebook.tqdm(train_data,position=0, leave=True):
        metrics = model.train_step(data)
    
    # print the metrics
    print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
    
    # logging the validation metrics to the log file which is used by tensorboard
    with train_summary_writer.as_default():
        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch+300)
    
    # reset all metrics (requires a reset_metrics method in the model)
    model.reset_metrics()

    # generate a tes´xt
    if temp == 5:
      text = model.generate_text("I am ", 50)
      with train_summary_writer.as_default():
          tf.summary.text("generated_text", text, step = epoch+300)
      print(text)
      temp = 1
    else:
      temp+= 1

    print("\n")
     
  
  text = model.generate_text("Achilles ", 100)
with train_summary_writer.as_default():
    tf.summary.text("generated_text_final", text, step = 2)
     

%tensorboard --logdir logs/
model.save_weights(f"saved_model_{hyperparameter_string}", save_format="tf")
loaded_model = MyModel(tokenizer, 3, optimizer, cross_entropy_loss)
loaded_model.load_weights(f"saved_model_{hyperparameter_string}");
