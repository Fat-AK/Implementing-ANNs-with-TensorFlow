#import required packages 
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import datetime

#2.1 Prepare the data
#load the data
(train_ds, test_ds), ds_info = tfds.load('cifar10', split =['train', 'test'],
                                               as_supervised =True , with_info = True )

#display examples of images
tfds.show_examples(train_ds, ds_info)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#prepare the data
def prepare_cifar_data(cifar):
  #convert data 
  cifar = cifar.map(lambda img, target: (tf.cast(img, tf.float32),target))
  #input normalization
  cifar = cifar.map(lambda img, target: ((img/128.)-1., target))
  cifar = cifar.map(lambda img, target: (img, tf.cast(target,tf.int32)))
  #one-hot target
  cifar = cifar.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache progress in the memory
  cifar = cifar.cache()
  cifar = cifar.shuffle(2000)
  cifar = cifar.batch(16)
  cifar = cifar.prefetch(20)
  #return preprocessed dataset
  return cifar

train_ds = train_ds.apply(prepare_cifar_data)
test_ds = test_ds.apply(prepare_cifar_data)

def try_model(model, ds):
  for x, t in ds.take(5):
    y = model(x)
    
    
#2.2 CNN model
class cnn_model(tf.keras.Model):
    
    def __init__(self):
        
        super(cnn_model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()
        
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc") 
                       ]
        
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()   
        
        # building the layers
        self.convlayer1 = tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu')
        self.convlayer2 = tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu')
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')
        self.convlayer4 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.pooling(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.global_pool(x)
        x = self.out(x)
        return x
    
    #reset metrics 
    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
            
    # training and testing 
    @tf.function
    def train_step(self, data):
        
        x, targets = data
       
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.metrics[0].update_state(loss)
        
        #accuracy
        for metric in self.metrics[1:]:
            metric.update_state(targets,predictions)

        #keep track of trained data
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_st(self, data):
       
        x, targets = data
        predictions = self(x, training=False)
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        self.metrics[0].update_state(loss)
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}
      
  #2.3 Train the network
  def train_mod(epochs,batch_size,learning_rate):
    trainX, trainY, testX, testY = load_dataset()

  
    trainX, testX = prep_pixels(trainX, testX)

    #model definition
    mod = model_(learning_rate)
 
    past_data = mod.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=1)
    
    # Check the accuracy on the test dataset
    _, acc = mod.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
  
    return past_data
  
  #now train the model
epochs = 15
batch_size = 64
learning_rate = 0.001
past_data = train_mod(epochs,batch_size,learning_rate)

#2.4 visualize 
def result_visualization(past_data):
  plt.figure(figsize=(8,4))
  plt.subplot(211)
  plt.title('Cross_Entropy_Loss')
  plt.plot(past_data.past_data['loss'], color='green', label='train')
  plt.plot(past_data.past_data['val_loss'], color='yellow', label='test')
  plt.legend(loc="best")
  
 #3 change hyperparameters
epochs = 10
batch_size = 128
learning_rate = 0.004
history = train_mod(epochs,batch_size,learning_rate)
