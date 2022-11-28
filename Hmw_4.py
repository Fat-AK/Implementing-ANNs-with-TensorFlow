#Task 2
# In the first section we import the necessary packages for loading the MNIST dataset and building a neural network.    

import tensorflow_datasets as tfds

 

import tensorflow as tf

# Here we define a preprocessing function that can be used to format the data for either of the two subtasks. This function takes as input a dataset and a condition, and outputs a dataset that is formatted according to the specified condition. 

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

#Here we define a train function that can be used to train a neural network for either of the two subtasks. This function takes as input a subtask and an optimizer, and outputs a trained neural network.
 

def preprocessing_fn(ds, condition):

  if condition == 'subtraction':

    pass

  elif condition == 'larger_than_five':

    pass

  else:

    raise ValueError('Condition must be either "subtraction" or "larger_than_five".')

    

#In this section we use the train function to train a neural network for the subtask of subtracting two MNIST digits.

def train_fn(subtask, optimizer):

  if subtask not in ['subtraction', 'larger_than_five']:

    raise ValueError('Subtask must be either "subtraction" or "larger_than_five".')

    

  ds = preprocessing_fn(train_ds, subtask)

  

  if subtask == 'subtraction':

    model = tf.keras.Sequential([

      tf.keras.layers.Input(shape=(28, 28, 1)),

      tf.keras.layers.Conv2D(32, 3, activation='relu'),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(10, activation='softmax')

    ])

    

    model.compile(

      optimizer=optimizer,

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy']

    )

    

    model.fit(

      ds,

      epochs=5

    )

    

  elif subtask == 'larger_than_five':

    model = tf.keras.Sequential([

      tf.keras.layers.Input(shape=(28, 28, 1)),

      tf.keras.layers.Conv2D(32, 3, activation='relu'),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    

    model.compile(

      optimizer=optimizer,

      loss='binary_crossentropy',

      metrics=['accuracy']

    )

    

    model.fit(

      ds,

      epochs=5

    )

    
#In this secion we use the train function to train a neural network for the subtask of determining whether the sum of two MNIST digits is greater than five.
    

train_fn('subtraction', tf.keras.optimizers.Adam())

train_fn('larger_than_five', tf.keras.optimizers.Adam()) 
                                                                                                    
                                                                                                    
                                                                                             
                                  
