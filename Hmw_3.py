#Task 2.1
import tensorflow_datasets as tfds 
import tensorflow as tf

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

#Questions
#How many training/test images are there? 
# It contains 70,000 images (60k in the training set and 10k in the testing set)of handwritten digits from 0 to 9.

#What’s the image shape?
#Small square shape.

#What range are pixel values in?
# 28×28 pixel grayscale.

tfds.show_examples(train_ds , ds_info)
print(ds_info)

#Task 2.2
def prepare_mnist_data(mnist):
  #flatten into the vectors
  mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
  #convert data 
  mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization
  mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
  #create one-hot targets
  mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache this progress in memory
  mnist = mnist.cache()
  #shuffle, batch, prefetch
  mnist = mnist.shuffle(1000)
  mnist = mnist.batch(32)
  mnist = mnist.prefetch(20)
  #return preprocessed dataset
  return mnist

train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

#Task 2.3
from tensorflow.keras.layers import Dense

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x
      
#Task 2.4
def train_step(model, input, target, loss_function, optimizer):
  
  with tf.GradientTape() as tape:
    prediction = model(input)
    loss = loss_function(target, prediction)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  # test complete test data

  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy

tf.keras.backend.clear_session()

#For showcasing we only use a subset of the training and test data (generally use all of the available data!)
train_dataset = train_dataset.take(1000)
test_dataset = test_dataset.take(100)

### Hyperparameters
num_epochs = 10
learning_rate = 0.1

# Initialize the model.
model = MyModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for visualization.
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

#testing 
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check model performence 
train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training 
    epoch_loss_agg = []
    for input,target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    
#Task 2.5
import matplotlib.pyplot as plt

plt.figure()
line1 , = plt.plot(train_losses , "b-")
line2 , = plt.plot(test_losses , "r-") 
line3 , = plt.plot(train_accuracies , "b:")
line4 , = plt.plot(test_accuracies , "r:") 
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
plt.show()


#Task 3.0

# Onour main training data we have number of epochs = 10, learning rate =0.1. Both of them are small numbers so we can see that test loss and training are parallel to each other.
# Now if we increase the number of epochs and also learning rate as follow: number of epochs =20, learning rate =50. 
#As the learning is quite big so we can see both the training loss and test loss increase rapidly and accuracy reduces to almost zero.

# Now we, increase the number of epochs but keep the learning rate as initial value.
# Here, number of epochs = 20, learning rate = 0.1
#Because of increased number of epochs compared to step 1, the model learns better so we can see cross section between training loss and test loss lines and also test loss is lower than training loss after about 40% of training steps.

# Now, we increase the number of epochs and decrease the learning rate ( a kind of ideal state for model to learn.
# Here, number of epochs = 200, learning rate = 0.001
#Upon changing the hyper-parameters to the mentioned values we can see that lower limit of test loss at 100% of training steps approaches to 0%. Also we see that the cross-section between test loss and training loss has shifted from 40% to 15%. The accuracy also approaches to 1 (i.e. 100%). 

#Now, we exponentially increase and decrease the number of epochs and the learning rate respectively.
# Here, number of epochs  = 10000, learning rate = 0.0001
#As too much increase and decrease in the hyper-parameters also results in bad learning of model, the example in this step shows the same.
#We can see both lower limit of training loss and test loss at 100% of the training steps got increased by significant amount, which means that the model will produce more loss as compared to parameters taken in above steps i.e. model is learning badly.


