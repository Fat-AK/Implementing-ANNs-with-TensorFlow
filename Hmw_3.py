#Task 2.1
import tensorflow_datasets as tfds 
import tensorflow as tf

(train_ds, test_ds), ds_info = tfds.load(’mnist’, split=[’train’, ’
test’], as_supervised=True, with_info=True)

#Questions
#How many training/test images are there? 

#What’s the image shape?

#What range are pixel values in?

#Task 2.2

