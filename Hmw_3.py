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

