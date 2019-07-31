#Importing Kera Libraries for Creating a Neural Network
 
from keras.models import Sequential
"""Initialize neural network model as a sequential network (sequence of layers or graph)"""

from keras.layers import Conv2D
"""Perform convolution operations by splitting images into 2 dimensional arrays"""

from keras.layers import MaxPooling2D
"""#Perform pooling operation to build cnn. Maxpooling provides the maximum value pixel from a region"""
 
from keras.layers import Flatten
"""Perform flattening proces by converting the 2 dimensional array into a single vector"""

from keras.layers import Dense
"""Perform the connection of the neural network"""

#create an object from sequential class
classifier = Sequential()

#add convolution layer
#arguments: number of filters, shape of filter, inpute shape/type, activation function (rectifier)
classifier.add(Conv2D(32,(3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#add pooling layer to reduce complexity
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#flatten pooled image into single vector
classifier.add(Flatten())

#create fully connected layer
#units is the number of nodes in this layer <input, output>
classifier.add(Dense(units = 128, activation = 'relu'))

#initialise output layer
classifier.add(Dense(units = 1, activation = 'sigmoid' ))
#compile
#optimizer: choose stochastic gradient descent algorithm
#loss: choose loss function
#metrics: performance metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentßßropy', metrics = ['accuracy'])

#preprocessing data to prevent  overfitting
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

#fit data to model
classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 25,
validation_data = test_set,
validation_steps = 2000)

#making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('test_image/cat_test.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'