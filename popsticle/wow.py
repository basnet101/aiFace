# path of the training images
TrainingImagePath = './cottonCandy/'

# Importing necessary library
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
import numpy as np
from keras.utils import load_img, img_to_array


# ImageDataGenerator is made for augmentation during training session
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

# calling image data generator function and storing it in variable
test_datagen = ImageDataGenerator()

#enhancement generator with target size of image, batch size and class mode to classification 
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# class indices for class number, starting from 0
test_set.class_indices

#mapping of class indices to class names in a dictionary
TrainClasses = training_set.class_indices
ResultMap = {}
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName

# Create a pickle file with the ResultMap dictionary.
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# Print the mappingof class
print("Mapping of Face and its ID:", ResultMap)

# Based on the number of classes, estimate the number of output neurons.
OutputNeurons = len(ResultMap)
#  print the number of neurons to console
print('\nThe Number of output neurons:', OutputNeurons)

# Create a sequential model
classifier = Sequential()

# convolutional layer with input shap 64 by 64 and activation relu
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))

# max pooling layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# another convolutional layer
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

#max pooling layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Flattening the image
classifier.add(Flatten())

# connected layer with 64unit and activation relu
classifier.add(Dense(64, activation='relu'))

# output layer multiclass classification for class classification
classifier.add(Dense(OutputNeurons, activation='softmax'))

# compiling with adam optimier as it is best and most used for this kind of project with metrix of accuracy
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# training the model with variables and epochs itaration
import time
StartTime = time.time()
classifier.fit_generator(
                    training_set,
                    epochs=10,
                   
                    validation_data=test_set,
                    validation_steps=10)

#importing image for testing
# image path
ImagePath = './blueberry/1.jpg'
# loding image with sirze 64 by 64
test_image = load_img(ImagePath, target_size=(64, 64))
# creating array of the image we provided earlier
test_image = img_to_array(test_image)

# dimention of image expanded to match the picture size
test_image = np.expand_dims(test_image, axis=0)

# predict the face in the image
result = classifier.predict(test_image, verbose=0)

# prints prediction class in the console log
print('Prediction is:', ResultMap[np.argmax(result)])