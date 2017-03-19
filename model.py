import csv
import cv2
import numpy as np
import os
from keras.layers import Lambda
import sklearn
from random import shuffle

samples = []
with open('../project_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
#print out raw sample input
print("Total data:   {} samples".format(len(samples)))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#print out raw training and test set numbers 
print("Training Set:   {} samples".format(len(train_samples)))
print("Test Set:       {} samples".format(len(validation_samples)))

#use generator to process data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) # shuffle all the input data first
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                #find all 3 files names 
                center_name = '../project_data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = '../project_data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = '../project_data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)

                #get center_angel, add offset for left and right angle
                center_angle = float(batch_sample[3])
                left_angle = center_angle + 0.2 
                right_angle = center_angle - 0.2 

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            #augment data by flip the images and multiply angle by -1.0
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*(-1.0))

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3))) # trim image to only see section with road
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator,
    samples_per_epoch = len(train_samples)*6, #data augmented
    validation_data = validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=3, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()  # no GUI on AWS
 
model.save('model.h5')
exit()
