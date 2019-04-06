import cv2
import csv
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Activation, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D

def defineDataPaths():
    """
    define datapaths of dataset
    """
    dataPaths = list()
    #dataPaths.append("./data")
    dataPaths.append("./Data/Datasample_udacity")
    #dataPaths.append("./01_Data/Track_1_back")
    #dataPaths.append("./01_Data/Track_1_forw")
    #dataPaths.append("./01_Data/Track_1_rec")
    return dataPaths

def getLinesFromDrivingLogs(dataPath):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        
        # skipp first line (header of csv)
        next(reader, None)
        
        for line in reader:
            lines.append(line)
    return lines

def sortImages(dataPaths):
    """
    sorts all paths of the images for the right left and center image
    """
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataPaths:
        lines = getLinesFromDrivingLogs(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)


def corrSteeringAngle(centerMeasurement, corrFac):
    """
    corrects the steering angel for the left and right image by adding(left img)/substracting(right img) 'corrFac' to the measurement
    """
    leftMeasurement = [x+corrFac for x in centerMeasurement]
    rightMeasurement = [x-corrFac for x in centerMeasurement]
    
    return leftMeasurement, rightMeasurement


def combineImagePaths(centerImagePath, leftImagePath, rightImagePath, centerMeasurement, leftMeasurement, rightMeasurement):
    """
    combines cnter/left/right images and measurements to one list
    """
    # combine measurements
    measurements = []
    measurements.extend(centerMeasurement)
    measurements.extend(leftMeasurement)
    measurements.extend(rightMeasurement)
    
    # combine image paths 
    imagePaths = []
    imagePaths.extend(centerImagePath)
    imagePaths.extend(leftImagePath)
    imagePaths.extend(rightImagePath)
    
    return imagePaths, measurements


def generator(samples, batchSize=32):
    """
    Generate the required images and measurments for training
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batchSize):
            batch_samples = samples[offset:offset+batchSize]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                
                # Flipping image and steering angle to abtain more data (double)
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # convert to np array
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

            
def modelNVIDIA(actFct='relu'):
    """
    Building the model according to Nvidia's Model from https://arxiv.org/pdf/1604.07316v1.pdf
    using the activation function 'actFct'
    """
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3))) 
    model.add(Cropping2D(cropping=((66,22), (0,0)))) # cropping image so that only the street is on the picture
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation=actFct))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation=actFct))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation=actFct))
    model.add(Convolution2D(64,3,3,activation=actFct))
    model.add(Convolution2D(64,3,3,activation=actFct))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(1164))
    model.add(Activation(actFct))
    model.add(Dense(100))
    model.add(Activation(actFct))
    model.add(Dense(50))
    model.add(Activation(actFct))
    model.add(Dense(10))
    model.add(Activation(actFct))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model
            
            
## Load and prepare data
corrFac = 0.2 # factor for the correction of the steering angle -> left/right pictures   
testRatio = 0.2 # defines size of test dataset compared to the training dataset 
batchSize = 32 # batch size for training and validation
actFct = 'relu' # define activation function for model
epochs = 1

# get datapaths
dataPaths = defineDataPaths()

# sort image paths
[centerImagePath, leftImagePath, rightImagePath, centerMeasurement] = sortImages(dataPaths)

# correct steering angle for left and right image

leftMeasurement, rightMeasurement = corrSteeringAngle(centerMeasurement, corrFac)

# create complete list of image paths and measurements
imagePaths, measurements = combineImagePaths(centerImagePath, leftImagePath, rightImagePath, centerMeasurement, leftMeasurement, rightMeasurement)

## Create and train the model
# Split data 
trainData, validationData = train_test_split(list(zip(imagePaths, measurements)), test_size = testRatio)

# Create generators          
trainGen = generator(trainData, batchSize)
validationGen = generator(validationData, batchSize)
              
# Create the model
model = modelNVIDIA(actFct)

# Compiling model
model.compile(loss='mse', optimizer='adam')
              
# Train the model
history_object = model.fit_generator(trainGen, samples_per_epoch= \
                 len(trainData), validation_data=validationGen, \
                 nb_val_samples=len(validationData), nb_epoch=epochs, verbose=1)

# Save the model
model.save('model.h5')
              
# Show training history
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


