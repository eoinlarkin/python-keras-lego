#%%
import numpy as np
import os
import cv2

#####################################################################
### Defining our Data Directory
#####################################################################

#%%
# Directory with Data
os.chdir("C:/Users/Eoin/OneDrive/Data Science/Github/python_keras_lego") #Changing the working directory
DATADIR = os.getcwd() +  "/Lego/"

DATADIR_valid = DATADIR+ "valid"
DATADIR_train = DATADIR+ "train"
CATEGORIES = ["3022 Plate 2x2", "3069 Flat Tile 1x2", "3040 Roof Tile 1x2x45deg","6632 Technic Lever 3M"]
SIZE = 32
#SIZE=64 #Used to analyse impact of using size 64 instead of 32

#####################################################################
### FITTING OUR KERAS MODEL
####################################################################
#%%
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D  
import sklearn.preprocessing as skp
import random
from numpy.random import seed
seed(1) #Setting random seed to ensure consistent results
tf.random.set_seed(2)

#%%
#Following function creates our Keras model data by loading the resized data from the data folder
def create_keras_data(DIR):
    keras_data = []
    for category in CATEGORIES:

        #path = Path(DATADIR +'/'+category)  # create path to dogs and cats
        path = os.path.join(DIR,category)
        print(path)
        class_num = CATEGORIES.index(category)  # get the classification  (0,1,2,3). 
        for img in tqdm(os.listdir(path)):  # iterate over each image (tqdm is progress bar)
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (SIZE, SIZE))  # resize to normalize data size
                keras_data.append([new_array, class_num])  # add this to our training_data
    random.shuffle(keras_data)
    return keras_data

#%%
#Following function formats our Keras data, splitting between data and categories
def format_keras_data(data):
    X = []
    Y = []

    for features,label in data:
        X.append(features)
        Y.append(label)
    X = np.array(X).reshape(-1, SIZE, SIZE,1)
    Y = np.array(Y)
    X = tf.keras.utils.normalize(X, axis=-1, order=1)#L1 normalization
    return X,Y

#%%
training_data = create_keras_data(DATADIR_train) 
valid_data = create_keras_data(DATADIR_valid)
X,Y = format_keras_data(training_data)
X_valid, Y_valid = format_keras_data(valid_data)

###################################
##KERAS Model Definition
##################################
#%%
model = Sequential()
from numpy.random import seed
seed(1) #Setting random seed to ensure consistent results
tf.random.set_seed(2)


model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu')) #activation layer rectified linear
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Dense(len(CATEGORIES), activation='softmax'))
model.save_weights('model.keras')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%
#Calculating accuracy by class
def keras_accuracy(X,Y):
    accuracy = [0,0,0,0]
    n = np.shape(X)[0]
    for i in range(n):
        test=X[i,:,:,:].reshape(-1,SIZE,SIZE,1)
        if(np.argmax(model.predict(test)) == Y[i]) :
            accuracy[Y[i]] = accuracy[Y[i]] + 1
    accuracy = [a / (n/4) for a in accuracy]
    return accuracy

model.load_weights('model.keras')
keras_lego1 = model.fit(X, Y, batch_size=32, epochs=3) 

accuracy_train =keras_accuracy(X,Y) 
accuracy_valid =keras_accuracy(X_valid,Y_valid)

##Following is used for testing 15 epochs
#seed(1)
#model.load_weights('model.keras')
#keras_lego2 = model.fit(X, Y, batch_size=32, epochs=15) 

##Used to save accuracy for 15 Epochs
#accuracy_train_15e =keras_accuracy(X,Y) 
#accuracy_valid_15e =keras_accuracy(X_valid,Y_valid)

#Used to save accuracy for 64 Image size
#accuracy_train_64 =keras_accuracy(X,Y) 
#accuracy_valid_64 =keras_accuracy(X_valid,Y_valid)



###################################################
##############BOKEH PLOTTING#######################
###################################################
#%%
from bokeh.io import show
from bokeh.io import output_file 
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import export_png
from bokeh.models import NumeralTickFormatter
from bokeh.layouts import row

#-------------------------------------------------#
#----Plotting the block Shapes as Images----------#
#-------------------------------------------------#
#%%
#Path names of first shape in each directory
def brick_path(n):
    fullpath = []
    for i in range(4):
        path = os.path.join(DATADIR_train,CATEGORIES[i])  # create path to bricks
        img = os.listdir(path)[n-1:n]  # iterate over each image per category
        fullpath.append(os.path.join(path,img[0]))
    return fullpath

# Plotting the original images before resizing
fullpath = brick_path(1) #Gets the path for the 1st brick in each directory
output_file('image.html')
p1 = figure(x_range=(0,1), y_range=(0,1), title=CATEGORIES[0]) #toolbar_location=None, title=CATEGORIES[0])

p1.image_url(url=[fullpath[0]] , x=0, y=0, w=1, h=1,anchor="bottom_left")
p2 = figure(x_range=(0,1), y_range=(0,1),title=CATEGORIES[1]) #toolbar_location=None, 
p2.image_url(url=[fullpath[1]] , x=0, y=0, w=1, h=1,anchor="bottom_left")
p3 = figure(x_range=(0,1), y_range=(0,1), title=CATEGORIES[2]) #toolbar_location=None, )
p3.image_url(url=[fullpath[2]] , x=0, y=0, w=1, h=1,anchor="bottom_left")
p4 = figure(x_range=(0,1), y_range=(0,1),title=CATEGORIES[3]) #toolbar_location=None, )
p4.image_url(url=[fullpath[3]] , x=0, y=0, w=1, h=1,anchor="bottom_left")
show(row(p1, p2, p3, p4))
#export_png(row(p1, p2, p3, p4), filename="plot_blocks_raw.png")

#%%
#-------------------------------------------------#
#----Plotting the block Shapes as Images----------#
#-------------------------------------------------#
# Plotting the resized images that will be used by the model
p1_new = np.flipud(cv2.resize(cv2.imread(fullpath[0] ,cv2.IMREAD_GRAYSCALE), (SIZE, SIZE)))
p2_new = np.flipud(cv2.resize(cv2.imread(fullpath[1] ,cv2.IMREAD_GRAYSCALE), (SIZE, SIZE)))
p3_new = np.flipud(cv2.resize(cv2.imread(fullpath[2] ,cv2.IMREAD_GRAYSCALE), (SIZE, SIZE)))
p4_new = np.flipud(cv2.resize(cv2.imread(fullpath[3] ,cv2.IMREAD_GRAYSCALE), (SIZE, SIZE)))

p1 = figure(x_range=(0,256), y_range=(0,256), width=400, height=400,toolbar_location=None, title=CATEGORIES[0])
p1.image(image=[p1_new], x=0, y=0, dw=256, dh=256, palette="Viridis256")

p1 = figure(x_range=(0,256), y_range=(0,256), width=400, height=400,toolbar_location=None,title=CATEGORIES[0])
p1.image(image=[p1_new], x=0, y=0, dw=256, dh=256, palette="Viridis256")
p2 = figure(x_range=(0,256), y_range=(0,256), width=400, height=400,toolbar_location=None, title=CATEGORIES[1])
p2.image(image=[p2_new], x=0, y=0, dw=256, dh=256, palette="Viridis256")
p3 = figure(x_range=(0,256), y_range=(0,256), width=400, height=400,toolbar_location=None, title=CATEGORIES[2])
p3.image(image=[p3_new], x=0, y=0, dw=256, dh=256, palette="Viridis256")
p4 = figure(x_range=(0,256), y_range=(0,256), width=400, height=400,toolbar_location=None, title=CATEGORIES[3])
p4.image(image=[p4_new], x=0, y=0, dw=256, dh=256, palette="Viridis256")

show(row(p1, p2, p3, p4))
#export_png(row(p1, p2, p3, p4), filename="plot_blocks_resized.png")

#%%
#-------------------------------------------------#
#--------------Line plot of fit Accuracy----------#
#-------------------------------------------------#

p1 = figure(plot_width=400, plot_height=400, title="Kears Model Accuracy after 3 Epochs",toolbar_location=None)
p1.xaxis.ticker = list(range(1,16))
p1.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
p1.line(list(range(1,16)), keras_lego1.history['acc'], line_width=2)

p2 = figure(plot_width=400, plot_height=400, title="Kears Model Loss after 3 Epochs", toolbar_location=None)
p2.xaxis.ticker = list(range(1,16))
p2.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
p2.line(list(range(1,16)), keras_lego1.history['loss'], line_width=2,line_color = "#e84d60")
show(row(p1, p2))

##15 EPOCH Graph
#p1 = figure(plot_width=400, plot_height=400, title="Kears Model Accuracy after 15 Epochs",toolbar_location=None)
#p1.xaxis.ticker = list(range(1,16))
#p1.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
#p1.line(list(range(1,16)), keras_lego2.history['acc'], line_width=2)
#
#p2 = figure(plot_width=400, plot_height=400, title="Kears Model Loss after 15 Epochs", toolbar_location=None)
#p2.xaxis.ticker = list(range(1,16))
#p2.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
#p2.line(list(range(1,16)), keras_lego2.history['loss'], line_width=2,line_color = "#e84d60")
#show(row(p1, p2))



#-------------------------------------------------#
#--------------Bar plot of accuracy by shape------#
#-------------------------------------------------#

def bokeh_accuracy(accur1,accur2,label1,label2,title):
    palette = ["#718dbf", "#e84d60"]
    
    x = [ (brick, model) for brick in CATEGORIES for model in (label1, label2)]
    acc = list(zip(accur1, accur2))
    acc = [item for sublist in acc for item in sublist]
    
    source = ColumnDataSource(data=dict(x=x, acc=acc))
    
    p = figure(x_range=FactorRange(*x), plot_height=350, title=title,
               toolbar_location=None, tools="")
    
    p.vbar(x='x', top='acc', width=0.9, source=source, line_color="white",
           fill_color=factor_cmap('x', palette=palette, factors=(label1, label2), start=1, end=2))
    
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    export_png(p, filename="plot.png")
    show(p)

#Graph of Accuracy for Keras train and validation
bokeh_accuracy(accuracy_train, accuracy_valid, "Training", "Validtion", "Model Accuracy for Training & Validation Datasets")

##Graph of Accuracy for Keras validation 32 vs validation 64
#bokeh_accuracy(accuracy_valid, accuracy_valid_64, "Validation - 32 Size", "Validation - 64 Size", "Model Accuracy for Validation Datasets - Size 32 vs 64")

##Graph of Accuracy for Keras validation 3 Epochs vs 15 Epochs
#bokeh_accuracy(accuracy_valid, accuracy_valid_15e, "Validation - 3 Epochs", "Validation - 15 Epochs", "Model Accuracy for Validation Datasets - Epochs 3 vs 15")

#


# %%
