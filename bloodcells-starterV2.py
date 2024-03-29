import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time
import keras
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Dense, Layer, Activation, Flatten, AveragePooling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


# !tar -xvzf dataset2-master.tar.gz


###
### CHANGE ME ---------------------------------------------------------
###
### Parameters for training
### Follow instructions to see which of these
### you should change at different places in
### the assignment
###
nx            = 240
ny            = 320
n_epochs      = 100
batch_size    = 64
nc            = 16  # number of hidden nodes
lss           = 'categorical_crossentropy'
opt           = keras.optimizers.Adam()
#opt           = keras.optimizers.Adadelta()
dfactor       = 4 # downsample factor
###
### END CHANGE ME -----------------------------------------------------
###



###
### load in files
### given folder location, constructs an np array
### for the data and its labels
###
def load(loc, celltypes):

    # make list of images and labels
    imglist      = [[]]*len(celltypes)
    labellist    = [[]]*len(celltypes)
    for cidx, cell in enumerate(celltypes):
        # extract list of files from directory
        filenames       = glob.glob(loc + '/' + cell + "/*.jpeg")
        # for each file, load the data and then cast to numpy array
        imglist[cidx]   = np.array(list(map( lambda file: plt.imread(file), filenames)))
        labellist[cidx] = cidx * np.ones((imglist[cidx].shape[0],1))

    # stack together each of the numpy arrays for each cell type
    imgs   = np.vstack(imglist)
    labels = np.vstack(labellist)[:,0]
    onehot = to_categorical(labels)

    return imgs, labels, onehot


###
### Load data
###
cell_types = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
rootloc    = './dataset2-master/images/' #'../dataset2-master/images/'

print('TRAIN')
train_imgs, train_labels_vec, train_labels = load( rootloc+'TRAIN', cell_types)
print('\t', train_imgs.shape)
print('\t', train_labels_vec.shape)
print('\t', train_labels.shape, '\n')

print('TEST')
test_imgs, test_labels_vec, test_labels = load( rootloc+'TEST', cell_types)
print('\t', test_imgs.shape)
print('\t', test_labels_vec.shape)
print('\t', test_labels.shape, '\n')


# cast to float32, since this is the tensorflow default
# rescale for inputs between [0,1]
type_in    = np.float32
train_imgs = train_imgs.astype(type_in)/255.0
test_imgs  = test_imgs.astype(type_in)/255.0

n_train       = train_imgs.shape[0]
n_test        = test_imgs.shape[0]
n_classes     = len(cell_types)
n_batches     = int(n_train / batch_size)

cbk           = [keras.callbacks.TerminateOnNaN()]
met           = ['accuracy']




###
### CHANGE ME ---------------------------------------------------------
###
### Basic feed-forward net
### Returns a Keras Model() of a one-hidden layer feed-forward network
### Ref: https://www.programcreek.com/python/example/89711/keras.layers.AveragePooling2D
###
def base_net():
    # ###
    # ### your code here
    # ###
    # layer_in = Input(shape=(nx, ny, 3), name='img_goes_here')
    # ### create your input layer here
    # ### downsample the data
    # layer = AveragePooling2D(pool_size=dfactor, padding='same')(layer_in)
    # layer = Flatten()(layer)
    # ###
    # layer = Dense(nc, input_dim=nx * ny * 3, init="uniform", activation="relu")(layer)
    # layer = Dense(len(cell_types), activation='softmax')(layer)
    # model = Model(input=layer_in, output=layer)

    model = Sequential()
    model.add(AveragePooling2D(pool_size=dfactor, padding='same', input_shape=(nx, ny, 3)))
    model.add(Dense(nc, input_dim=nx * ny * 3, init="uniform", activation="relu"))
    model.add(Flatten())
    model.add(Dense(4, activation="softmax"))
    # ### your code here
    # ###

    return model ### change this too
###
### END CHANGE ME -----------------------------------------------------
###


### Basic CNN (not news)
### Returns a Keras Model() of a one-hidden layer feed-forward network
###
def base_cnn():
    # ### your code here
    # ###
    model = Sequential()
    model.add(AveragePooling2D(pool_size=dfactor, padding='same', input_shape=(nx, ny, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(4, activation="softmax"))

    return model ### change this too

def improved_cnn():
    # ### your code here
    # ###
    model = Sequential()
    model.add(AveragePooling2D(pool_size=dfactor, padding='same', input_shape=(nx, ny, 3)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(4, activation="softmax"))


    return model ### change this too

###
### CHANGE ME ---------------------------------------------------------
###
### Dictionary of models
### For each model you create, add a key
### pair to the dictionary of the form below
models = { \
    "base_fnn"             :  base_net  ,
    "base_cnn"          :  base_cnn  ,
    "base_cnn_Adadelta"          :  base_cnn  ,
    #"improved_cnn"      :  improved_cnn  ,
### "model_example"  : get_model_example ,
    }
###
### END CHANGE ME -----------------------------------------------------
###





### Loop through the dictionary of models
### Train each model and output
###       -- model summary
###       -- training output
###       -- confusion matrix
###       -- train/prediction time
###
for m in models:

    print("Model: ", m)
    if m == "base_cnn_Adadelta":
        opt=keras.optimizers.Adadelta()
    net = models[m]()
    net.compile(loss=lss, optimizer=opt, metrics=met)

    print(net.summary())

    train_datagen   = ImageDataGenerator()
    train_generator = train_datagen.flow(train_imgs, train_labels, batch_size=batch_size)
    starttime       = time.time()
    history         = net.fit_generator(
                            train_generator,
                            steps_per_epoch=n_batches,
                            validation_steps=1, #how many batches for validation data
                            epochs=n_epochs,
                            callbacks=cbk,
                            shuffle=True)
    endtime         = time.time()
    traintime       = endtime - starttime
    pred            = net.predict(test_imgs).argmax(axis=1)
    predtime        = time.time() - endtime

    cmtrx = confusion_matrix(test_labels_vec, pred, labels=[*range(n_classes)])
    print("Confusion matrix:")
    print( cmtrx )

    print("Training time:\t\t", traintime)
    print("Prediction time:\t", predtime)
    print('\n')
