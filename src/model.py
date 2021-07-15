#general imports
import numpy as np
import pandas as pd
import os

#model imports
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks


class CNN():
    def __init__(self, train_dir, val_dir, test_dir, n_classes, target_size=(64,64), batch_size = 32, n_steps=100,n_epoch = 10):
        self.model = Sequential()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.train_df = self.get_df(train_dir,'train.csv')
        self.val_df = self.get_df(val_dir,'val.csv')
        self.test_df = self.get_df(test_dir,'test.csv')

        self.n_classes = n_classes
        self.target_size = target_size
        # self.input_size = target_size+(3)
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_epoch = n_epoch

        self.n_train = len(os.listdir(train_dir))
        self.n_val = len(os.listdir(val_dir))
        self.n_test = len(os.listdir(test_dir))

        # self.preprocessing = preprocessing
        self.history = None

    def conv(self):
        '''
        Create Conv/Pooling layers
        '''
        #first conv layer
        self.model.add(Conv2D(32, (3, 3), input_shape=(64,64,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(2,2))

        #second conv layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2)))
        
        #third conv layer
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        #fourth conv layer and pooling
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(2,2))
        
        #flatten to dense layers
        self.model.add(Flatten())
        
        #add dropout layers to prevent overfitting
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048))
        self.model.add(Activation('relu'))
        
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048))
        self.model.add(Activation('relu'))
        
        #output layer
        self.model.add(Dense(37))
        self.model.add(Activation('sigmoid'))

    def get_generator(self):
        '''
        generator function to grab batches of images with augmentation
        '''
        train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=360,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip=True
                    )
        val_datagen = ImageDataGenerator(rescale=1./255) 
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_gen = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory = self.train_dir,
            x_col = 'asset_id',
            y_col = list(self.train_df.columns[1:]),
            target_size = self.target_size,
            class_mode = 'raw',
            batch_size = self.batch_size
        )
        self.val_gen = val_datagen.flow_from_dataframe(
            dataframe=self.val_df,
            directory = self.val_dir,
            x_col = 'asset_id',
            y_col = list(self.val_df.columns[1:]),
            target_size = self.target_size,
            class_mode = 'raw',
            batch_size = self.batch_size,
        )
        self.test_gen = test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory = self.test_dir,
            x_col = 'asset_id',
            y_col = list(self.test_df.columns[1:]),
            target_size = self.target_size,
            class_mode = 'raw',
            batch_size = self.batch_size,
            shuffle=False
        )
    def fit(self):
        self.conv()
        self.get_generator()
        optimizer = optimizers.Adam(lr=1e-6)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mse'])
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='auto')
        checkpoint = callbacks.ModelCheckpoint(filepath=('evaluate/checkpoint.hdf5'), verbose=1, save_best_only=True)
        hist = callbacks.History()
        #initialize fit generator
        self.model.summary()

        self.history = self.model.fit(
            self.train_gen,
            steps_per_epoch=self.n_train//self.batch_size,
            epochs=self.n_epoch,
            validation_data=self.val_gen,
            validation_steps = self.n_steps//self.n_val,
            callbacks=[checkpoint,early_stopping, hist]
        )

    def get_df(self,dir,csv):
        '''
        read target csv and clean nan values
        '''
        df = pd.read_csv(dir+'/'+csv)
        df['asset_id']=df['asset_id'].astype('str')+'.jpg'
        df=df.dropna()
        return df

    def test_gen(self):
        '''
        a function for debugging the generator functions
        '''
        train_datagen = ImageDataGenerator(
                    rescale=1./255
                    # rotation_range=360,
                    # width_shift_range = 0.2,
                    # height_shift_range = 0.2,
                    # zoom_range = 0.2,
                    # horizontal_flip=True
                    )
        self.train_gen = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory = self.train_dir,
            x_col = 'asset_id',
            y_col = list(self.train_df.columns[1:]),
            target_size = self.target_size,
            class_mode = 'raw',
            batch_size = self.batch_size,
            shuffle=False
        )
        self.batch = next(self.train_gen)
        print(self.batch[1])



if __name__=='__main__':
    train_dir = 'data/img/train'
    val_dir = 'data/img/validation'
    test_dir = 'data/img/test'
    n_classes = 37 #number of feature columns
    target_size=(64,64) #this is what the images are cropped to in data_pipe.py
    batch_size = 32
    n_steps=1000
    n_epoch=10

    model = CNN(train_dir,val_dir,test_dir, n_classes, target_size, batch_size, n_steps,n_epoch)
    # model.test_gen()
    model.fit()
