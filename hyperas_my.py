from __future__ import print_function
from sklearn.utils import shuffle
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adagrad
from keras.applications.xception import Xception, preprocess_input
from keras.callbacks import EarlyStopping
from keras import regularizers
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.preprocessing import image
import numpy as np


def data():
    
    train_batch_size = 32
    val_batch_size = 32

    def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x
    train_datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=.2,
                            preprocessing_function=preprocess_input,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

    # 训练集的图片生成器
    train_gen = train_datagen.flow_from_directory(directory='data/train/',
                                  target_size=(299,299),
                                  batch_size=train_batch_size)

    # 验证集的图片生成器
    val_gen = train_datagen.flow_from_directory(directory='data/valid/',
                                  target_size=(299,299),
                                  batch_size=val_batch_size)
    return train_gen, val_gen


def create_model(train_gen, val_gen):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    base_model = Xception(include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2({{uniform(0, 0.1)}}))(x)
    x = Dropout({{uniform(0, 1)}})(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adam')

    model.fit_generator(train_gen,
              steps_per_epoch=1,
              epochs=1,
              verbose=2,
              validation_data=val_gen)
    score, acc = model.evaluate_generator(val_gen, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def run_hyperas():
    best_run, best_model = optim.minimize(model=create_model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=5,
                              trials=Trials())
    train_gen, val_gen = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(val_gen))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
