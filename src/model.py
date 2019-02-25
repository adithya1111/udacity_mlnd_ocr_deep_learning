import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from custom_recurrents import AttentionDecoder


from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import OrderedDict
from io import BytesIO

slice_height = 40
# total_width = 200
divisor = 1
slice_width = int(20 / divisor)
max_slicenum = 24 * divisor
y_max_len = 40


class OCR_Model:
    def __init__(self,char_num,weights_path=None):

        # RNN Parameters
        OB_REG = 0.001
        OW_REG = 0.00001
        B_REG = 0.0001
        W_REG = 0.000001
        DROP_OUT = 0.2
        LEARN_RATE = 0.0001

        attn_num_hidden = 256

        # Model Definitions
        inp = Input(shape=(max_slicenum, slice_height, slice_width, 1))
        x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(inp)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
        x = BatchNormalization(axis=4)(x)
        x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
        x = BatchNormalization(axis=4)(x)
        x = TimeDistributed(Conv2D(512, (2, 2), padding='valid', activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 1)))(x)

        x = TimeDistributed(Flatten())(x)
        x = Dropout(DROP_OUT)(x)

        x = Bidirectional(GRU(attn_num_hidden, init='glorot_uniform', inner_init='orthogonal', return_sequences=False,
                              kernel_regularizer=l2(W_REG), bias_regularizer=l2(l=B_REG)),
                          merge_mode='concat')(x)
        x = Dropout(DROP_OUT)(x)

        x = RepeatVector(y_max_len)(x)
        x = AttentionDecoder(attn_num_hidden, char_num, activation='softmax', kernel_regularizer=l2(W_REG),
                             bias_regularizer=l2(l=B_REG))(x)

        # x = Dense(char_num, activation='softmax')(x)

        model = Model(inp, x)

        rmsprop = RMSprop(lr=LEARN_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop,metrics=['accuracy'])
        # print(model.summary())
        if weights_path is not None:            
            model.load_weights(weights_path)
        print("loaded model")
        # self.extractor = load_model(EXTRACTOR_PATH)
        self.model = model

    