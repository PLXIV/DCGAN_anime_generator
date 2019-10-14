import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.layers import Dense

from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
 
np.random.seed(1)

class NN_architectures():

    def __init__(self, noise_shape, img_shape, initialize=False):
        self.noise_shape = noise_shape
        self.img_shape = img_shape
        self.gen_input = Input(shape=self.noise_shape)
        self.dis_inp_dis = Input(shape=self.img_shape)

        if initialize:
            self.generator = self.generator_architecture()
            self.compile_generator()
            self.discriminator = self.discriminator_architecture()
            self.compile_discriminator()
            self.gan = self.gan_generator()
        else:
            self.generator = None
            self.discriminator = None
            self.gan = None

    def generator_architecture(self):
        kernel_init = 'glorot_uniform'
        momentum = 0.6
        dropout_prob = 0.1
        generator=Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(1,1), padding="valid", data_format="channels_last", kernel_initializer=kernel_init)(self.gen_input)
        generator=BatchNormalization(momentum=momentum)(generator)
        generator=LeakyReLU(0.2)(generator)

        generator=Dropout(dropout_prob)(generator)
        generator=Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(generator)
        generator=BatchNormalization(momentum=momentum)(generator)
        generator=LeakyReLU(0.2)(generator)

        generator=Dropout(dropout_prob)(generator)
        generator=Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(generator)
        generator=BatchNormalization(momentum=momentum)(generator)
        generator=LeakyReLU(0.2)(generator)

        generator=Dropout(dropout_prob)(generator)
        generator=Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(generator)
        generator=BatchNormalization(momentum=momentum)(generator)
        generator=LeakyReLU(0.2)(generator)

        generator=Dropout(dropout_prob)(generator)
        generator=Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(generator)
        generator=BatchNormalization(momentum=momentum)(generator)
        generator=LeakyReLU(0.2)(generator)

        generator=Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(generator)
        generator=Activation('sigmoid')(generator)

        return generator

    def compile_generator(self):
        gen_opt=Adam(lr=0.00015, beta_1=0.5)
        generator_model = Model(input=self.gen_input, output=self.generator)
        generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
        generator_model.summary()
        self.generator = generator_model


    def discriminator_architecture(self):

        dropout_prob=0.1
        momentum = 0.6
        kernel_init='glorot_uniform'

        discriminator =Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(self.dis_inp_dis)
        discriminator =LeakyReLU(0.2)(discriminator)

        discriminator=Dropout(dropout_prob)(discriminator)
        discriminator = Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(discriminator)
        discriminator = BatchNormalization(momentum=momentum)(discriminator)
        discriminator = LeakyReLU(0.2)(discriminator)
        #discriminator=MaxPooling2D(pool_size=(2, 2))(discriminator)

        discriminator=Dropout(dropout_prob)(discriminator)
        discriminator = Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(discriminator)
        discriminator = BatchNormalization(momentum=momentum)(discriminator)
        discriminator = LeakyReLU(0.2)(discriminator)
        #discriminator=MaxPooling2D(pool_size=(2, 2))(discriminator)

        discriminator=Dropout(dropout_prob)(discriminator)
        discriminator = Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=kernel_init)(discriminator)
        discriminator = BatchNormalization(momentum=momentum)(discriminator)
        discriminator = LeakyReLU(0.2)(discriminator)
        #discriminator=MaxPooling2D(pool_size=(2, 2))(discriminator)

        discriminator = Flatten()(discriminator)
        discriminator = Dense(1)(discriminator)
        discriminator = Activation('sigmoid')(discriminator)

        return discriminator

    def compile_discriminator(self):
        dis_opt = Adam(lr=0.0002, beta_1=0.5)
        discriminator_model = Model(input=self.dis_inp_dis, output=self.discriminator)
        discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
        discriminator_model.summary()
        self.discriminator = discriminator_model

    def gan_generator(self):
        self.discriminator.trainable = False
        opt = Adam(lr=0.00015, beta_1=0.5)
        gen_inp = Input(shape=self.noise_shape)
        gen_out = self.generator(gen_inp)
        disc_out = self.discriminator(gen_out)
        gan = Model(input=gen_inp, output=disc_out)
        gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        #gan.summary()
        self.discriminator.trainable = True
        return gan

