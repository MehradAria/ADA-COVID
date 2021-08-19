import random
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras import regularizers

def build_embedding(param, inp):
    network = eval(param["network_name"])
    base = network(weights = 'imagenet', input_tensor=Input(shape=(224, 224, 3)), include_top = False, pooling ='avg')
    flat = base(inp)

    # base = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    # base = MaxPooling2D(pool_size=(2, 2))(base)
    # base = BatchNormalization(axis = -1)(base)

    # base = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), padding='same')(base)
    # base = MaxPooling2D(pool_size=(2, 2))(base)
    # base = BatchNormalization(axis = -1)(base)

    # base = Conv2D(64, (3, 3), activation='relu', padding='same')(base)
    # base = MaxPooling2D(pool_size=(2, 2))(base)
    # base = BatchNormalization(axis = -1)(base)
    # flat = GlobalAveragePooling2D()(base)
    
    return flat

def build_classifier(param, embedding):
    dense1 = Dense(400, name = 'class_dense1')(embedding)
    bn1 = BatchNormalization(name = 'class_bn1')(dense1)
    act1 = Activation('relu', name = 'class_act1')(bn1)
    drop2 = Dropout(param["drop_classifier"], name = 'class_drop1')(act1)

    dense2 = Dense(100, name = 'class_dense2')(drop2)
    bn2 = BatchNormalization(name = 'class_bn2')(dense2)
    act2 = Activation('relu', name = 'class_act2')(bn2)
    drop2 = Dropout(param["drop_classifier"], name = 'class_drop2')(act2)

    embeddings = Dense(embedding_size, activation=None)(drop2)
    embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)

# On 2nd train
    # densel = Dense(param["source_label"].shape[1], name = 'class_dense_last')(embeddings)
    # bnl = BatchNormalization(name = 'class_bn_last')(densel)
    # actl = Activation('softmax', name = 'class_act_last')(bnl)
    return embeddings # actl

def build_discriminator(param, embedding):
    dense1 = Dense(400, name = 'dis_dense1')(embedding)
    bn1 = BatchNormalization(name='dis_bn1')(dense1)
    act1 = Activation('relu', name = 'dis_act1')(bn1)
    drop1 = Dropout(param["drop_discriminator"], name = 'dis_drop1')(act1)

    dense2 = Dense(100, name = 'dis_dense2')(drop1)
    bn2 = BatchNormalization(name='dis_bn2')(dense2)
    act2 = Activation('relu', name = 'dis_act2')(bn2)
    drop2 = Dropout(param["drop_discriminator"], name = 'dis_drop2')(act2)

    densel = Dense(1, name = 'dis_dense_last')(drop2)
    bnl = BatchNormalization(name = 'dis_bn_last')(densel)
    actl = Activation('sigmoid', name = 'dis_act_last')(bnl)
    return actl

def build_combined_classifier(inp, classifier):
    comb_model = Model(inputs = inp, outputs = [classifier])
    return comb_model

def build_combined_discriminator(inp, discriminator):
    comb_model = Model(inputs = inp, outputs = [discriminator])
    return comb_model

def build_combined_model(inp, comb):
    comb_model = Model(inputs = inp, outputs = comb)
    return comb_model