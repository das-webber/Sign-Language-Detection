# %%
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(num_classes, input_shape=(224,224,3)):
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg')
    x = base.output
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model

def build_cnn_lstm(num_classes, frame_shape=(224,224,3), timesteps=16):
    cnn = build_cnn(num_classes=None, input_shape=frame_shape)
    cnn.trainable = False
    inp = layers.Input(shape=(timesteps,)+frame_shape)
    x = layers.TimeDistributed(cnn)(inp)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

# %%



