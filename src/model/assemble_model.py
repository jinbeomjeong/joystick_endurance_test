import tensorflow as tf
from tensorflow import keras
from src.model.layer import FeatureWiseScalingLayer, gelu_approximate
from src.model.model import time_mixer_block
from src.model.sub_layer import count_divisions_by_two
from tensorflow.keras.metrics import Precision, Recall, AUC


strategy = tf.distribute.MirroredStrategy()


with strategy.scope():
    def build_predict_model(input_shape, d_dims=64, dropout_rate=0.2, learning_rate=0.001):
        input_layer = keras.layers.Input(shape=input_shape)
        x_res = keras.layers.Dense(units=d_dims, activation=gelu_approximate)(input_layer)

        for i in range(count_divisions_by_two(input_shape[0])+1):
            dilation_rate = 2 ** i
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation=gelu_approximate, padding='causal',
                                    dilation_rate=dilation_rate)(x_res)
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation=gelu_approximate, padding='causal',
                                    dilation_rate=dilation_rate)(x)

            x_res = keras.layers.BatchNormalization()(x + x_res)
            x_res = keras.layers.Activation(gelu_approximate)(x_res)

        y = keras.layers.Flatten()(x_res)
        y = keras.layers.Dropout(dropout_rate)(y)

        y = FeatureWiseScalingLayer()(y)
        y_res = keras.layers.Dense(units=input_shape[0], activation='linear')(y)
        y_res = keras.layers.LayerNormalization()(y_res)

        for j in range(3):
            y = time_mixer_block(input_layer=y_res, pred_len=input_shape[0], dropout_rate=dropout_rate)
            y_res = y + y_res

        y = keras.layers.LayerNormalization()(y_res)
        
        y = FeatureWiseScalingLayer()(y)
        y = keras.layers.Dropout(dropout_rate)(y)
        y = keras.layers.Dense(units=1, activation='linear')(y)

        model = keras.models.Model(inputs=input_layer, outputs=y)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=keras.losses.logcosh,
                      metrics=['mean_absolute_error','mean_absolute_percentage_error'])

        return model