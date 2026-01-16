from tensorflow import keras
from src.model.sub_layer import count_divisions_by_two
from src.model.layer import DecompositionLayer, gelu_approximate


def time_mixer_block(input_layer, pred_len=1, go_backward=False, dropout_rate=0.2):
    input_raw = keras.ops.reverse(input_layer, axes=1) if go_backward else input_layer
    input_raw = keras.ops.expand_dims(input_raw, axis=2)

    multi_scale_input_list = [input_raw]

    for i in range(count_divisions_by_two(input_raw.shape[1])-1):
        i = (i*2)+2
        avg_layer = keras.layers.AveragePooling1D(pool_size=i, strides=i, padding='valid')(input_raw)
        multi_scale_input_list.append(avg_layer)

    seasonal_list = []
    trend_list = []

    for multi_scale_input_layer in multi_scale_input_list:
        seasonal, trend = DecompositionLayer(kernel_size=3)(multi_scale_input_layer)

        seasonal = keras.ops.squeeze(seasonal, axis=2)
        seasonal_output = keras.layers.Dense(units=multi_scale_input_layer.shape[1], activation='linear')(seasonal)
        seasonal_output = keras.layers.Dropout(dropout_rate)(seasonal_output)
        seasonal_list.append(seasonal_output)

        trend = keras.ops.squeeze(trend, axis=2)
        trend_output = keras.layers.Dense(units=multi_scale_input_layer.shape[1], activation='linear')(trend)
        trend_output = keras.layers.Dropout(dropout_rate)(trend_output)
        trend_list.append(trend_output)

        #output_list.append(keras.layers.Add()([seasonal_output, trend_output]))

    output_1 = seasonal_list[0]
    seasonal_mix_list = [output_1]

    for i in range(len(seasonal_list)-1):
        output_1 = keras.layers.Dense(units=seasonal_list[i+1].shape[1], activation='linear')(output_1)
        output_1 = keras.layers.LayerNormalization()(output_1)
        output_1 = keras.layers.Dropout(dropout_rate)(output_1)
        output_1 = keras.layers.Activation(gelu_approximate)(output_1) #gelu
        output_1 = keras.layers.add([output_1, seasonal_list[i+1]])
        seasonal_mix_list.append(output_1)

    trend_list.reverse()
    output_2 = trend_list[0]
    trend_mix_list = [output_2]

    for i in range(len(trend_list)-1):
        output_2 = keras.layers.Dense(units=trend_list[i+1].shape[1], activation='linear')(output_2)
        output_2 = keras.layers.LayerNormalization()(output_2)
        output_2 = keras.layers.Dropout(dropout_rate)(output_2)
        output_2 = keras.layers.Activation(gelu_approximate)(output_2) #gelu
        output_2 = keras.layers.add([output_2, trend_list[i+1]])
        trend_mix_list.append(output_2)

    trend_mix_list.reverse()

    mix_output_list = []
    hidden_units = 128

    for seasonal_mix_layer, trend_mix_layer in zip(seasonal_mix_list, trend_mix_list):
        mix_output = seasonal_mix_layer+trend_mix_layer
        mix_output = keras.layers.Dense(units=hidden_units, activation='linear')(mix_output)
        mix_output = keras.layers.LayerNormalization()(mix_output)
        mix_output = keras.layers.Dropout(dropout_rate)(mix_output)
        mix_output = keras.layers.Dense(units=pred_len, activation=gelu_approximate)(mix_output) #gelu
        mix_output_list.append(mix_output)

    return keras.layers.add(mix_output_list)
