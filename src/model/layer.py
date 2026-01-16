import numpy as np
import tensorflow as tf

from tensorflow import keras


def gelu_approximate(x):
    return tf.nn.gelu(x, approximate=True)

class DecompositionLayer(keras.layers.Layer):
    """
    이동 평균을 사용하여 시계열을 추세와 계절성 성분으로 분해합니다.
    """
    def __init__(self, kernel_size, **kwargs):
        super(DecompositionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.avg = keras.layers.AvgPool1D(pool_size=kernel_size, strides=1, padding='same')

    def call(self, x):
        trend = self.avg(x)
        seasonal = x - trend
        return seasonal, trend

    # 💡 아래 메서드를 추가하여 오류를 해결합니다.
    def get_config(self):
        """레이어의 설정을 직렬화(serialize)하기 위해 호출됩니다."""
        config = super(DecompositionLayer, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class FeatureWiseScalingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.layers.Activation(gelu_approximate)
        self.scaling_vector = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.scaling_vector = self.add_weight(shape=(feature_dim,), initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        y = inputs*self.scaling_vector
        y = self.activation(y)

        return y

    def compute_output_shape(self, input_shape):
        return input_shape