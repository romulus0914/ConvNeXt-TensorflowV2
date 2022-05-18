# TensorflowV2 implementation of ConvNeXt.
# Refer to `A ConvNet for the 2020s` (https://arxiv.org/pdf/2201.03545.pdf).

import tensorflow as tf

class ConvNeXtBlock(tf.keras.layers.Layer):
    """ConvNeXt block.

    dwconv -> layernorm -> 1x1 conv -> GELU -> 1x1 conv
    """

    def __init__(self, filters, layer_scale_init=1e-6):
        """Initialize."""
        super().__init__()

        self._dwconv = tf.keras.layers.DepthwiseConv2D(7, padding='same')
        self._norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._pwconv_expand = tf.keras.layers.Conv2D(filters * 4, 1)
        self._act = tf.keras.layers.Activation('relu')
        self._pwconv_project = tf.keras.layers.Conv2D(filters, 1)

        # Layer scale.
        self.gamma = self.add_weight(
            'gamma',
            shape=(filters,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(layer_scale_init),
            trainable=True
        )

        # TODO
        # Drop path.

    def call(self, inputs):
        """Call."""
        residual = inputs

        inputs = self._dwconv(inputs)
        inputs = self._norm(inputs)
        inputs = self._pwconv_expand(inputs)
        inputs = self._act(inputs)
        inputs = self._pwconv_project(inputs)
        inputs = inputs * self.gamma
        inputs = inputs + residual

        return inputs


class DownsampleLayer(tf.keras.layers.Layer):
    """Downsample layer.

    layernorm -> 2x2 conv (stride 2)
    """

    def __init__(self, filters):
        """Initialize."""
        super().__init__()
        self._norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._conv = tf.keras.layers.Conv2D(filters, 2, strides=2)

    def call(self, inputs):
        """Call."""
        return self._conv(self._norm(inputs))


class ConvNeXt(tf.keras.Model):
    """ConvNeXt model."""

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        widths=(96, 192, 384, 768),
        num_classes=1000,
        layer_scale_init=1e-6
    ):
        super().__init__()

        self._stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(widths[0], 4, strides=4),
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ]
        )

        self._stages = tf.keras.Sequential(
            [
                tf.keras.Sequential(
                    [ConvNeXtBlock(widths[0], layer_scale_init) for _ in range(depths[0])]
                ),
                DownsampleLayer(widths[1]),
                tf.keras.Sequential(
                    [ConvNeXtBlock(widths[1], layer_scale_init) for _ in range(depths[1])]
                ),
                DownsampleLayer(widths[2]),
                tf.keras.Sequential(
                    [ConvNeXtBlock(widths[2], layer_scale_init) for _ in range(depths[2])]
                ),
                DownsampleLayer(widths[3]),
                tf.keras.Sequential(
                    [ConvNeXtBlock(widths[3], layer_scale_init) for _ in range(depths[3])]
                )
            ]
        )

        self._head = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(num_classes)
            ]
        )

    def call(self, inputs, **kwargs):
        """Call."""
        inputs = self._stem(inputs)
        inputs = self._stages(inputs)
        inputs = self._head(inputs)

        return inputs
