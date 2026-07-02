"""Custom Keras layers: Gradient Reversal Layer for adversarial domain adaptation.

Reference:
    Ganin, Y. & Lempitsky, V. (2015). Unsupervised Domain Adaptation by
    Backpropagation. ICML. https://arxiv.org/abs/1409.7495
"""

import tensorflow as tf


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, hp_lambda: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x, training=None):
        return self._reverse_gradient(x)

    @tf.custom_gradient
    def _reverse_gradient(self, x):
        lam = self.hp_lambda

        def grad(dy):
            return -lam * dy

        return tf.identity(x), grad

    def get_config(self):
        config = super().get_config()
        config.update({"hp_lambda": self.hp_lambda})
        return config
