from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K

class Attention(Layer):
    """
    Attention mechanism layer which computes the weighted average of inputs based on 
    learned importance weights.

    Parameters:
    - step_dim: The dimensionality of the input sequence.
    - W_regularizer: Regularizer function for the weight matrix (optional).
    - b_regularizer: Regularizer function for the bias vector (optional).
    - W_constraint: Constraint function for the weight matrix (optional).
    - b_constraint: Constraint function for the bias vector (optional).
    - bias: Boolean, whether the layer uses a bias vector.
    """

    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, 
                 W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1)))
        eij = K.reshape(eij, (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'step_dim': self.step_dim})
        return config
