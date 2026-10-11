import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN


class UnitPathIntegratorTF(tf.keras.Model):
    """
    Specification of the SorscherRNN model -
    stripping the model down to the bone in an effort to probe
    which parts of the model is necessary to do path integration.
    """

    def __init__(self, Ng=4096, Np=512, weight_decay=1e-4, activation="relu", **kwargs):
        super(UnitPathIntegratorTF, self).__init__(**kwargs)
        # define network architecture
        self.velocity_encoder = Dense(Ng, name="velocity_encoder", use_bias=False)
        
        # init_pos encoder & recurrent weights in one. Since two subsequent linear 
        # layers does not increase expressivity, i.e.: ABx = Ax.
        # This can be done since seq_len = 1, so recurrent non-linearity does not
        # happend in the first recurrent iteration (for inital pos).
        self.init_position_encoder = Dense(
            Ng,
            name="init_position_encoder",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        # Linear read-out weights
        self.decoder = Dense(Np, name="decoder", use_bias=False)
        # Non-linearity (activation function)
        self.activation_fn = tf.keras.layers.Activation(activation=activation)

    def g(self, inputs):
        v, p0 = inputs
        init_state = self.init_position_encoder(p0)

        # (unit) RNN-like
        velocity_state = self.velocity_encoder(v)
        return self.activation_fn(init_state + velocity_state)

    def call(self, inputs, softmax=False):
        place_preds = self.decoder(self.g(inputs))
        return place_preds if not softmax else tf.keras.activations.softmax(place_preds)
