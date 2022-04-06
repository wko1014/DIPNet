# Import APIs
import tensorflow as tf

# Define dual-path MSNN
class MSNN(tf.keras.Model):
    tf.keras.backend.set_floatx('float64')
    def __init__(self, F0, F1_TS, T1_TS, F2_TS, T2_TS, F3_TS, T3_TS, F1_ST, T1_ST, F2_ST, T2_ST, F3_ST, T3_ST,
                 num_channels=64, sample_freq=256): # tunable parameters optimized by Bayesian HPO
        super(MSNN, self).__init__()
        self.n_c = num_channels # number of electrodes
        self.f_s = sample_freq # samping frequency

        # Network hyperparmeter learned by Bayesian optimization
        # Spectral convolution depth
        self.F0 = F0
        # Temporal-spatio path kernels and depths
        self.F1_TS, self.T1_TS = F1_TS, T1_TS
        self.F2_TS, self.T2_TS = F2_TS, T2_TS
        self.F3_TS, self.T3_TS = F3_TS, T3_TS
        # Spatio-temporal path kernels and depths
        self.F1_ST, self.T1_ST = F1_ST, T1_ST
        self.F2_ST, self.T2_ST = F2_ST, T2_ST
        self.F3_ST, self.T3_ST = F3_ST, T3_ST

        # Regularizer
        self.regularizer = tf.keras.regularizers.L1L2(l1=.001, l2=.01)

        # Activation functions
        self.l_relu = tf.keras.layers.LeakyReLU()
        self.softmax = tf.keras.layers.Softmax()

        # Define convolutions
        conv = lambda D, kernel: tf.keras.layers.Conv2D(D, kernel, kernel_regularizer=self.regularizer)
        sepconv = lambda D, kernel: tf.keras.layers.SeparableConv2D(D, kernel, padding="same",
                                                                    depthwise_regularizer=self.regularizer,
                                                                    pointwise_regularizer=self.regularizer)

        # Spectral convolution
        self.conv0 = conv(self.F0, (1, int(self.f_s/2)))

        # Spatio-Temporal path (Path 1, EEGNet, Shallow ConvNet, Deep ConvNet-ish path)
        self.conv1s_ST = conv(self.F1_ST, (self.n_c, 1))
        self.conv1t_ST = sepconv(self.F1_ST, (1, self.T1_ST))
        self.conv2t_ST = sepconv(self.F2_ST, (1, self.T2_ST))
        self.conv3t_ST = sepconv(self.F3_ST, (1, self.T3_ST))

        # Temporal-Spatio path (Path 2, MSNN-ish path)
        self.conv1t_TS = sepconv(self.F1_TS, (1, self.T1_TS))
        self.conv1s_TS = conv(self.F1_TS, (self.n_c, 1))

        self.conv2t_TS = sepconv(self.F2_TS, (1, self.T2_TS))
        self.conv2s_TS = conv(self.F2_TS, (self.n_c, 1))

        self.conv3t_TS = sepconv(self.F3_TS, (1, self.T3_TS))
        self.conv3s_TS = conv(self.F3_TS, (self.n_c, 1))

        # Flattening
        self.flatten = tf.keras.layers.Flatten()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(.5)

        # Decision making layer
        self.dense = tf.keras.layers.Dense(5, kernel_regularizer=self.regularizer)

    def __call__(self, x):
        feature = self.embedding(x)
        y_hat = self.classifier(feature)
        return y_hat

    def embedding(self, x):
        f = self.l_relu(self.conv0(x))

        # Temporal-spatial path, multi-scale representation
        f_TS = self.l_relu(self.conv1t_TS(f))
        f1_TS = self.l_relu(self.conv1s_TS(f_TS))

        f_TS = self.l_relu(self.conv2t_TS(f_TS))
        f2_TS = self.l_relu(self.conv2s_TS(f_TS))

        f_TS = self.l_relu(self.conv3t_TS(f_TS))
        f3_TS = self.l_relu(self.conv3s_TS(f_TS))

        # Spatio-temporal path
        f_ST = self.l_relu(self.conv1s_ST(f))
        f_ST = self.l_relu(self.conv1t_ST(f_ST))
        f_ST = self.l_relu(self.conv2t_ST(f_ST))
        f_ST = self.l_relu(self.conv3t_ST(f_ST))

        # Concatenating features to exploit temporal-spatial information and spatio-temporal information
        feature = tf.concat((f1_TS, f2_TS, f3_TS, f_ST), -1)

        # Global average pooling (can be changed to adpative average pooling)
        feature = tf.reduce_mean(feature, -2)
        return feature

    def classifier(self, feature):
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.dense(feature))
        return y_hat
