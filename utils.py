# Import APIs
import scipy.io # To load .mat files (data load)
import numpy as np
import tensorflow as tf

from mne.filter import filter_data, notch_filter

# Define load imagined speech EEG dataset class
class load_dataset():
    def __init__(self, sbj_idx):
        self.sbj_idx = sbj_idx
        self.path = '/Define/Your/Own/Path' # Define the data path

    def load_data(self):
        data_tr = scipy.io.loadmat(self.path + f'Trainingset/Data_Sample{self.sbj_idx:02d}.mat')
        data_vl = scipy.io.loadmat(self.path + f'Validationset/Data_Sample{self.sbj_idx:02d}.mat')

        X_tr, Y_tr = np.array(data_tr['epo_train']['x'])[0, 0], np.array(data_tr['epo_train']['y'])[0, 0]
        X_vl, Y_vl = np.array(data_vl['epo_validation']['x'])[0, 0], np.array(data_vl['epo_validation']['y'])[0, 0]

        del data_tr, data_vl

        # For 5-fold cross-validation
        X = np.concatenate((X_tr, X_vl), axis=-1)
        Y = np.concatenate((Y_tr, Y_vl), axis=-1)

        # Reorder axis
        X = np.moveaxis(np.moveaxis(X, -1, 0), -1, 1)
        Y = np.moveaxis(np.moveaxis(Y, -1, 0), -1, 1)
        return X, Y

    def preprocessing(self, data):
        # BPF (30~125Hz, gamma range) using a 4th order Butterworth filter
        # data = filter_data(data, sfreq=256, l_freq=30, h_freq=125, verbose=False)

        # Remove 60Hz line noise with the 120Hz harmonic
        # data = notch_filter(data, Fs=256, freqs=np.arange(60, 121, 60), verbose=False)

        # Remove dummy signals after imagined speech performing
        data = data[..., :640]

        # Reject baselines
        data = data[..., 128:]

        # Remove the first and the last 0.5 sec
        data = data[..., 128:384]
        return data

    def call(self, fold):
        X, Y = self.load_data()
        X = self.preprocessing(X)

        num_samples = int(X.shape[0]/(5)) # Samples per fold

        # Set training/validation/testing data indices
        rand_idx = np.random.RandomState(seed=951014).permutation(X.shape[0])
        test_idx = rand_idx[(fold - 1) * num_samples:fold * num_samples]
        train_idx = np.setdiff1d(rand_idx, test_idx)
        valid_idx = np.random.RandomState(seed=5930).permutation(train_idx.shape[0])[:28]
        valid_idx = train_idx[valid_idx]
        train_idx = np.setdiff1d(train_idx, valid_idx)

        X = np.expand_dims(X, axis=-1) # (350, 64, 256, 1)

        X_tr, X_vl, X_ts = X[train_idx, ...], X[valid_idx, ...], X[test_idx, ...]
        Y_tr, Y_vl, Y_ts = Y[train_idx, ...], Y[valid_idx, ...], Y[test_idx, ...]

        return (X_tr, Y_tr), (X_vl, Y_vl), (X_ts, Y_ts)

# Define neural network training function
def gradient(model, x, y, smoothing=.1):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=y_hat, label_smoothing=smoothing)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad
