# Import APIs
import utils
import network
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix

# Define experiment class
class experiment():
    def __init__(self, sbj_idx, fold_idx, init_LR, F0,
                 F1_TS, T1_TS, F2_TS, T2_TS, F3_TS, T3_TS,
                 F1_ST, T1_ST, F2_ST, T2_ST, F3_ST, T3_ST):
        self.sbj_idx = sbj_idx
        self.fold_idx = fold_idx

        # Network hyperparameter
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

        # Learning schedules
        self.init_LR = init_LR
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.init_LR, # init_LR = 5e-4
                                                                 decay_steps=10000, decay_rate=.96, staircase=False)
        self.num_epochs = 50
        self.num_batches = 28
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def training(self):
        # print(f"Start Training, Subject {self.sbj_idx}") # just to check

        # Load dataset
        ld = utils.load_dataset(sbj_idx=self.sbj_idx)
        D_tr, D_vl, D_ts = ld.call(self.fold_idx)
        X_tr, Y_tr = D_tr
        X_vl, Y_vl = D_vl
        Y_vl = np.argmax(Y_vl, axis=-1)
        X_ts, Y_ts = D_ts
        Y_ts = np.argmax(Y_ts, axis=-1)

        # Call MSNN
        msnn = network.MSNN(F0=self.F0,
                            F1_TS=self.F1_TS, T1_TS=self.T1_TS,
                            F2_TS=self.F2_TS, T2_TS=self.T2_TS,
                            F3_TS=self.F3_TS, T3_TS=self.T3_TS,
                            F1_ST=self.F1_ST, T1_ST=self.T1_ST,
                            F2_ST=self.F2_ST, T2_ST=self.T2_ST,
                            F3_ST=self.F3_ST, T3_ST=self.T3_ST)

        # Prepare optimizer
        optimizer = self.optimizer
        num_iters = int(X_tr.shape[0]/self.num_batches)

        loss_report = []
        ACC_vl_report = []
        Conf_vl_report = []

        for epoch in range(self.num_epochs):
            loss_per_epoch = 0
            # Randomize the training dataset
            rand_idx = np.random.permutation(X_tr.shape[0])
            X_tr, Y_tr = X_tr[rand_idx, ...], Y_tr[rand_idx, ...]

            for batch in range(num_iters):
                # Sample a minibatch
                x_b = X_tr[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                y_b = Y_tr[batch * self.num_batches:(batch + 1) * self.num_batches, ...]

                # Estimating loss
                loss, grads = utils.gradient(model=msnn, x=x_b, y=y_b)

                # Update the parameters
                optimizer.apply_gradients(zip(grads, msnn.trainable_variables))
                loss_per_epoch += np.mean(loss)

            loss_per_epoch /= num_iters
            loss_report.append(loss_per_epoch)

            # Reporting
            Y_vl_hat = np.argmax(msnn(X_vl), axis=-1)
            ACC_vl = accuracy_score(y_true=Y_vl, y_pred=Y_vl_hat)
            ACC_vl_report.append(ACC_vl)
            Conf_vl = confusion_matrix(y_true=Y_vl, y_pred=Y_vl_hat)
            Conf_vl_report.append(Conf_vl)

        # To reset
        tf.keras.backend.clear_session()
        return loss_report, ACC_vl_report, Conf_vl_report
