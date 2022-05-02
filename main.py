# Import APIs
import time
import numpy as np
import tensorflow as tf

import experiment
from GPyOpt.methods import BayesianOptimization # for Bayesian hyperparameter optimization

# To control the GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the domain for Bayesian optimization and searching areas
domain = [{'name': 'init_lr', 'type': 'continuous', 'domain': (5e-5, 5e-3)},
          {'name': 'F0', 'type': 'discrete', 'domain': list(range(2, 9))}, # for the first convolution

          # for the spatial filtering path
          {'name': 'F1_TS', 'type': 'discrete', 'domain': list(range(8, 17))},
          {'name': 'T1_TS', 'type': 'discrete', 'domain': list(range(10, 21))},

          {'name': 'F2_TS', 'type': 'discrete', 'domain': list(range(16, 33))},
          {'name': 'T2_TS', 'type': 'discrete', 'domain': list(range(5, 11))},

          {'name': 'F3_TS', 'type': 'discrete', 'domain': list(range(32, 65))},
          {'name': 'T3_TS', 'type': 'discrete', 'domain': list(range(2, 6))},

          # for the temporal dynamics representation path
          {'name': 'F1_ST', 'type': 'discrete', 'domain': list(range(8, 17))},
          {'name': 'T1_ST', 'type': 'discrete', 'domain': list(range(10, 21))},
          {'name': 'F2_ST', 'type': 'discrete', 'domain': list(range(16, 33))},
          {'name': 'T2_ST', 'type': 'discrete', 'domain': list(range(5, 11))},
          {'name': 'F3_ST', 'type': 'discrete', 'domain': list(range(32, 65))},
          {'name': 'T3_ST', 'type': 'discrete', 'domain': list(range(2, 6))}]

for sbj_idx in range(6, 16):
    for fold_idx in range(1, 6):
        # Define objective function for the Bayesian optimization
        def objective_function(args, sbj=sbj_idx, fold=fold_idx):
            init_lr = args[0, 0]

            F0 = args[0, 1].astype(int)

            F1_TS, T1_TS = args[0, 2].astype(int), args[0, 3].astype(int)
            F2_TS, T2_TS = args[0, 4].astype(int), args[0, 5].astype(int)
            F3_TS, T3_TS = args[0, 6].astype(int), args[0, 7].astype(int)

            F1_ST, T1_ST = args[0, 8].astype(int), args[0, 9].astype(int)
            F2_ST, T2_ST = args[0, 10].astype(int), args[0, 11].astype(int)
            F3_ST, T3_ST = args[0, 12].astype(int), args[0, 13].astype(int)

            exp = experiment(sbj_idx=sbj, fold_idx=fold, init_LR=init_lr, F0=F0,
                             F1_TS=F1_TS, T1_TS=T1_TS, F2_TS=F2_TS, T2_TS=T2_TS, F3_TS=F3_TS, T3_TS=T3_TS,
                             F1_ST=F1_ST, T1_ST=T1_ST, F2_ST=F2_ST, T2_ST=T2_ST, F3_ST=F3_ST, T3_ST=T3_ST)

            loss, ACC_vl, Conf_vl, ACC_ts, Conf_ts = exp.training()

            current = int(time.time())
            print(f'Interim findings, Validation ACC: {np.max(np.array(ACC_vl)):.02f}')

            # Save results
            np.save(f'./results/SBJ{sbj_idx}_FOLD{fold_idx}_Valid_ACC_{current}.npy', np.array(ACC_vl))
            np.save(f'./results/SBJ{sbj_idx}_FOLD{fold_idx}_Valid_Conf_Matrix_{current}.npy', np.array(Conf_vl))

            return 1 - np.max(np.array(ACC_vl))  # The optimizer tries to find the minimal point.

        print(f'Start Training, Subject: {sbj_idx}')

        # Solve the problem
        Bopt = BayesianOptimization(f=objective_function, domain=domain)
        Bopt.run_optimization(max_iter=20)
