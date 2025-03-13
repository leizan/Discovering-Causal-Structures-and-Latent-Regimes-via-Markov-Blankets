import os
import sys
sys.path.append('/home/user/code_non_stationary')
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from tigramite import data_processing as pp
from tigramite.models import LinearMediation
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
import pickle
from tqdm import tqdm

from Method.rdynotears_mb import Regimenotears

import warnings
warnings.filterwarnings("ignore")

# path_data = '../Dataset_0.3_lag_1_sta'  # '../Dataset_0.3_lag_1_sta'
dataset_size = [600, 1200]
regime_num = 2
marker = '_regimes'
experiment = 'n'
max_transitions = 3
list_switch_thres = [0.05]  # [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
list_num_iterations = [20]  #  # iteration times 20
list_max_anneal = [50]  # starting points 50
tau_min = 0
tau_max = 1
list_pc_alpha = [0.1]   # [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # pc_alpha = 0.05
alpha_level = None
n_jobs = -1
list_suffixes = ['']  # ['', '_bruit_02', '_bruit_005', '_dense', '_not_always_selfcause', '_sparse']

for suffixe in list_suffixes:
    path_data = '../../Dataset'+suffixe
    for max_anneal in list_max_anneal:
        for num_iterations in list_num_iterations:
            for pc_alpha in list_pc_alpha:
                alpha_level = pc_alpha
                for var in dataset_size:
                    for switch_thres in list_switch_thres:
                        list_err = []
                        list_results = []
                        for i in tqdm(range(50)):
                            data_frame = pd.read_csv(os.path.join(path_data, experiment, str(regime_num)+marker, str(var), 'Data_raw','data_'+str(i+1)+'.csv'), index_col=0)
                            data_frame = data_frame.to_numpy()
                            data_frame = pp.DataFrame(data_frame)


                            dynotears = Regimenotears(dataframe=data_frame,  cond_ind_test=ParCorr(), prediction_model=LinearRegression(), verbosity=1)
                            results = dynotears.run_rdynotears(num_regimes=regime_num, max_transitions=max_transitions, switch_thres=switch_thres, num_iterations=num_iterations, max_anneal=max_anneal,
                                                tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha_level, n_jobs=n_jobs)

                            list_results.append(results)


                            if not os.path.exists(os.path.join('../../results', '2regimes', 'results_rdynotears'+suffixe, str(var))):
                                os.makedirs(os.path.join('../../results', '2regimes', 'results_rdynotears'+suffixe, str(var)))


                            open_file = open(os.path.join('../../results', '2regimes', 'results_rdynotears'+suffixe, str(var), 'switch_thres_' + str(switch_thres) + '_alpha_' + str(pc_alpha)
                                                          + '_ann_' + str(max_anneal) + '_niter_' + str(num_iterations) + '_res_mb.pkl'), "wb")
                            pickle.dump(list_results, open_file)
                            open_file.close()
