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
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.cmiknn import CMIknn
import pickle
from tqdm import tqdm
import glob
import json

from Method.rcbnb_mb import RCBNBW_noCindex

regime_num = 2
max_transitions = 4
list_switch_thres = [0.05]
list_num_iterations = [20]
list_max_anneal = [200]
tau_min = 0
tau_max = 1
list_pc_alpha = [0.1]
alpha_level = None
n_jobs = -1

simplify_node_name = {
    'Real time merger bolt de not_found sur Storm-1': 'Real time merger bolt',
    'Check message bolt de not_found sur storm-1': 'Check message bolt',
    'Message dispatcher bolt de not_found sur storm-1': 'Message dispatcher bolt',
    'Metric bolt de not_found sur Storm-1': 'Metric bolt',
    'Pre-Message dispatcher bolt de not_found sur storm-1': 'Pre-Message dispatcher bolt',
    'capacity_last_metric_bolt de Apache-Storm-bolt_capacity_topology - monitoring_ingestion sur prd-ovh-storm-01': 'Last_metric_bolt',
    'capacity_elastic_search_bolt de Apache-Storm-bolt_capacity_topology - monitoring_ingestion sur prd-ovh-storm-01': 'Elastic_search_bolt',
    'Group status information bolt de not_found sur storm-1': 'Group status information bolt'
}

boolean_variables = []
param_data = pd.DataFrame()
dict_anomaly = pd.DataFrame()
directoryPath = '../../Real_Data/IT_monitoring/real_monitoring_data/'

for file_name in glob.glob(directoryPath + '*.csv'):
    if "data_with_incident_between_46683_and_46783" not in file_name:
        col_value = pd.read_csv(file_name, low_memory=False)
        with open(file_name.replace('.csv', '.json')) as json_file:
            x_descri = json.load(json_file)
        param_data[simplify_node_name[x_descri["metric_name"]]] = col_value['value']
        dict_anomaly[simplify_node_name[x_descri["metric_name"]]] = x_descri["anomalies"]

anomaly_start = 46683
anomaly_end = 46783
normal_interval = 1000
param_data = param_data.iloc[anomaly_start - normal_interval:anomaly_end + normal_interval]
data_frame = param_data.to_numpy()
# print(data_frame.shape)
data_frame = pp.DataFrame(data_frame)

num_test = 1

for i in range(num_test):
    for max_anneal in list_max_anneal:
        for num_iterations in list_num_iterations:
            for pc_alpha in list_pc_alpha:
                alpha_level = pc_alpha
                for switch_thres in list_switch_thres:
                    list_err = []
                    list_results = []

                    rcbnbw = RCBNBW_noCindex(dataframe=data_frame, cond_ind_test=RobustParCorr(),
                                             prediction_model=LinearRegression(), verbosity=1)
                    results = rcbnbw.run_rcbnbw(num_regimes=regime_num, max_transitions=max_transitions,
                                                switch_thres=switch_thres, num_iterations=num_iterations,
                                                max_anneal=max_anneal,
                                                tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,
                                                alpha_level=alpha_level, n_jobs=n_jobs)

                    list_results.append(results)

                    # For 2 regimes
                    if not os.path.exists(os.path.join('../../results', 'IT_monitorning', 'results_rcbnb')):
                        os.makedirs(os.path.join('../../results', 'IT_monitorning', 'results_rcbnb'))

                    # For 2 regimes
                    open_file = open(os.path.join('../../results', 'IT_monitorning', 'results_rcbnb', 'switch_thres_' + str(switch_thres) + '_alpha_' + str(pc_alpha) + '_ann_' + str(max_anneal) + '_niter_' +
                                     str(num_iterations) + '_res_mb.pkl'),"wb")

                    pickle.dump(list_results, open_file)
                    open_file.close()

