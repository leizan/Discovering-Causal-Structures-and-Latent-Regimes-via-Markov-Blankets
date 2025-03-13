import random

import networkx as nx
import pandas as pd
import numpy as np

from itertools import combinations


from sklearn.linear_model import LinearRegression as lr

from tigramite.pcmci import PCMCI
# from pcmci_with_bk import PCMCI as PCMCIbk
# from tigramite.independence_tests import ParCorr
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp
from tigramite.pcmci_base import PCMCIbase

from subprocess import Popen, PIPE
import os
import glob

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from Method.lingam_master.lingam.var_lingam import VARLiNGAM

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def clear_args(dir_path):
    files = glob.glob(dir_path + '/args/*')
    for f in files:
        os.remove(f)


def clear_results(dir_path):
    files = glob.glob(dir_path + '/results/*')
    for f in files:
        os.remove(f)


def process_data(data, nlags):
    nodes_to_temporal_nodes = dict()
    temporal_nodes = []
    for node in data.columns:
        nodes_to_temporal_nodes[node] = []
        for gamma in range(nlags + 1):
            if gamma == 0:
                temporal_node = str(node) + "_t"
                nodes_to_temporal_nodes[node].append(temporal_node)
                temporal_nodes.append(temporal_node)
            else:
                temporal_node = str(node) + "_t_" + str(gamma)
                nodes_to_temporal_nodes[node].append(temporal_node)
                temporal_nodes.append(temporal_node)

    new_data = pd.DataFrame()
    for gamma in range(0, nlags + 1):
        shifteddata = data.shift(periods=-nlags + gamma)

        new_columns = []
        for node in data.columns:
            new_columns.append(nodes_to_temporal_nodes[node][gamma])
        shifteddata.columns = new_columns
        new_data = pd.concat([new_data, shifteddata], axis=1, join="outer")
    new_data.dropna(axis=0, inplace=True)
    return new_data, nodes_to_temporal_nodes, temporal_nodes


def run_varlingam(data, tau_max):
    # temporal_data, col_to_temporal_col, _ = process_data(data, nlags)
    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=False)
    model.fit(data)
    order = model.causal_order_

    order = [data.columns[i] for i in order]
    order.reverse()

    order_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns, index=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    # order_matrix[col_j].loc[col_i] = 2
                    # order_matrix[col_i].loc[col_j] = 1
                    order_matrix[col_j][col_i] = 2
                    order_matrix[col_i][col_j] = 1
                # print('order_matrix')
                # print(order_matrix)
                # print('order_matrix[0][1]')
                # print(order_matrix[3][4])
                # print('order_matrix[1][0]')
                # print(order_matrix[4][3])
                # exit(0)
                # if order_matrix[col_j][col_i] != order_matrix[col_j].loc[col_i]:
                #     print('error')
                #     exit(0)
                # else:
                #     print('continue')
    return order_matrix


def get_dependence_and_significance(x, e, indtest="linear"):
    e = e.reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    dim_x = x.shape[1]
    dim_e = e.shape[1]
    a = np.concatenate((x, e), axis=1)
    xe = np.array([0] * dim_x + [1] * dim_e)

    if indtest == "linear":
        test = ParCorr(significance='analytic')
        statval = test.get_dependence_measure(a, xe)
        # pval = test.get_shuffle_significance(a, xe, statval)
        pval = test.get_analytic_significance(value=statval, T=a.shape[0], dim=a.shape[1], xyz=xe)
    else:
        pval, statval = 0, 0
        exit(0)
    return pval, statval


def get_prediction(X, y, model="linear"):
    if model == "linear":
        reg = lr().fit(X, y)
        yhat = reg.predict(X)
    else:
        yhat = 0
        exit(0)
    return yhat


def run_timino2(list_targets, list_parents, data, nlags, model="linear", indtest="linear"):
    sub_temporal_data, col_to_temporal_col, temporal_nodes = process_data(data[list_targets + list_parents], nlags)

    list_temporal_target = []
    for node in list_targets:
        list_temporal_target.append(col_to_temporal_col[node][0])
    list_temporal_parents = list(set(temporal_nodes) - set(list_temporal_target))

    order = []
    list_targets_saved = list_targets.copy()
    # list_parents = list(set(list(data.columns)) - set(list_targets))
    while len(list_temporal_target) > 1:
        list_pval = []
        list_statval = []
        temporal_cols = list_temporal_target.copy()
        temporal_cols = temporal_cols + list_temporal_parents
        for temporal_col_i in list_temporal_target:
            temporal_data_temp = sub_temporal_data[temporal_cols].copy()
            X = temporal_data_temp.drop(temporal_col_i, inplace=False, axis=1).values
            y = temporal_data_temp[temporal_col_i].values
            yhat = get_prediction(X, y, model=model)
            err = y - yhat
            pval, statval = get_dependence_and_significance(X, err, indtest=indtest)
            list_pval.append(pval)
            list_statval.append(statval)
        if len(set(list_pval)) == 1:
            tmp = min(list_statval)
            index = list_statval.index(tmp)
        else:
            tmp = max(list_pval)
            index = list_pval.index(tmp)
        temporal_col_index = list_temporal_target[index]
        col_index = list_targets[index]
        list_temporal_target.remove(temporal_col_index)
        order.append(col_index)
        list_targets.remove(col_index)
    order.append(list_targets[0])

    order_matrix = pd.DataFrame(np.zeros([len(list_targets_saved), len(list_targets_saved)]), columns=list_targets_saved, index=list_targets_saved, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix[col_j].loc[col_i] = 2
                    order_matrix[col_i].loc[col_j] = 1
    return order_matrix


def run_timino(list_targets, list_parents, data, nlags, model="linear", indtest="linear"):
    order = []
    list_targets_saved = list_targets.copy()
    # list_parents = list(set(list(data.columns)) - set(list_targets))
    while len(list_targets) > 1:
        list_pval = []
        list_statval = []
        temporal_cols = list_targets.copy()
        temporal_cols = temporal_cols + list_parents
        for temporal_col_i in list_targets:
            temporal_data_temp = data[temporal_cols].copy()
            X = temporal_data_temp.drop(temporal_col_i, inplace=False, axis=1).values
            y = temporal_data_temp[temporal_col_i].values
            yhat = get_prediction(X, y, model=model)
            err = y - yhat
            pval, statval = get_dependence_and_significance(X, err, indtest=indtest)
            list_pval.append(pval)
            list_statval.append(statval)
        if len(set(list_pval)) == 1:
            tmp = min(list_statval)
            index = list_statval.index(tmp)
        else:
            tmp = max(list_pval)
            index = list_pval.index(tmp)
        col_index = list_targets[index]
        order.append(col_index)
        list_targets.remove(col_index)
    order.append(list_targets[0])

    order_matrix = pd.DataFrame(np.zeros([len(list_targets_saved), len(list_targets_saved)]), columns=list_targets_saved, index=list_targets_saved, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    # order_matrix[col_j].loc[col_i] = 2
                    # order_matrix[col_i].loc[col_j] = 1
                    order_matrix[col_j][col_i] = 2
                    order_matrix[col_i][col_j] = 1

    return order_matrix


def run_timino_from_r(arg_list):
    # Remove all arguments from directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script = dir_path + "/timino.R"
    clear_args(dir_path)
    clear_results(dir_path)
    r_arg_list = []
    # COMMAND WITH ARGUMENTS
    for a in arg_list:
        if isinstance(a[0], pd.DataFrame):
            a[0].to_csv(dir_path + "/args/"+a[1]+".csv", index=False)
            r_arg_list.append(dir_path + "/args/" + a[1] + ".csv")
        if isinstance(a[0], int):
            f = open(dir_path + "/args/"+a[1]+".txt", "w")
            f.write(str(a[0]))
            f.close()
            r_arg_list.append(dir_path + "/args/" + a[1] + ".txt")
        if isinstance(a[0], float):
            f = open(dir_path + "/args/"+a[1]+".txt", "w")
            f.write(str(a[0]))
            f.close()
            r_arg_list.append(dir_path + "/args/" + a[1] + ".txt")

    r_arg_list.append(dir_path)
    cmd = ["Rscript", script] + r_arg_list

    p = Popen(cmd, cwd="./", stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # Return R output or error
    output, error = p.communicate()
    # print(output)
    if p.returncode == 0:
        print('R Done')
        g_df = pd.read_csv(dir_path + "/results/result.csv", header=0, index_col=0)
        # print(g_df)
        # g_df = g_df.transpose()
        return g_df
    else:
        print('R Error:\n {0}'.format(error))
        exit(0)


class CBNBw:
    def __init__(self, data, tau_min, tau_max, sig_level, model="linear",  indtest="linear", cond_indtest="linear"):
        """
        :param extra_background_knowledge_list:
        """
        self.data = data
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.model =model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.causal_order = []
        self.graph = []

        self.forbidden_orientation = []

        self.window_causal_graph_dict = dict()
        self.window_causal_graph = None
        list_nodes = []
        for col in data.columns:
            self.window_causal_graph_dict[col] = []
            list_nodes.append(GraphNode(col))
        self.causal_graph = GeneralGraph(list_nodes)

        self.constraint_result = None
        self.parent_set_generator = None

    def constraint_based(self, c_index):
        data = self.data.values
        T, N = data.shape
        indexs_var = np.arange(N)
        tv_var = {}
        indep_test = ParCorr(significance='analytic')

        if self.tau_min == 0:
            for i in range(N):
                tv_var[i] = True

            for i in range(N):
                depth = 0
                set_z = np.delete(indexs_var, np.where(indexs_var == i))
                independent = False
                while depth <= len(set_z):
                    for S in combinations(set_z, depth):
                        if len(S) == 0:
                            _, pval = indep_test.run_test_raw(x=data[:,i].reshape(-1,1), y =c_index)
                        else:
                            _, pval = indep_test.run_test_raw(x=data[:,i].reshape(-1,1), y =c_index, z=data[:,S])
                        if pval > self.sig_level:
                            tv_var[i]=False
                            independent=True
                            break
                    if independent is True:
                        break
                    depth+=1

            link_assumptions = {}
            link_assumptions[N] = {}
            for i in tv_var.keys():
                link_assumptions[i] = {}
                if tv_var[i] is True:
                    link_assumptions[i][(N, 0)] = '-->'
                    link_assumptions[N][(i, 0)] = '<--'

            link_assumptions = PCMCIbase.build_link_assumptions(link_assumptions_absent_link_means_no_knowledge=link_assumptions, n_component_time_series=N+1, tau_max=self.tau_max)

            #remove non-existance instantaneous adjacencies from c_index
            for i in tv_var.keys():
                if tv_var[i] is False:
                    link_assumptions[i].pop((N, 0))
                    link_assumptions[N].pop((i, 0))

            #remove autodependency of c_index and lagged cross-dependency
            for m in range(1, self.tau_max+1):
                link_assumptions[N].pop((N, -m))
                for i in indexs_var:
                    link_assumptions[N].pop((i, -m))
                    link_assumptions[i].pop((N, -m))

            data_with_c = np.append(data, c_index, axis=1)
            data = pp.DataFrame(data_with_c)
        else:
            link_assumptions = None
            data = pp.DataFrame(data)


        pcmci = PCMCI(dataframe=data, cond_ind_test=indep_test)
        output = pcmci.run_pcmciplus(tau_min=self.tau_min, tau_max=self.tau_max, pc_alpha=self.sig_level, link_assumptions=link_assumptions)
        self.window_causal_graph = output["graph"]
        if self.tau_min == 0:
            self.data = pd.DataFrame(data_with_c)
        self.constraint_result = output
        self.parent_set_generator = pcmci


    def find_cycle_groups(self, ):
        instantaneous_nodes = []
        instantaneous_graph = nx.Graph()
        # the last node is always the parent of other nodes
        for i in range(len(self.data.columns)-1):
            for j in range(len(self.data.columns)-1):
                t = 0
                if (self.window_causal_graph[i, j, t] == "o-o") or (self.window_causal_graph[i, j, t] == "x-x") or \
                        (self.window_causal_graph[i, j, t] == "-->") or (self.window_causal_graph[i, j, t] == "<--"):
                    instantaneous_graph.add_edge(self.data.columns[i], self.data.columns[j])
                    if self.data.columns[i] not in instantaneous_nodes:
                        instantaneous_nodes.append(self.data.columns[i])
        list_cycles = nx.cycle_basis(instantaneous_graph)


        # create cycle groups
        cycle_groups = dict()
        idx = 0
        for i in range(len(list_cycles)):
            l1 = list_cycles[i]
            test_inclusion = True
            for k in cycle_groups.keys():
                for e1 in l1:
                    if e1 not in cycle_groups[k]:
                        test_inclusion = False
            if (not test_inclusion) or (len(cycle_groups.keys()) == 0):
                cycle_groups[idx] = l1
                idx = idx + 1
                for j in range(i + 1, len(list_cycles)):
                    l2 = list_cycles[j]
                    if l1 != l2:
                        if len(list(set(cycle_groups[idx - 1]).intersection(l2))) >= 2:
                            cycle_groups[idx - 1] = cycle_groups[idx - 1] + list(set(l2) - set(cycle_groups[idx - 1]))

        # adding edges that do not belong to any cycles
        for edge in instantaneous_graph.edges:
            if len(list_cycles) > 0:
                for cycle in list_cycles:
                    if (edge[0] not in cycle) or (edge[1] not in cycle):
                        if list(edge) not in list_cycles:
                            list_cycles.append(list(edge))
                            cycle_groups[idx] = list(edge)
                            idx = idx + 1
            else:
                list_cycles.append(list(edge))
                cycle_groups[idx] = list(edge)
                idx = idx + 1
        return cycle_groups, list_cycles, instantaneous_nodes


    def noise_based(self):
        cycle_groups, list_cycles, instantaneous_nodes = self.find_cycle_groups()
        # print(instantaneous_nodes)

        list_columns = list(self.data.columns)
        if len(instantaneous_nodes)>1:
            for idx in cycle_groups.keys():
                instantaneous_nodes = cycle_groups[idx]

                parents_nodes = list(set(list_columns) - set(instantaneous_nodes))
                parents_nodes_temp = parents_nodes.copy()
                for node in instantaneous_nodes:
                    j = list_columns.index(node)
                    for parent_node in parents_nodes_temp:
                        if parent_node in parents_nodes:
                            test_parent = True
                            i = list_columns.index(parent_node)
                            for t in range(1, self.tau_max + 1):
                                if self.window_causal_graph[i, j, t] == "-->":
                                    test_parent = False
                            if test_parent:
                                parents_nodes.remove(parent_node)

                sub_data = self.data[instantaneous_nodes + parents_nodes]
                causal_order = run_varlingam(sub_data, self.tau_max)
                # print(causal_order)
                # causal_order = run_timino2(instantaneous_nodes, parents_nodes, sub_data, self.tau_max)

                for col_i in instantaneous_nodes:
                    for col_j in instantaneous_nodes:
                        if (causal_order[col_j].loc[col_i] == 2) and (causal_order[col_i].loc[col_j] == 1):
                            i = list_columns.index(col_i)
                            j = list_columns.index(col_j)
                            t = 0
                            if (self.window_causal_graph[i, j, t] == "o-o") or (
                                    self.window_causal_graph[i, j, t] == "x-x") or \
                                    (self.window_causal_graph[i, j, t] == "-->") or (self.window_causal_graph[i, j, t] == "<--"):
                                self.window_causal_graph[i, j, t] = "-->"
                                self.window_causal_graph[j, i, t] = "<--"
                                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", i, j, t)

    def run(self, c_index):
        # print("######## Running Constraint-based ########")
        self.constraint_based(c_index)
        # print("######## Running Noise-based ########")
        self.noise_based()
        parents_set = self.parent_set_generator.return_parents_dict(graph=self.window_causal_graph, val_matrix=self.constraint_result['val_matrix'], include_lagzero_parents=True)
        return self.constraint_result, self.window_causal_graph, parents_set


class CBNBw_noCindex:
    def __init__(self, data, tau_min, tau_max, sig_level, model="linear",  indtest="linear", cond_indtest="linear"):
        """
        :param extra_background_knowledge_list:
        """
        self.data = data
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.model =model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.causal_order = []
        self.graph = []

        self.forbidden_orientation = []

        self.window_causal_graph_dict = dict()
        self.window_causal_graph = None
        list_nodes = []
        for col in data.columns:
            self.window_causal_graph_dict[col] = []
            list_nodes.append(GraphNode(col))
        self.causal_graph = GeneralGraph(list_nodes)

        self.constraint_result = None
        self.parent_set_generator = None

    def constraint_based(self):
        data = self.data.values
        indep_test = ParCorr(significance='analytic')

        data = pp.DataFrame(data)

        pcmci = PCMCI(dataframe=data, cond_ind_test=indep_test)
        output = pcmci.run_pcmciplus(tau_min=self.tau_min, tau_max=self.tau_max, pc_alpha=self.sig_level)
        self.window_causal_graph = output["graph"]
        self.constraint_result = output
        self.parent_set_generator = pcmci


    def find_cycle_groups(self, ):
        instantaneous_nodes = []
        instantaneous_graph = nx.Graph()
        for i in range(len(self.data.columns)):
            for j in range(len(self.data.columns)):
                t = 0
                if (self.window_causal_graph[i, j, t] == "o-o") or (self.window_causal_graph[i, j, t] == "x-x") or \
                        (self.window_causal_graph[i, j, t] == "-->") or (self.window_causal_graph[i, j, t] == "<--"):
                    instantaneous_graph.add_edge(self.data.columns[i], self.data.columns[j])
                    if self.data.columns[i] not in instantaneous_nodes:
                        instantaneous_nodes.append(self.data.columns[i])
        list_cycles = nx.cycle_basis(instantaneous_graph)


        # create cycle groups
        cycle_groups = dict()
        idx = 0
        for i in range(len(list_cycles)):
            l1 = list_cycles[i]
            test_inclusion = True
            for k in cycle_groups.keys():
                for e1 in l1:
                    if e1 not in cycle_groups[k]:
                        test_inclusion = False
            if (not test_inclusion) or (len(cycle_groups.keys()) == 0):
                cycle_groups[idx] = l1
                idx = idx + 1
                for j in range(i + 1, len(list_cycles)):
                    l2 = list_cycles[j]
                    if l1 != l2:
                        if len(list(set(cycle_groups[idx - 1]).intersection(l2))) >= 2:
                            cycle_groups[idx - 1] = cycle_groups[idx - 1] + list(set(l2) - set(cycle_groups[idx - 1]))

        # adding edges that do not belong to any cycles
        for edge in instantaneous_graph.edges:
            # test_inclusion = list()
            if len(list_cycles) > 0:
                for cycle in list_cycles:
                    # if (edge[0] not in cycle) or (edge[1] not in cycle):
                    if (edge[0] in cycle) and (edge[1] in cycle):
                        # test_inclusion.append(0)
                        if list(edge) not in list_cycles:
                            list_cycles.append(list(edge))
                            cycle_groups[idx] = list(edge)
                            idx = idx + 1

                # if sum(test_inclusion) != 0:
                #     # if list(edge) not in list_cycles:
                #     list_cycles.append(list(edge))
                #     cycle_groups[idx] = list(edge)
                #     idx = idx + 1
            else:
                list_cycles.append(list(edge))
                cycle_groups[idx] = list(edge)
                idx = idx + 1

        # Temporal solution !!!
        cycle_groups = {0:list(range(len(self.data.columns)))}
        # print('cycle_groups')
        # print(cycle_groups)
        return cycle_groups, list_cycles, instantaneous_nodes


    def noise_based(self):
        cycle_groups, list_cycles, instantaneous_nodes = self.find_cycle_groups()
        # print(instantaneous_nodes)

        list_columns = list(self.data.columns)
        if len(instantaneous_nodes)>1:
            for idx in cycle_groups.keys():
                instantaneous_nodes = cycle_groups[idx]

                parents_nodes = list(set(list_columns) - set(instantaneous_nodes))
                parents_nodes_temp = parents_nodes.copy()
                for node in instantaneous_nodes:
                    j = list_columns.index(node)
                    for parent_node in parents_nodes_temp:
                        if parent_node in parents_nodes:
                            test_parent = True
                            i = list_columns.index(parent_node)
                            for t in range(1, self.tau_max + 1):
                                if self.window_causal_graph[i, j, t] == "-->":
                                    test_parent = False
                            if test_parent:
                                parents_nodes.remove(parent_node)

                sub_data = self.data[instantaneous_nodes + parents_nodes]
                causal_order = run_varlingam(sub_data, self.tau_max)
                # print(causal_order)
                # causal_order = run_timino2(instantaneous_nodes, parents_nodes, sub_data, self.tau_max)

                for col_i in instantaneous_nodes:
                    for col_j in instantaneous_nodes:
                        if (causal_order[col_j].loc[col_i] == 2) and (causal_order[col_i].loc[col_j] == 1):
                            i = list_columns.index(col_i)
                            j = list_columns.index(col_j)
                            t = 0
                            if (self.window_causal_graph[i, j, t] == "o-o") or (
                                    self.window_causal_graph[i, j, t] == "x-x") or \
                                    (self.window_causal_graph[i, j, t] == "-->") or (self.window_causal_graph[i, j, t] == "<--"):
                                self.window_causal_graph[i, j, t] = "-->"
                                self.window_causal_graph[j, i, t] = "<--"
                                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", i, j, t)

    def run(self):
        # print("######## Running Constraint-based ########")
        self.constraint_based()
        # print("######## Running Noise-based ########")
        self.noise_based()
        parents_set = self.parent_set_generator.return_parents_dict(graph=self.window_causal_graph, val_matrix=self.constraint_result['val_matrix'], include_lagzero_parents=True)
        return self.constraint_result, self.window_causal_graph, parents_set

def uniform_with_gap(min_value=-1, max_value=1, min_gap=-0.5, max_gap=0.5):
    while True:
        r = random.uniform(min_value, max_value)
        if min_gap>r or max_gap<r:
            break
    return r



if __name__ == '__main__':

    # res = run_timino(param_data, 4)
    # print(res)
    # nbcb = CBNBw(param_data, 4, 0.05)
    # nbcb.run()
    # print(nbcb.causal_order)
    # print(nbcb.window_causal_graph_dict)

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(0, 3)
    g.add_edge(0, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(0, 5)
    g.add_edge(0, 6)
    g.add_edge(6, 7)
    g.add_edge(5, 7)

    list_cycles = nx.cycle_basis(g)
    print(list_cycles)





