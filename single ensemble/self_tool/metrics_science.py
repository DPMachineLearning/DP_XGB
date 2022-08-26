a = 'F:/linux_python/vision2/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
import sys
import os
import time
import datetime

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import xgb_no_privacy
import XGBoost
import AAAI_paper
import GBDT

def get_train_test_data(file_X=None, file_y=None, k=None, b=None, train_rate=0.8, back_mapping=True, binary_classification=False, kFold=False):
    X = pd.read_csv(file_X)
    y = pd.read_csv(file_y)
    X_train, X_test, y_train, y_test = None, None, None, None
    kFold_data_list = []
    if kFold is True:
        for train_index, test_index in KFold(n_splits=5).split(X):
            X_train = X.iloc[train_index, :].reset_index(drop=True)
            X_test = X.iloc[test_index, :].reset_index(drop=True)
            y_train = y.iloc[train_index, :].reset_index(drop=True)
            y_test = y.iloc[test_index, :].reset_index(drop=True)
            kFold_data_list.append([X_train, X_test, y_train, y_test])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_rate, random_state=1)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    if binary_classification is True:
        if kFold is True:
            for k_index in range(len(kFold_data_list)):
                k_y_test = kFold_data_list[k_index][3]
                for i in range(k_y_test.shape[0]):
                    if k_y_test.iloc[i, 0] > 0:
                        k_y_test.iloc[i, 0] = 1
                    else:
                        k_y_test.iloc[i, 0] = -1
                kFold_data_list[k_index][3] = k_y_test
        else:
            for i in range(y_test.shape[0]):
                if y_test.iloc[i, 0] > 0:
                    y_test.iloc[i, 0] = 1
                else:
                    y_test.iloc[i, 0] = -1
    if back_mapping is True:
        if kFold is True:
            for k_index in range(len(kFold_data_list)):
                for i in range(kFold_data_list[k_index][3].shape[0]):
                    kFold_data_list[k_index][3].loc[i, y_test.columns] = k * kFold_data_list[k_index][3].loc[i, y_test.columns] + b
        else:
            for i in range(y_test.shape[0]):
                y_test.loc[i, y_test.columns] = k * y_test.loc[i, y_test.columns] + b
    if kFold is True:
        return kFold_data_list
    else:
        return [[X_train, X_test, y_train, y_test]]

def all_process(xgboost=False, AAAI=False, gbdt=False, X_train=None, X_test=None, y_train=None, y_test=None,
                pri_bud=1, internal_rate=0.5, lap1_rate=0.5, tree=50, k=None, b=None, binary_classification=False):
    """
    训练树
    :param xgboost: Scheme of this paper
    :param AAAI: Recurring AAAI
    :param gbdt: Recurring GBDT
    :param X_train: Training set X
    :param X_test: Test set X
    :param y_train:  Training set y
    :param y_test: Test set y
    :param pri_bud: total privacy budget
    :param internal_rate: The ratio of the privacy budget of the internal nodes of each tree to the total privacy budget
    :param lap1_rate: The ratio of the privacy budget allocated to laplace1 by the leaf node to the total privacy budget
    of the current leaf node
    :param tree: How many trees have been trained in total
    :param k: Coefficient of reflection of prediction results
    :param b: Coefficient of reflection of prediction results
    :param binary_classification: Whether to perform secondary classification, regression is performed by default
    :return: Test error (secondary classification) or root mean square error (regression), sample deletion rate
    """
    model = None
    del_samples_rate = None
    if xgboost is True:
        time_xgb_pre = int(time.time())
        model = XGBoost.xgbRegression(n_estimator=tree)
        del_samples_rate = model.fit(X_train, y_train, pri_bud, internal_rate, lap1_rate)
        time_xgb_aft = int(time.time())
        print('Privacy protection xgboost one round 50% off training time (average per tree)：{}'.format((time_xgb_aft-time_xgb_pre)/tree))
    if AAAI is True:
        time_aaai_pre = int(time.time())
        model = AAAI_paper.xgbRegression(n_estimator=tree)
        del_samples_rate = model.fit(X_train, y_train, pri_bud)
        time_aaai_aft = int(time.time())
        print('Privacy protection: AAAI one round 50% off training time (average per tree)：{}'.format((time_aaai_aft - time_aaai_pre)/tree))
    if gbdt is True:
        time_gbdt_pre = int(time.time())
        model = GBDT.GBDTRegression(n_estimator=tree)
        model.fit(X_train, y_train, pri_bud)
        time_gbdt_aft = int(time.time())
        print('Privacy protection gbdt: 50% off training time per round (average per tree)：{}'.format((time_gbdt_aft - time_gbdt_pre)/tree))
    y_pred = model.predict(X_test)

    # 二分类
    if binary_classification is True:
        for i in range(len(y_pred)):
            if y_pred[i] > 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        error_rate = (1 - accuracy_score(y_pred, y_test)) * 100
        print('\033[0;32m privacy budget is  {} to test error：{}\033[0m'.format(pri_bud, error_rate))
        return error_rate, del_samples_rate

    # 回归
    for i in range(len(y_pred)):
        y_pred[i] = k * y_pred[i] + b
    rmse = mean_squared_error(y_pred, y_test) ** 0.5
    mae = mean_absolute_error(y_pred, y_test)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\033[0;32m mae of no differnetial privacy：{}\033[0m'.format(rmse))
    print('\033[0;32m rmse of no differential privacy：{}\033[0m'.format(mae))
    return rmse, del_samples_rate, mae

def no_privacy_process(file_x=None, file_y=None, tree=50, binary_classification=False):
    four_data = get_train_test_data(file_x, file_y, back_mapping=False)
    if len(four_data) != 1:
        sys.exit('file {} number {} raise error： Cross validation is not used here'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))
    X_train, X_test, y_train, y_test = four_data[0][0], four_data[0][1], four_data[0][2], four_data[0][3]
    time_no_pri_xgb_pre = int(time.time())
    xgb = xgb_no_privacy.xgbRegression(n_estimator=tree)
    xgb.fit(X_train, y_train)
    time_no_pri_xgb_aft = int(time.time())
    print('No privacy protection, xgboost, 50% off one round, average training time per tree：{}'.format((time_no_pri_xgb_aft - time_no_pri_xgb_pre)/tree))
    y_pred = xgb.predict(X_test)

    if binary_classification is True:
        for i in range(len(y_pred)):
            if y_pred[i] > 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        error_rate = (1 - accuracy_score(y_pred, y_test)) * 100
        print('\033[0;32mTest error without differential privacy：{}\033[0m'.format(error_rate))
        return error_rate

    # 回归
    rmse = mean_squared_error(y_pred, y_test) ** 0.5
    mae = mean_absolute_error(y_pred, y_test)
    print('\033[0;32m mae of no differnetial privacy：{}\033[0m'.format(rmse))
    print('\033[0;32m rmse of no differential privacy：{}\033[0m'.format(mae))
    return rmse, mae

def  save_experiment_result(result_list=None, list_0_label=None, save_file=None, first_line=False, privacy_list=None):
    result_list.insert(0, list_0_label)
    with open(save_file, 'a') as f:
        if first_line is True and privacy_list is not None:
            privacy_list.insert(0, 'all privacy budget')
            f.write('*' * 60 + str(privacy_list) + '\r\n')
            del privacy_list[0]
        f.write(str(result_list) + '\r\n')
    del result_list[0]



def visual_experiment_result(x_value, y_value_xgb, y_value_AAAI, base_line_list, y_value_GBDT=None, title=None, binary_classification=False, del_rate=False):
    xticks_index = range(len(x_value))
    plt.plot(xticks_index, y_value_xgb, 'r*-', label ='DP_xgboost')
    plt.plot(xticks_index, y_value_AAAI, 'g^-', label ='DPBoost')
    if y_value_GBDT is not None:
        plt.plot(xticks_index, y_value_GBDT, 'b>-', label='DP_gbdt')
    if len(base_line_list) == len(x_value) :
        plt.plot(xticks_index, base_line_list, 'ko-', label='xgboost')
    if del_rate is True:
        plt.xlabel('internal_privacy_rate')
    else:
        plt.xlabel('privacy budget')

    if binary_classification is True:
        plt.ylabel('test error (%)')
    elif del_rate is True:
        plt.ylabel('delete rate (%)')
    else:
        plt.ylabel('RMSE')
    plt.xticks(xticks_index, x_value)
    plt.legend()
    plt.show()

def random_pick(score_list, probability_list):

    x = np.random.uniform(0, 1)
    cumulative_probability = 0
    i = -1
    for item, item_probability in zip(score_list, probability_list):
        i += 1
        cumulative_probability += item_probability
        if x < cumulative_probability: return i
    return len(score_list) - 1

def exp(score, exp_budget, sensitive):

    exponents = []
    for i in score:
        expo = 0.5 * i * exp_budget / sensitive
        exponents.append(np.exp(expo))

    max = np.max(exponents)
    exponents = exponents / max
    sum = np.sum(exponents)

    for j in range(len(exponents)):
        exponents[j] = exponents[j] / sum
    return random_pick(score, exponents)


def del_samples(X, y_copy, threshold):
    if 'grad' not in X.columns:
        sys.exit('file {} number {} raise error： Please connect the first derivative "grad" column to x first'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

    X = copy.deepcopy(X)
    y_pre = copy.deepcopy(y_copy)


    y_columns = y_pre.columns
    y = pd.DataFrame(y_pre[y_columns[0]], columns=[y_columns[0]])

    index_list = []

    for i in X.index:
        if abs(X.loc[i, 'grad']) > threshold:
            index_list.append(i)
    X.drop(index_list, inplace=True)
    y.drop(index_list, inplace=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, index_list



def continue_to_dispersed(dataset_X):
    dataset_X = dataset_X.reset_index(drop=True)
    for feature in dataset_X.columns:
        three_to_one_list = []
        feature_value_list = list(np.unique(dataset_X[feature]))
        if len(feature_value_list) < 100:
            continue
        for index in range(0, len(feature_value_list), 3):
            if index < int(len(feature_value_list)/3) * 3:
                three_to_one_list.append((feature_value_list[index] + feature_value_list[index + 1] + feature_value_list[index + 2]) / 3)
            else:
                if len(feature_value_list) % 3 == 1:
                    three_to_one_list.append(feature_value_list[-1])
                if len(feature_value_list) % 3 == 2:
                    three_to_one_list.append((feature_value_list[-1] + feature_value_list[-2]) / 2)

        for value_index in range(dataset_X.shape[0]):
            dataset_X.loc[value_index, feature] = three_to_one_list[int(feature_value_list.index(dataset_X.loc[value_index, feature]) / 3)]

    return dataset_X

def run_experiment(core_str=None, xgb=False, AAAI=False, Gbdt=False, Base=False, save_test_result=False,
                   binary_cla=False, k=None, b=None, all_tree=50, internal_rate=0.1, kFold=False, pri_list=None):
    """

    :param core_str: dataset
    :param xgb: Whether to execute XGB experiment, default execution
    :param AAAI: Whether to repeat AAAI experiment, default not to execute
    :param Base: Whether to execute the privacy experiment without difference, default not to execute
    :param save_test_result: Whether to save the experimental results, default not to execute
    :param binary_cla: Regression or classification, default regression task
    :param k: Coefficient of regression task
    :param b: Coefficient of regression task
    :return: None
    """

    if pri_list is None:
        all_privacy = [0.5, 1, 2, 3, 4, 5]
    else:
        all_privacy = pri_list

    task_type = 'regression'
    label_y_half = ''
    if binary_cla is True:
        task_type = 'classification'

    file_X_xgb = 'F:/XGBoost_science/data/' + task_type + '/' + core_str + '/map_' + core_str + '_X' # path of feature set in data set
    file_y = 'F:/XGBoost_science/data/' + task_type + '/' + core_str + '/map_' + core_str + '_y' + label_y_half # path of label in data set
    save_file = 'F:/XGBoost_science/data/dataset_test_result/' + task_type + '/' + core_str # path of saving experiment result
    RESULT_xgb, RESULT_AAAI, RESULT_gbdt, BASE_value_list = [], [], [], []
    RESULT_xgb_mae, RESULT_AAAI_mae, RESULT_gbdt_mae, BASE_value_list_mae = [], [], [], []
    if xgb is True:
        DEL_samples_rate_xgb = []
        X_y_train_test_xgb = get_train_test_data(file_X=file_X_xgb,
                                                 file_y=file_y,
                                                 k=k, b=b,
                                                 back_mapping=not binary_cla,
                                                 binary_classification=binary_cla,
                                                 kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_xgb_temp, DEL_samples_rate_xgb_temp, RESULT_xgb_mae_temp = [], [], []
            for k_index in range(len(X_y_train_test_xgb)):
                X_train_xgb, X_test_xgb = X_y_train_test_xgb[k_index][0], X_y_train_test_xgb[k_index][1]
                y_train, y_test = X_y_train_test_xgb[k_index][2], X_y_train_test_xgb[k_index][3]
                result_xgb, del_sample_rate_xgb, result_xgb_mae = all_process(xgboost=True,
                                                                              X_train=X_train_xgb,
                                                                              X_test=X_test_xgb,
                                                                              y_train=y_train,
                                                                              y_test=y_test,
                                                                              pri_bud=privacy_budget,
                                                                              internal_rate=internal_rate,
                                                                              tree=all_tree,
                                                                              k=k, b=b,
                                                                              binary_classification=binary_cla)
                RESULT_xgb_mae_temp.append(result_xgb_mae)
                RESULT_xgb_temp.append(result_xgb)
                DEL_samples_rate_xgb_temp.append(del_sample_rate_xgb)
            RESULT_xgb.append(np.mean(RESULT_xgb_temp))
            DEL_samples_rate_xgb.append(np.mean(DEL_samples_rate_xgb_temp))
            RESULT_xgb_mae.append(np.mean(RESULT_xgb_mae_temp))
        print('\033[0;32mRESULT_xgb: {}\033[0m'.format(RESULT_xgb))
        print('\033[0;32mRESULT_mae_xgb: {}\033[0m'.format(RESULT_xgb_mae))
        print('\033[0;32mDEL_samples_rate_xgb: {}\033[0m'.format(DEL_samples_rate_xgb))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_xgb, list_0_label='xgb_RESULT', save_file=save_file,
                                   first_line=True, privacy_list=all_privacy)
            save_experiment_result(result_list=RESULT_xgb_mae, list_0_label='xgb_mae_RESULT', save_file=save_file,
                                   first_line=True, privacy_list=all_privacy)
            save_experiment_result(result_list=DEL_samples_rate_xgb, list_0_label='xgb_del_rate', save_file=save_file)

    if AAAI is True:
        file_X_AAAI = 'F:/XGBoost_science/data/' + task_type + '/' + core_str + '/' + core_str + '_X' # path of feature set in data set
        DEL_samples_rate_AAAI = []
        X_y_train_test_AAAI = get_train_test_data(file_X=file_X_AAAI,
                                                  file_y=file_y,
                                                  k=k, b=b,
                                                  back_mapping=not binary_cla,
                                                  binary_classification=binary_cla,
                                                  kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_xgb_temp, DEL_samples_rate_xgb_temp, RESULT_xgb_mae_temp = [], [], []
            for k_index in range(len(X_y_train_test_AAAI)):
                X_train_AAAI, X_test_AAAI = X_y_train_test_AAAI[k_index][0], X_y_train_test_AAAI[k_index][1]
                y_train, y_test = X_y_train_test_AAAI[k_index][2], X_y_train_test_AAAI[k_index][3]
                result_AAAI, del_sample_rate_AAAI, result_AAAI_mae = all_process(AAAI=True,
                                                                                 X_train=X_train_AAAI,
                                                                                 X_test=X_test_AAAI,
                                                                                 y_train=y_train,
                                                                                 y_test=y_test,
                                                                                 pri_bud=privacy_budget,
                                                                                 tree=all_tree,
                                                                                 k=k, b=b,
                                                                                 binary_classification=binary_cla)
                RESULT_xgb_temp.append(result_AAAI)
                RESULT_xgb_mae_temp.append(result_AAAI_mae)
                DEL_samples_rate_xgb_temp.append(del_sample_rate_AAAI)
            RESULT_AAAI.append(np.mean(RESULT_xgb_temp))
            RESULT_AAAI_mae.append(np.mean(RESULT_xgb_mae_temp))
            DEL_samples_rate_AAAI.append(np.mean(DEL_samples_rate_xgb_temp))
        print('\033[0;32mRESULT_AAAI: {}\033[0m'.format(RESULT_AAAI))
        print('\033[0;32mRESULT_AAAI_mae: {}\033[0m'.format(RESULT_AAAI_mae))
        print('\033[0;32mDEL_samples_rate_AAAI: {}\033[0m'.format(DEL_samples_rate_AAAI))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_AAAI, list_0_label='AAAI_RESULT', save_file=save_file)
            save_experiment_result(result_list=RESULT_AAAI_mae, list_0_label='AAAI_RESULT_mae', save_file=save_file)
            save_experiment_result(result_list=DEL_samples_rate_AAAI, list_0_label='AAAI_del_rate', save_file=save_file)

    if Gbdt is True:
        X_y_train_test_gbdt = get_train_test_data(file_X=file_X_xgb,
                                                 file_y=file_y,
                                                 k=k, b=b,
                                                 back_mapping=not binary_cla,
                                                 binary_classification=binary_cla,
                                                 kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_gbdt_temp, RESULT_gbdt_mae_temp = [], []
            for k_index in range(len(X_y_train_test_gbdt)):
                X_train_gbdt, X_test_gbdt = X_y_train_test_gbdt[k_index][0], X_y_train_test_gbdt[k_index][1]
                y_train, y_test = X_y_train_test_gbdt[k_index][2], X_y_train_test_gbdt[k_index][3]
                result_gbdt, del_rate_temp_, result_gbdt_mae = all_process(gbdt=True,
                                                                           X_train=X_train_gbdt,
                                                                           X_test=X_test_gbdt,
                                                                           y_train=y_train,
                                                                           y_test=y_test,
                                                                           pri_bud=privacy_budget,
                                                                           tree=all_tree,
                                                                           k=k, b=b,
                                                                           binary_classification=binary_cla)
                RESULT_gbdt_temp.append(result_gbdt)
                RESULT_gbdt_mae_temp.append(result_gbdt_mae)
            RESULT_gbdt.append(np.mean(RESULT_gbdt_temp))
            RESULT_gbdt_mae.append(np.mean(RESULT_gbdt_mae_temp))
        print('\033[0;32mRESULT__gbdt: {}\033[0m'.format(RESULT_gbdt))
        print('\033[0;32mRESULT__gbdt_mae: {}\033[0m'.format(RESULT_gbdt_mae))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_gbdt, list_0_label='gbdt_RESULT', save_file=save_file)
            save_experiment_result(result_list=RESULT_gbdt_mae, list_0_label='gbdt_RESULT_mae', save_file=save_file)

    if Base is True:
        y_map = ''
        if binary_cla is True:
            y_map = 'map_'
        file_no_map_y = 'F:/XGBoost_science/data/' + task_type + '/' + core_str + '/' + y_map + core_str + '_y'
        result, result_mae = no_privacy_process(file_x=file_X_xgb,
                                                file_y=file_no_map_y,
                                                tree=all_tree,
                                                binary_classification=binary_cla)
        BASE_value_list.append(result)
        BASE_value_list_mae.append(result_mae)
        if save_test_result is True:
            save_experiment_result(result_list=BASE_value_list, list_0_label='no_privacy', save_file=save_file)
            save_experiment_result(result_list=BASE_value_list_mae, list_0_label='no_privacy', save_file=save_file)











