import numpy as np
import pandas as pd
import sys
import os
import copy
import random

from self_tool import metrics_science

tree_high = 5

class TreeNode():

    def __init__(self, feature_idx=None, feature_val=None, node_val=None, left_child=None, right_child=None):

        self._feature_idx = feature_idx
        self._feature_val = feature_val
        self._node_val = node_val
        self._left_child = left_child
        self._right_child = right_child

class Tree(object):
    def __init__(self, min_sample=2, max_depth=tree_high, reg_lambda=0.1):

        self._root = None
        self._min_sample = min_sample
        self._max_depth = max_depth
        self.reg_lambda = reg_lambda

    def _fit(self, X, y, exp, internal_rate, shrinkage, lap1_rate, n_trees, cur_tree):

        self._root = self._build_tree(X, y, exp*internal_rate, exp*(1-internal_rate), shrinkage, lap1_rate, n_trees, cur_tree, cur_depth=0)

    def predict(self, x):

        if x.shape[0] != 1:
            sys.exit('file {} number {} raise error： Please ensure that the number of samples passed in when the '
                     'predict function is called is 1, The current number of incoming samples is:{}'
                     .format(os.path.basename(__file__), sys._getframe().f_lineno - 2, x.shape[0]))
        return self._predict(x, self._root)

    def _build_tree(self, X, y, exp_internal, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree, cur_depth):

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if cur_depth >= tree_high:
            node_val = self._calc_node_val(X, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree)
            return TreeNode(node_val=node_val)

        if np.unique(y).shape[0] == 1:
            node_val = self._calc_node_val(X, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree)
            return TreeNode(node_val=node_val)

        exponent = []

        n_sample = X.shape[0]
        n_feature = X.shape[1] - 1

        feature_list = []
        for item in (x for x in X.columns if x not in ['grad']):
            feature_list.append(item)

        if feature_list is None:
            sys.exit('file {} number {} raise error： feature_list is null'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

        if n_sample >= self._min_sample:
            for i in range(n_feature):
                feature_value = np.unique(X.iloc[:, i])
                for fea_val in feature_value:
                    X_left = X[X.iloc[:, i] <= fea_val]
                    y_left = y[X.iloc[:, i] <= fea_val]


                    X_right = X[X.iloc[:, i] > fea_val]
                    y_right = y[X.iloc[:, i] > fea_val]

                    if X_left.shape[0] > 0 and y_left.shape[0] > 0 and X_right.shape[0] > 0 and y_right.shape[0] > 0:
                        before_divide = self._calc_evaluation(X.loc[:, 'grad'].sum(), X.shape[0], self.reg_lambda)
                        after_divide_left = self._calc_evaluation(X_left.loc[:, 'grad'].sum(), X_left.shape[0], self.reg_lambda)
                        after_divide_right = self._calc_evaluation(X_right.loc[:, 'grad'].sum(), X_right.shape[0], self.reg_lambda)
                        if (after_divide_left + after_divide_right) > before_divide:
                            gain = (after_divide_left + after_divide_right - before_divide) / 2
                            exponent.append([(feature_list[i], fea_val), gain])

        k = []
        v = []
        for i in range(len(exponent)):
            k.append(exponent[i][0])
            v.append(exponent[i][1])

        if len(exponent) != 0 and len(v) != 0:

            optimal_index = metrics_science.exp(v, exp_internal / self._max_depth, 2)

            best_feature = k[optimal_index][0]
            best_feature_idx = feature_list.index(best_feature)
            best_feature_val = k[optimal_index][1]

            X_left = X[X[best_feature] <= best_feature_val]
            y_left = y[X[best_feature] <= best_feature_val]

            X_right = X[X[best_feature] > best_feature_val]
            y_right = y[X[best_feature] > best_feature_val]

            left_child = self._build_tree(X_left, y_left, exp_internal, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree, cur_depth=cur_depth+1)
            right_child = self._build_tree(X_right, y_right, exp_internal, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree, cur_depth=cur_depth+1)

            return TreeNode(feature_idx=best_feature_idx, feature_val=best_feature_val,
                            left_child=left_child, right_child=right_child)

        node_val = self._calc_node_val(X, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree)
        return TreeNode(node_val=node_val)

    def _predict(self, x, tree=None):

        if tree is None:
            tree = self._root

        if tree._node_val is not None:
            return tree._node_val

        feature_val = x.iloc[:, tree._feature_idx].values
        if feature_val <= tree._feature_val:
            return self._predict(x, tree._left_child)
        return self._predict(x, tree._right_child)

    def _calc_node_val(self, x, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree):

        for i in range(x.shape[0]):
            if abs(x.iloc[i, -1]) > np.power(1 - shrinkage, cur_tree - 1):
                if x.iloc[i, -1] > 0:
                    x.iloc[i, -1] = np.power(1 - shrinkage, cur_tree - 1)
                else:
                    x.iloc[i, -1] = -np.power(1 - shrinkage, cur_tree - 1)

        gi = x.loc[:, 'grad'].sum()
        hi = x.shape[0]
        vt = - gi / (hi + self.reg_lambda)

        sensitive = np.power(1 - shrinkage, cur_tree - 1) * 0.91
        return (vt + np.random.laplace(0, sensitive / exp_leaf)) * 0.1


    def _calc_evaluation(self, gi_sum, hi_sum, lam):
        return np.power(gi_sum, 2) / (hi_sum + lam)

class xgbRegression(Tree):
    def __init__(self, n_estimator=50, trees_in_ensemble=50, max_depth=tree_high, reg_lambda=0.1, learning_rate=0.1, min_sample=2):
        super().__init__()

        self._n_estimator = n_estimator
        self._trees_in_ensemble = trees_in_ensemble
        self.max_depth = max_depth
        self._reg_lambda = reg_lambda
        self._l_r = learning_rate
        self._min_sample = min_sample
        self._trees = []

        for _ in range(self._n_estimator):
            tree = Tree(min_sample, max_depth)
            self._trees.append(tree)

    def fit(self, X, y, exp, internal_rate, lap1_rate):

        if 'grad' in X.columns:
            sys.exit('file {} number {} raise error： The dataset has a feature named "grad"'
                     .format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

        n_samples = X.shape[0]

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)

        shrinkage = 0.1

        delete_index_list = None

        predict_value = np.zeros((n_samples, 1))


        gi =  -y_copy.values
        x_gi = pd.DataFrame(gi, columns=['grad'])
        X_G = pd.concat([X_copy, x_gi], axis=1)

        y_pre = pd.DataFrame(predict_value, columns=['predict_value'])
        y_and_pre = pd.concat([y_copy, y_pre], axis=1)
        filter_num_sum = 0
        for i in range(self._n_estimator):

            te = (i + 1) % self._trees_in_ensemble
            if te == 0:
                te = 50

            if te == 1:
                X_G = copy.deepcopy(X_G)
                y_and_pre = copy.deepcopy(y_and_pre)
            randomly_pick = int(n_samples * shrinkage * np.power(1 - shrinkage, te-1) / (1 - np.power(1 - shrinkage, self._trees_in_ensemble-1)))
            if randomly_pick > X_G.shape[0]:
                choose_samples_list = list(range(X_G.shape[0]))
                X_new = X_G
                y_new = y_and_pre
            else:
                choose_samples_list = random.sample(range(X_G.shape[0]), randomly_pick)
                X_new = pd.DataFrame(X_G, index=choose_samples_list)
                y_new = pd.DataFrame(y_and_pre, index=choose_samples_list)

            X_delete, y_delete, delete_index_list = metrics_science.del_samples(X_new, y_new, 1 * np.power(1 - shrinkage, te-1))
            used_index_list = [i for i in choose_samples_list if i not in delete_index_list]

            X_copy.drop(used_index_list, inplace=True)
            X_G.drop(used_index_list, inplace=True)
            y_and_pre.drop(used_index_list, inplace=True)
            X_copy = X_copy.reset_index(drop=True)
            X_G = X_G.reset_index(drop=True)
            y_and_pre = y_and_pre.reset_index(drop=True)
            filter_num_sum = filter_num_sum + len(delete_index_list)

            self._trees[i]._fit(X=X_delete,
                                y=y_delete,
                                exp=exp,
                                internal_rate=internal_rate,
                                shrinkage=shrinkage,
                                lap1_rate=lap1_rate,
                                n_trees=self._n_estimator,
                                cur_tree=i+1)

            for j in range(X_copy.shape[0]):
                y_and_pre.iloc[j, 1] += self._trees[i].predict(X_copy[j:j + 1])


            if i == 0:
                shrinkage_list = np.zeros((n_samples, 1))
                for s in range(y_and_pre.shape[0]):
                    if abs(y_and_pre.iloc[s, 1]) < abs(y_and_pre.iloc[s, 0]):
                        shrinkage_list[s] = abs(y_and_pre.iloc[s, 1]) / abs(y_and_pre.iloc[s, 0])
                    else:
                        shrinkage_list[s] = abs(y_and_pre.iloc[s, 0]) / abs(y_and_pre.iloc[s, 1])
                shrinkage = np.mean(shrinkage_list)



                if shrinkage > 1:
                    meet_the_requirements = []
                    for r in shrinkage_list:

                        if r < 1:
                            meet_the_requirements.append(r)
                    shrinkage = np.mean(meet_the_requirements)

            for k in range(X_G.shape[0]):
                X_G.iloc[k, -1] = y_and_pre.iloc[k, 1] - y_and_pre.iloc[k, 0]

        print('\033[0;32mAAAI privacy budget is {} train {} trees,The average number of deleted samples is {}\033[0m'.format(exp, self._n_estimator, filter_num_sum / n_samples))
        return filter_num_sum / n_samples

    def predict(self, x):
        y_pred = [0] * x.shape[0]
        for tree in self._trees:
            for i in range(x.shape[0]):
                y_pred[i] += tree.predict(x[i:i+1])
        return y_pred











