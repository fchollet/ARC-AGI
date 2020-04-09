"""
Stolen shamelessly from here: https://www.kaggle.com/meaninglesslives/stacking-models-and-new-features-for-arc
It's pretty cool to see the ensemble model work so well.
"""

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              BaggingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

from .problem_fetcher import ProblemFetcher
import numpy as np
import os
import json


def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):
    # pdb.set_trace()

    if (cur_row <= 0) or (cur_col > ncols - 1):
        top = -1
    else:
        top = color[cur_row - 1][cur_col]

    if (cur_row >= nrows - 1) or (cur_col > ncols - 1):
        bottom = -1
    else:
        bottom = color[cur_row + 1][cur_col]

    if (cur_col <= 0) or (cur_row > nrows - 1):
        left = -1
    else:
        left = color[cur_row][cur_col - 1]

    if (cur_col >= ncols - 1) or (cur_row > nrows - 1):
        right = -1
    else:
        right = color[cur_row][cur_col + 1]

    return top, bottom, left, right


def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
    if cur_row == 0:
        top_left = -1
        top_right = -1
    else:
        if cur_col == 0:
            top_left = -1
        else:
            top_left = color[cur_row - 1][cur_col - 1]
        if cur_col == ncols - 1:
            top_right = -1
        else:
            top_right = color[cur_row - 1][cur_col + 1]

    return top_left, top_right


def get_vonN_neighbours(color, cur_row, cur_col, nrows, ncols):
    if cur_row == 0:
        top_left = -1
        top_right = -1
    else:
        if cur_col == 0:
            top_left = -1
        else:
            top_left = color[cur_row - 1][cur_col - 1]
        if cur_col == ncols - 1:
            top_right = -1
        else:
            top_right = color[cur_row - 1][cur_col + 1]

    if cur_row == nrows - 1:
        bottom_left = -1
        bottom_right = -1
    else:

        if cur_col == 0:
            bottom_left = -1
        else:
            bottom_left = color[cur_row + 1][cur_col - 1]
        if cur_col == ncols - 1:
            bottom_right = -1
        else:
            bottom_right = color[cur_row + 1][cur_col + 1]

    return top_left, top_right, bottom_left, bottom_right


def make_features(input_color, nfeat, local_neighb):
    nrows, ncols = input_color.shape
    feat = np.zeros((nrows * ncols, nfeat))
    cur_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            feat[cur_idx, 0] = i
            feat[cur_idx, 1] = j
            feat[cur_idx, 2] = input_color[i][j]
            feat[cur_idx, 3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
            try:
                feat[cur_idx, 7] = len(np.unique(input_color[i - 1, :]))
                feat[cur_idx, 8] = len(np.unique(input_color[:, j - 1]))
            except IndexError:
                pass

            feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
            feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
            feat[cur_idx, 11] = len(np.unique(input_color[i - local_neighb:i + local_neighb,
                                              j - local_neighb:j + local_neighb]))

            feat[cur_idx, 12:16] = get_moore_neighbours(input_color, i + 1, j, nrows, ncols)
            feat[cur_idx, 16:20] = get_moore_neighbours(input_color, i - 1, j, nrows, ncols)

            feat[cur_idx, 20:24] = get_moore_neighbours(input_color, i, j + 1, nrows, ncols)
            feat[cur_idx, 24:28] = get_moore_neighbours(input_color, i, j - 1, nrows, ncols)

            feat[cur_idx, 28] = len(np.unique(feat[cur_idx, 3:7]))
            try:
                feat[cur_idx, 29] = len(np.unique(input_color[i + 1, :]))
                feat[cur_idx, 30] = len(np.unique(input_color[:, j + 1]))
            except IndexError:
                pass
            cur_idx += 1

    return feat


def features(task, nfeat, local_neighb):
    mode = 'train'
    cur_idx = 0
    num_train_pairs = len(task[mode])
    #     total_inputs = sum([len(task[mode][i]['input'])*len(task[mode][i]['input'][0]) for i in range(num_train_pairs)])

    feat, target = [], []
    for task_num in range(num_train_pairs):
        for a in range(3):
            input_color = np.array(task[mode][task_num]['input'])
            target_color = task[mode][task_num]['output']
            if a == 1:
                input_color = np.fliplr(input_color)
                target_color = np.fliplr(target_color)
            if a == 2:
                input_color = np.flipud(input_color)
                target_color = np.flipud(target_color)

            nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

            target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])

            if (target_rows != nrows) or (target_cols != ncols):
                print('Number of input rows:', nrows, 'cols:', ncols)
                print('Number of target rows:', target_rows, 'cols:', target_cols)
                not_valid = 1
                return None, None, 1

            imsize = nrows * ncols
            offset = imsize * task_num * 3  # since we are using three types of aug
            feat.extend(make_features(input_color, nfeat, local_neighb))
            target.extend(np.array(target_color).reshape(-1, ))
            cur_idx += 1

    return np.array(feat), np.array(target), 0


def do_bullshit(task, task_name):
    nfeat = 31
    local_neighb = 5
    valid_scores = {}
    model_accuracies = {'ens': []}
    pred_taskids = []

    feat, target, not_valid = features(task, nfeat, local_neighb)
    if not_valid:
        print('ignoring task', task_name)
        print()
        not_valid = 0
        return -1

    estimators = [
        ('xgb', XGBClassifier(n_estimators=25, n_jobs=-1)),
        ('extra_trees', ExtraTreesClassifier()),
        ('bagging', BaggingClassifier()),
        ('LogisticRegression', LogisticRegression())
    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=XGBClassifier(n_estimators=10, n_jobs=-1)
    )

    clf.fit(feat, target)

    #     training on input pairs is done.
    #     test predictions begins here

    num_test_pairs = len(task['test'])
    if num_test_pairs > 1:
        print('{} num test pairs'.format(num_test_pairs))
    cur_idx = 0
    preds = []
    for task_num in range(num_test_pairs):
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])

        feat = make_features(input_color, nfeat, local_neighb)

        print('Made predictions for ', task_name)

        preds.append(clf.predict(feat).reshape(nrows, ncols).tolist())
    return preds
