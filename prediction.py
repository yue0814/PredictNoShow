"""
offline 5-fold cv simple tree model

LightGBM, XGBoost, CatBoost

"""
import pandas as pd
import numpy as np
from collections import OrderedDict
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import  KFold # GridSearchCV, cross_val_score
from sklearn.metrics import log_loss, roc_auc_score
import _data
import catboost
import warnings
warnings.filterwarnings("ignore")

# pre-process and extraction
X, public, private = _data.load_data()

# before fitting in models
id_public = public.pop("ID")
id_private = private.pop("ID")
X.pop("ID")
status = X.pop("Status")
y = np.array([1 if i == "No-Show" else 0 for i in status])

# add features from DNN hidden layer
X = pd.concat([X, pd.DataFrame(A1)], axis=1)

# Grid Search
# params = {"n_estimators": [10, 20, 30], "max_depth":[3, 5, 7, 9, 11],\
#           "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],\
#           "min_child_weight": [0.6, 0.7, 0.8], "scale_pos_weight": [0.8, 0.9, 1]}
# xg_clf = xgb.XGBClassifier(eval_metric="logloss", objective="binary:logistic", seed=7)
# xg_grid = GridSearchCV(xg_clf, params, cv=5)
# xg_grid.fit(X, y)

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
preds = np.zeros((len(public), 5))
for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    model1 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bytree=1, eval_metric='logloss', gamma=0,
            learning_rate=0.2, max_delta_step=0, max_depth=5,
            min_child_weight=0.6, missing=None, n_estimators=30, n_jobs=1,
            nthread=None, objective='binary:logistic', random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=7, silent=True,
            subsample=1)
    model1.fit(X.iloc[train_idx, :], y[train_idx])
    print("Train Log loss is %.4f" % log_loss(y[train_idx], model1.predict_proba(X.iloc[train_idx, :])[:, 1]))
    print("Validation Log loss is %.4f" % log_loss(y[val_idx], model1.predict_proba(X.iloc[val_idx, :])[:, 1]))
    preds[:, i] = model1.predict_proba(public)[:, 1]

# ================= CatBoost ===================== #

cat_feature_inds = []
for i, c in enumerate(X.columns.values):
    num_uniques = len(X[c].unique())
    if num_uniques < 3:
        cat_feature_inds.append(i)

preds = np.zeros((len(public), 5))
for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    cat_model = catboost.CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=1,
        eval_metric='Logloss',
        random_seed=4 * 100 + 6)

    cat_model.fit(X.iloc[train_idx,:], y[train_idx], cat_features=cat_feature_inds)
    print("Train Log loss is %.4f" % log_loss(y[train_idx], cat_model.predict_proba(X.iloc[train_idx, :])[:, 1]))
    print("Validation Log loss is %.4f" % log_loss(y[val_idx], cat_model.predict_proba(X.iloc[val_idx, :])[:, 1]))
    preds[:, i] = cat_model.predict_proba(public)[:, 1]

# lightgbm
params = {'num_leaves': 31, 'application': 'binary',
        "objective": "binary",
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 20,
        "metric": "binary_logloss",
        "learning_rate": 0.003,
        "is_unbalance": "false",
        "max_depth": 20,
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        }
for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    ltrain = lgb.Dataset(X.iloc[train_idx, :], label=y[train_idx])
    lval = lgb.Dataset(X.iloc[val_idx, :], label=y[val_idx])
    
    model3 = lgb.train(params, ltrain, num_boost_round=3000, valid_sets=lval, early_stopping_rounds=500)
    print("Train Log loss is %.4f" % log_loss(y[train_idx], model3.predict(X.iloc[train_idx, :])))
    print("Validation Log loss is %.4f" % log_loss(y[val_idx], model3.predict(X.iloc[val_idx, :])))
    preds[:, i] = model3.predict(public)

dtrain = lgb.Dataset(X, label=y)

# public data
outcome = np.mean(preds, axis=1)
label = [1 if x > 0.41 else 0 for x in outcome]
public_csv = pd.DataFrame(OrderedDict({"ID": id_public, "prob": outcome, "label": label}))
public_csv.to_csv("public.csv", header=None, index=None)
