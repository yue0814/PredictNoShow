import pandas as pd
import numpy as np
from collections import OrderedDict
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import log_loss
import data

# pre-process and extraction
X, public, private = data.load_data()

# before fitting in models
id_public = public.pop("ID")
id_private = private.pop("ID")
X.pop("ID")
status = X.pop("Status")
y = [1 if i == "No-Show" else 0 for i in status]

# array
X = np.array(X)
y = np.array(y)
public = np.array(public)
private = np.array(private)

trainX, valX, trainY, valY = train_test_split(X, y, test_size=0.2, random_state=0)
trainX1, trainX2, trainY1, trainY2 = train_test_split(trainX, trainY, test_size=0.6, random_state=0)

# Grid Search
params = {"n_estimators": [10, 20, 30], "max_depth":[3, 5, 7, 9, 11],\
          "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],\
          "min_child_weight": [0.6, 0.7, 0.8], "scale_pos_weight": [0.8, 0.9, 1]}
xg_clf = xgb.XGBClassifier(eval_metric="logloss", objective="binary:logistic", seed=7)
xg_grid = GridSearchCV(xg_clf, params, cv=5)
xg_grid.fit(X, y)
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, eval_metric='logloss', gamma=0,
#        learning_rate=0.2, max_delta_step=0, max_depth=5,
#        min_child_weight=0.6, missing=None, n_estimators=30, n_jobs=1,
#        nthread=None, objective='binary:logistic', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=7, silent=True,
#        subsample=1)
model1 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, eval_metric='logloss', gamma=0,
       learning_rate=0.2, max_delta_step=0, max_depth=5,
       min_child_weight=0.6, missing=None, n_estimators=30, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=7, silent=False,
       subsample=1)
model1.fit(trainX1, trainY1)
model1.fit(X, y)
new_feature = model1.apply(trainX2)
new_feature = model1.apply(X)
trainX2_new = np.concatenate([trainX2, new_feature], 1)
X_new = np.concatenate([X, new_feature], 1)

model2 = xgb.XGBClassifier(learning_rate =0.05, #默认0.3
                           n_estimators=300,
                           max_depth=7,
                           min_child_weight=1,
                           gamma=0.5,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           objective= 'binary:logistic', #逻辑回归损失函数
                           nthread=8,  #cpu线程数
                           scale_pos_weight=1,
                           reg_alpha=1e-05,
                           reg_lambda=1,
                           seed=1024)  #随机种子
model2.fit(trainX2_new, trainY2)
model2.fit(X_new, y)

valX_new = np.concatenate([valX, model1.apply(valX)], 1)
np.mean(cross_val_score(model2, valX_new, valY, scoring="neg_log_loss", cv=5))

prob1 = model2.predict_proba(np.concatenate([public, model1.apply(public)], 1))[:, 1]
# xg_grid.best_estimator_ # eta0.15 maxdepth 7 minchildweight0.6 nesitimator30 scaleposweight 1

# params = {"n_estimators": [10, 20], "max_depth": [5, 10, 15], "min_samples_split": [2, 3]}
# rf_clf = RandomForestClassifier(random_state=7)
# clf_grid = GridSearchCV(rf_clf, params, cv=5, n_jobs=-1, verbose=1)
# clf_grid.fit(X, y)
#
# clf_grid.best_score_
# clf_grid.best_params_ # 20, 10, 2
#
# model2 = RandomForestClassifier(random_state=7, max_depth=10, min_samples_split=2, n_estimators=20)
# model2.fit(X, y)
# np.mean(cross_val_score(model2, X, y, scoring="neg_log_loss", cv=5))
# prob2 = model2.predict_proba(public)[:, 1]

# lightgbm
idx = [0, 1, 31, 32]
X1 = X[:, idx]
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
ltrain = lgb.Dataset(trainX, label=trainY)
lval = lgb.Dataset(valX, label=valY)
model3 = lgb.train(params, ltrain, num_boost_round=3000, valid_sets=lval, early_stopping_rounds=500)
log_loss(y, model3.predict(X))

xTrain = lgb.Dataset(X, label=y)
model3 = lgb.train(params, xTrain, num_boost_round=3000)
prob3 = model3.predict(public)

best = np.array(pd.read_csv("public1.csv", header=None))[:,1]

# public data
# outcome = 0.02*prob1 + 0.01*prob2 + 0.02*prob3 + 0.95*best 0.60748
outcome = 0.05*prob1 + 0.25*prob3 + 0.7*best

label = [1 if x > 0.4 else 0 for x in outcome]
public_csv = pd.DataFrame(OrderedDict({"ID": id_public, "prob": outcome, "label": label}))
public_csv.to_csv("public.csv", header=None, index=None)
