# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import calendar
from dateutil.parser import parse


def authors():
    print("Ludan Zhang, Jiachen Zhang, Yue Peng")


#Change strings into numbers
def str_num(input):
    x = input.copy()
    weekday_dict = {"Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7}
    x.loc[x['Gender'] == 'M',"Gender"] = 1
    x.loc[x['Gender'] == 'F',"Gender"] = 0
    x["DayOfTheWeek"] = [weekday_dict[day] for day in x["DayOfTheWeek"]]
    x["MadeAppointmentweekday"] = [parse(day).weekday()+1 for day in x["DateAppointmentWasMade"]]
    x["unhealthy"] = x["Diabetes"] + x["Alcoholism"] + x["Hypertension"] + x["Handicapped"] + x["Smoker"]
    return x


# save results
def save_pred(id, pred, name):
    result = pd.DataFrame({"ID":id, "pred":pred})
    result["label"] = 0
    result.loc[result["pred"] >= 0.5,"label"] = 1
    result.to_csv(name, header = None, index = None)


# train data
def predictNoshow(train):
    X = train.iloc[:,[1,2,3,5,6,7,8,9,10,11,12,13,14,15]]
    pubtest = pd.read_csv("Predict_NoShow_PublicTest_WithoutLabels.csv")
    prvtest = pd.read_csv("Predict_NoShow_PrivateTest_WithoutLabels.csv")
    pubtestX = pubtest.iloc[:,[1,2,3,5,6,7,8,9,10,11,12,13,14]]
    prvtestX = prvtest.iloc[:,[1,2,3,5,6,7,8,9,10,11,12,13,14]]
    pubid = pubtest["ID"]
    prvid = prvtest["ID"]



    newX = str_num(X)
    newX.loc[X["Status"] == "Show-Up", "Status"] = 0
    newX.loc[X["Status"] == "No-Show", "Status"] = 1
    Y = newX.pop("Status")
    newpubtestX = str_num(pubtestX)
    newprvtestX = str_num(prvtestX)

    #imbalance to balance
    # from imblearn.over_sampling import SMOTE
    # sm = SMOTE(ratio='minority', random_state=42, k_neighbors=5, m_neighbors=10, kind='regular')
    # var_select = ['Age', 'Gender','DaysUntilAppointment', 'Diabetes', 'Alcoholism', 'Hypertension',
    #        'Handicapped', 'Smoker', 'Scholarship', 'RemindedViaSMS', 'DayOfTheWeek']
    # balanceX, balanceY = sm.fit_sample(newX[var_select],Y)
    # sum(balanceY == 0) / sum(balanceY == 1) 
    # balanceX = pd.DataFrame(balanceX)
    # balanceX.columns = var_select
    # balanceX["Status"] = balanceY
    # balanceX.to_csv("balanced_train.csv")

    #fit models
    var_select1 = ['Age', 'Gender', 'DaysUntilAppointment', 'Diabetes', 'Alcoholism', 'Hypertension',
        'Handicapped', 'Smoker', 'Scholarship', 'RemindedViaSMS', 'DayOfTheWeek','Tuberculosis']
    xgb_best1 = xgb.XGBRegressor(objective = "binary:logistic", eval_metric = "logloss",
        learning_rate = 0.01,max_depth=4, min_child_weight=4, gamma = 0, subsample=1,
        colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1, n_estimators = 1100)
    xgb_best1.fit(newX[var_select1], Y)
    pubpred1 = xgb_best1.predict(newpubtestX[var_select1])
    prvpred1 = xgb_best1.predict(newprvtestX[var_select1])
    xgb_best2 = xgb.XGBRegressor(objective = "binary:logistic",
        n_estimators=20, max_depth=5, eval_metric="logloss")
    xgb_best2.fit(newX[var_select1], Y)
    pubpred2 = xgb_best2.predict(newpubtestX[var_select1])
    prvpred2 = xgb_best2.predict(newprvtestX[var_select1])


    var_select3 = ['Age', 'Gender', 'DaysUntilAppointment', 'unhealthy', 'Scholarship',
        'RemindedViaSMS', 'DayOfTheWeek']
    xgb_best3 = xgb.XGBRegressor(objective = "binary:logistic", eval_metric = "logloss",
        learning_rate = 0.1, max_depth=3, min_child_weight=5, gamma = 2, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.01, n_estimators=100)
    xgb_best3.fit(newX[var_select3], Y)
    pubpred3 = xgb_best3.predict(newpubtestX[var_select3])
    prvpred3 = xgb_best3.predict(newprvtestX[var_select3])

    var_select4 = ['Age', 'Gender', 'DaysUntilAppointment', 'Diabetes', 'Alcoholism',
        'Hypertension','Handicapped', 'Smoker', 'Scholarship', 'RemindedViaSMS', 'DayOfTheWeek','MadeAppointmentweekday']
    xgb_best4 = xgb.XGBRegressor(objective = "binary:logistic", eval_metric = "logloss",
        learning_rate = 0.01,max_depth=5, min_child_weight=3, gamma = 2, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.01, reg_lambda=3, n_estimators = 1000)
    xgb_best4.fit(newX[var_select4], Y)
    pubpred4 = xgb_best4.predict(newpubtestX[var_select4])
    prvpred4 = xgb_best4.predict(newprvtestX[var_select4])


    balanced = pd.read_csv("balanced_train.csv")
    var_select5 = ['Age', 'Gender', 'DaysUntilAppointment', 'Diabetes', 'Alcoholism',
        'Hypertension','Handicapped', 'Smoker', 'Scholarship', 'RemindedViaSMS', 'DayOfTheWeek']
    balanceX = balanced[var_select5]
    balanceY = balanced[["Status"]]
    xgb_best5 = xgb.XGBRegressor(objective = "binary:logistic", eval_metric = "logloss",
        learning_rate = 0.01,max_depth=16, min_child_weight=1, gamma = 2, subsample=1,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=3, n_estimators=500)
    xgb_best5.fit(balanceX,balanceY)
    pubpred5 = xgb_best5.predict(newpubtestX[var_select5])
    prvpred5 = xgb_best5.predict(newprvtestX[var_select5])

    pubpred = 0.9*(0.8*(0.1*pubpred3 + 0.4*pubpred4 + 0.5*pubpred1) + 0.2*pubpred2) + 0.1*pubpred5
    prvpred = 0.9*(0.8*(0.1*prvpred3 + 0.4*prvpred4 + 0.5*prvpred1) + 0.2*prvpred2) + 0.1*prvpred5



    save_pred(pubid, pubpred, "public.csv")
    save_pred(prvid, prvpred, "private.csv")


if __name__ == "__main__":
    authors()
    train = pd.read_csv("Predict_NoShow_Train.csv")
    predictNoshow(train)