# -*- coding: utf-8 -*-
"""
Preprocessing part on the dataset

@author: Yue Peng
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler


def age_format(age):
    if age <= 48:
        return "A"
    elif 48 < age <= 94:
        return "B"
    else:
        return "C"


def transform(x):
    # 1
    # x_train = x_train[x_train.DaysUntilAppointment.apply(lambda x: True if x <= 30 else False)]
    year_app_made = [l[0] for l in x.DateAppointmentWasMade.apply(lambda s: s.split("-"))]
    month_app_made = [l[1] for l in x.DateAppointmentWasMade.apply(lambda s: s.split("-"))]
    day_app_made =[l[2] for l in x.DateAppointmentWasMade.apply(lambda s: s.split("-"))]
    x["year_app_made"] = year_app_made
    x["month_app_made"] = month_app_made
    x["day_app_made"] = day_app_made
    year_app = [l[0] for l in x.DateOfAppointment.apply(lambda s: s.split("-"))]
    month_app = [l[1] for l in x.DateOfAppointment.apply(lambda s: s.split("-"))]
    day_app = [l[2] for l in x.DateOfAppointment.apply(lambda s: s.split("-"))]
    x["year_app"] = year_app
    x["month_app"] = month_app
    x["day_app"] = day_app
    x.DateAppointmentWasMade = x.DateAppointmentWasMade.apply(lambda s: int(s.replace("-", "")))
    x.DateOfAppointment = x.DateOfAppointment.apply(lambda s: int(s.replace("-", "")))

    mms = MinMaxScaler()
    mms.fit(x.DateAppointmentWasMade.values.reshape(-1, 1))
    x.DateAppointmentWasMade = mms.transform(x.DateAppointmentWasMade.values.reshape(-1, 1))
    mms.fit(x.DateOfAppointment.values.reshape(-1, 1))
    x.DateOfAppointment = mms.transform(x.DateOfAppointment.values.reshape(-1, 1))
    x["formatAge"] = x.Age.apply(age_format)
    # One-Hot for categorical predictors
    for _, v in enumerate(x.columns.values):
        if v not in ["ID", "Age", "Status", "DateAppointmentWasMade",\
                     "DateOfAppointment", "DaysUntilAppointment"]:
            le = LabelEncoder().fit(x[v])
            label = le.transform(x[v])
            ohe = OneHotEncoder(sparse=False).fit(label.reshape(-1, 1))
            arrOH = ohe.transform(label.reshape(-1, 1))
            for i in range(arrOH.shape[1]):
                x[v + "_" + str(i)] = arrOH[:, i]
            x.drop([v], axis=1, inplace=True)
    # Standard Scale for Age
    for v in ["DaysUntilAppointment", "Age"]:
        ss = StandardScaler().fit(x[v].values.reshape(-1, 1))
        x[v+"_scaled"] = ss.fit_transform(x[v].values.reshape(-1, 1))
        x.drop([v], axis=1, inplace=True)

    return x


def load_data():
    train_data = pd.read_csv("Predict_NoShow_Train.csv")
    private_test_data = pd.read_csv("Predict_NoShow_PrivateTest_WithoutLabels.csv")
    public_test_data = pd.read_csv("Predict_NoShow_PublicTest_WithoutLabels.csv")
    x_train = transform(train_data)
    public = transform(public_test_data)
    private = transform(private_test_data)
    if not x_train.shape[1]-1 == public.shape[1] == private.shape[1]:
        for i in list(set(x_train.columns.values) - set(public.columns.values)):
            if not i == "Status":
                public[i] = np.zeros((public.shape[0], 1))
        for i in list(set(x_train.columns.values) - set(private.columns.values)):
            if not i == "Status":
                private[i] = np.zeros((public.shape[0], 1))

    return x_train, public, private


# find threshold for age formatting
# d={}
# for i in range(1, 100):
#     tmp1 = train_data[train_data.Age>i].groupby("Status").size()[0]/\
#         (train_data[train_data.Age>i].groupby("Status").size()[0]+
#          train_data[train_data.Age > i].groupby("Status").size()[1])
#     tmp2 = train_data[train_data.Age <= i].groupby("Status").size()[0] / \
#            (train_data[train_data.Age <= i].groupby("Status").size()[0] +
#             train_data[train_data.Age <= i].groupby("Status").size()[1])
#     d[i] = abs(tmp1 - tmp2)
#
# [k for k, v in d.items() if v in sorted(list(d.values()))[90:99]]