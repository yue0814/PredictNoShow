# -*- coding: utf-8 -*-
"""
Preprocessing part on the dataset

@author: Yue Peng
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder


def age_format(age):
    if age <= 48:
        return 1
    elif 48 < age <= 94:
        return 2
    else:
        return 3


def transform(x):

    year_app_made = [int(l[0]) for l in x.DateAppointmentWasMade.apply(lambda s: s.split("-"))]
    month_app_made = [int(l[1]) for l in x.DateAppointmentWasMade.apply(lambda s: s.split("-"))]
    day_app_made =[int(l[2]) for l in x.DateAppointmentWasMade.apply(lambda s: s.split("-"))]
    x["year_app_made"] = year_app_made
    x["month_app_made"] = month_app_made
    x["day_app_made"] = day_app_made
    year_app = [int(l[0]) for l in x.DateOfAppointment.apply(lambda s: s.split("-"))]
    month_app = [int(l[1]) for l in x.DateOfAppointment.apply(lambda s: s.split("-"))]
    day_app = [int(l[2]) for l in x.DateOfAppointment.apply(lambda s: s.split("-"))]
    x["year_app"] = year_app
    x["month_app"] = month_app
    x["day_app"] = day_app
    x.DateAppointmentWasMade = x.DateAppointmentWasMade.apply(lambda s: int(s.replace("-", "")))
    x.DateOfAppointment = x.DateOfAppointment.apply(lambda s: int(s.replace("-", "")))

    x["Gender"] = x.Gender.apply(lambda s: 1 if "M" else 0)

    mms = MinMaxScaler()
    mms.fit(x.DateAppointmentWasMade.values.reshape(-1, 1))
    x.DateAppointmentWasMade = mms.transform(x.DateAppointmentWasMade.values.reshape(-1, 1))
    mms.fit(x.DateOfAppointment.values.reshape(-1, 1))
    x.DateOfAppointment = mms.transform(x.DateOfAppointment.values.reshape(-1, 1))
    x["formatAge"] = x.Age.apply(age_format)
    
    # Standard Scale for Age
    for v in ["DaysUntilAppointment", "Age"]:
        ss = StandardScaler().fit(x[v].values.reshape(-1, 1))
        x[v+"_scaled"] = ss.fit_transform(x[v].values.reshape(-1, 1))
        x.drop([v], axis=1, inplace=True)

    le = LabelEncoder().fit(x["DayOfTheWeek"])
    label = le.transform(x["DayOfTheWeek"])
    x["DayOfTheWeek"] = label
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
