import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, OneHotEncoder
from collections import OrderedDict
import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


train_data = pd.read_csv("Predict_NoShow_Train.csv")
private_test_data = pd.read_csv("Predict_NoShow_PrivateTest_WithoutLabels.csv")
public_test_data = pd.read_csv("Predict_NoShow_PublicTest_WithoutLabels.csv")


# remove some predictors
train_data.drop(["ID", "DateAppointmentWasMade", "DateOfAppointment"], axis=1, inplace=True)
# Age minmax
age_max, age_min = train_data.Age.max(), train_data.Age.min()
train_data.Age = train_data.Age.apply(lambda x: float(x - age_min)/ (age_max - age_min))
# Gender to numeric
train_data.Gender = train_data.Gender.apply(lambda x: 0 if x == "M" else 1)
# DayOfTheWeek to numeric
day_dict = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5,\
    "Saturday": 6, "Sunday": 7}
train_data.DayOfTheWeek = train_data.DayOfTheWeek.apply(lambda x: day_dict[x])
# DayOfTheWeek

train_data = train_data.drop(["Status"], axis=1)
train_data = np.array(train_data.astype("float64"))
train_data = scale(train_data)
# output
y = [0 if x == "Show-Up" else 1 for x in train_data["Status"]]
y = np.array(y).reshape(-1,1)
# one-hot
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()



X = tf.placeholder(tf.float64, shape=[None, 12])
y_ = tf.placeholder(tf.float64, shape=[None, 2])

W1 = tf.Variable(tf.truncated_normal([12, 64], stddev=0.1, dtype=tf.float64))
b1 = tf.Variable(tf.truncated_normal([64], stddev=0.1, dtype=tf.float64))
W2 = tf.Variable(tf.truncated_normal([64, 2], stddev=0.1, dtype=tf.float64))
b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1, dtype=tf.float64))
# W3 = tf.Variable(tf.truncated_normal([128, 32], stddev=0.1, dtype=tf.float64))
# b3 = tf.Variable(tf.truncated_normal([32], stddev=0.1, dtype=tf.float64))
# W4 = tf.Variable(tf.truncated_normal([32, 2], stddev=0.1, dtype=tf.float64))
# b4 = tf.Variable(tf.truncated_normal([2], stddev=0.1, dtype=tf.float64))

A1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
A2 = tf.matmul(A1, W2) + b2
# A3 = tf.nn.sigmoid(tf.matmul(A2, W3) + b3)
# A4 = tf.matmul(A3, W4) + b4
# A4 = tf.nn.dropout(A4, keep_prob=keep_prob)
prob = tf.nn.softmax(A2)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=A2)
loss = tf.reduce_mean(cross_entropy)
train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(A2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(5):
        total = 0
        acc_total = 0
        for i in range(9000):
            batch_xs, batch_ys = train_data[i*20: (i+1)*20, :], y[i*20:(i+1)*20,:]
            _, loss_, acc = sess.run([train, loss, accuracy], feed_dict={X: batch_xs, y_: batch_ys})
            total += loss_
            acc_total += 20*acc
            if i % 1000 == 0:
                print("%.6f" % (total/(20*(i+1))))
                print(acc_total/(20*(i+1)), "\n")
    pred = sess.run(prob, feed_dict={X: public_test_data, y_: np.zeros((60000, 2))})


# public data
# remove some predictors
id_public = public_test_data.ID
public_test_data = public_test_data.drop(["ID", "DateAppointmentWasMade", "DateOfAppointment"], axis=1)
# Age minmax
# age_max, age_min = public_test_data.Age.max(), public_test_data.Age.min()
# public_test_data.Age = public_test_data.Age.apply(lambda x: float(x - age_min)/ (age_max - age_min))
# Gender to numeric
gender_dict = {"M": 0, "F": 1}
gender = [gender_dict[x] for x in public_test_data["Gender"]]
public_test_data["Gender"] = pd.Series(gender)
# DayOfTheWeek to numeric
day_dict = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5,\
    "Saturday": 6, "Sunday": 7}
day = [day_dict[x] for x in public_test_data["DayOfTheWeek"]]
public_test_data["DayOfTheWeek"] = pd.Series(day)
public_test_data = np.array(public_test_data.astype("float64"))
public_test_data = scale(public_test_data)


label = [1 if x > 0.5 else 0 for x in pred[:, 1]]
public = pd.DataFrame(OrderedDict({"ID": id_public, "prob": pred[:,1], "label": label}))
public.to_csv("public.csv", header=None, index=None)


# PyTorch Network
# y = Variable(torch.Tensor(y), requires_grad=True)

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = DNN(12, 64, 1)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)