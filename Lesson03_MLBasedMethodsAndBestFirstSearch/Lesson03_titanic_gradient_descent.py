#!/usr/bin/env python
# _*_coding:utf-8_*_

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#  load data
content = pd.read_csv('datasource/titanic/train.csv')
content = content.dropna()

# select data
age_with_fares = content[
    (content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)
    ]
sub_fare = age_with_fares['Fare']
sub_age = age_with_fares['Age']
# plt.scatter(sub_age, sub_fare)
# plt.show()


# fit a line
def func(age, k, b):
    return k * age + b


def loss(y, yhat):
    """
    mean square loss
    :param y: the real fears
    :param yhat: the predict fears
    :return: how good is the estimated fares
    """
    return np.mean(np.abs(y - yhat))
    # return np.mean(np.sqrt(y - yhat))
    # return np.mean(no.square(y - yhat))


def derivate_k(y, yhat, x):
    """
    loss 对k求导 loss = | y - (k * x + b)|
    :param y:
    :param yhat:
    :param x:
    :return:
    """
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.sum([a * -x_i for a, x_i in zip(abs_values, x)])


def derivate_b(y, yhat):
    """
    loss 对b求导 loss = | y - (k * x + b)|
    :param y:
    :param yhat:
    :return:
    """
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.sum([-a for a in abs_values])


loop_times = 10000
losses = []
k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10
learning_rate = 1e-3


while loop_times > 0:
    #  update parameters -1是表示需要梯度下降
    k_delta = -1 * learning_rate * derivate_k(sub_fare, func(sub_age, k_hat, b_hat), sub_age)
    b_delta = -1 * learning_rate * derivate_b(sub_fare, func(sub_age, k_hat, b_hat))
    k_hat += k_delta
    b_hat += b_delta

    # loss
    estimated_fares = func(sub_age, k_hat, b_hat)
    error_rate = loss(y=sub_fare, yhat=estimated_fares)

    print('loop == {}'.format(10000 - loop_times))
    losses.append(error_rate)
    print('f(age) = {} * age + {} with error rate: {}'.format(k_hat, b_hat, error_rate))

    loop_times -= 1

# plot best fit line
# plt.scatter(sub_age, sub_fare)
# plt.plot(sub_age, func(sub_age, k_hat, b_hat), c='r')
# plt.show()

# Plot error rate converge
plt.plot(range(len(losses[:100])), losses[:100])
plt.show()
