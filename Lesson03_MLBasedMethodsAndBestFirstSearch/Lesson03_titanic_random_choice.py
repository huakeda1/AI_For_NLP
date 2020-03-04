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
plt.scatter(sub_age, sub_fare)
plt.show()


# fit a line
def func(age, k, b):
    return k * age + b


def loss(y, yhat):
    """
    :param y: the real fears
    :param yhat: the predict fears
    :return: how good is the estimated fares
    """
    return np.mean(np.abs(y - yhat))


min_error_rate = float('inf')
best_k, best_b = None, None
loop_times = 10000
losses = []
min_times = []

while loop_times > 0:
    # random choice  parameter k and b
    k_hat = random.random() * 20 - 10
    b_hat = random.random() * 20 - 20

    # loss
    estimated_fares = func(sub_age, k_hat, b_hat)
    error_rate = loss(y=sub_fare, yhat=estimated_fares)

    # save best k and b
    if error_rate < min_error_rate:
        best_k, best_b = k_hat, b_hat
        min_error_rate = error_rate
        print('loop == {}'.format(10000 - loop_times))
        min_times.append(10000 - loop_times)
        losses.append(min_error_rate)
        print('f(age) = {} * age + {} with error rate: {}'.format(best_k, best_b, min_error_rate))
    loop_times -= 1

# plot data with best fit line
# plt.scatter(sub_age, sub_fare)
# plt.plot(sub_age, func(sub_age, best_k, best_b), c='r')
# plt.show()

# plot error rate converge line
plt.plot(min_times[:10], losses[:10])
plt.show()
