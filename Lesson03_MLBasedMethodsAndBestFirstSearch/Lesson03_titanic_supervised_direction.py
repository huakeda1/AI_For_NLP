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
    mean square loss
    :param y: the real fears
    :param yhat: the predict fears
    :return: how good is the estimated fares
    """
    # return np.mean(np.abs(y - yhat))
    return np.mean(np.sqrt(y - yhat))
    # return np.mean(no.square(y - yhat))


min_error_rate = float('inf')
loop_times = 10000
losses = []

best_direction = None
change_directions = [
    (+1, -1),  # k increase, b decrease
    (+1, +1),
    (-1, +1),
    (-1, -1),  # k decrease, b decrease
]

k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10
best_k = k_hat
best_b = b_hat


def step():
    return random.random()


direction = random.choice(change_directions)
while loop_times > 0:
    #  update parameters
    k_delta_direction, b_delta_direction = direction
    k_delta = k_delta_direction * step()
    b_delta = b_delta_direction * step()
    new_k = k_delta + best_k
    new_b = b_delta + best_b

    # loss
    estimated_fares = func(sub_age, new_k, new_b)
    error_rate = loss(y=sub_fare, yhat=estimated_fares)

    #  save best direction
    if error_rate < min_error_rate:
        best_k, best_b = new_k, new_b
        min_error_rate = error_rate

        direction = (k_delta_direction, b_delta_direction)
        print('loop =={}'.format(10000 - loop_times))
        losses.append(min_error_rate)
        print('f(age) = {} * age + {} with error rate: {}'.format(best_k, best_b, min_error_rate))
    else:
        direction = random.choice(change_directions)

    loop_times -= 1

# plot line
plt.scatter(sub_age, sub_fare)
plt.plot(sub_age, func(sub_age, best_k, best_b), c='r')
plt.show()

# Plot error rate converge
# plt.plot(range(len(losses)), losses)
# plt.show()
