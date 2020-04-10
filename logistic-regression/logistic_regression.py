import csv
import numpy as np
from sklearn.model_selection import train_test_split
import descida_gradiente
import tensor_flow

X = []
Y = []

filename_read = 'normal_pulsar_stars.csv'
with open(filename_read) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        X.append(np.float64(row[0:8]))
        Y.append(np.float64(row[8]))


x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=0 )

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

learning_rate = 0.01
tot_iterations = 10000

theta_gd = np.random.uniform(-0.5, 0.5, size=(len(x_train[0]),))
theta_gd_initial = theta_gd

descida_gradiente.descida_gradiente(x_train, y_train, x_test, y_test, learning_rate, tot_iterations, theta_gd)

tensor_flow.tensor_flow(x_train, y_train, x_test, y_test, learning_rate, tot_iterations, theta_gd_initial)