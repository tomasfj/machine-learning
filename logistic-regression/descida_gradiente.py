import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import roc

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


# --Grafico-- #
def drawGraph(f, x, y, nome_graph):
    plt.plot(x, y)
    f.savefig(nome_graph)
# --- #


def h(X, theta):
    return 1.0 / (1.0 + np.exp(-np.matmul(X, theta)))

def J(hs, y):
    return  (np.matmul(np.transpose(y), np.log(hs))+np.matmul(np.transpose(1.0 - y), np.log(1.0 - hs)))/ - len(y)

# --- Descida do Gradiente --- #

learning_rate = 0.01
tot_iterations = 10000

theta_gd = np.random.uniform(-0.5, 0.5, size=(len(x_train[0]),))
theta_gd_initial = theta_gd

# lista de erros
Js_gd=[]

for i in range(tot_iterations):
    ts = np.zeros(len(x_train[0]), dtype=np.float64)
    predictions = h(x_train, theta_gd)

    for k in range(len(x_train[0])):
        ts[k] =  np.matmul(np.transpose(x_train[:,k]), predictions - y_train) / len(y_train)
        theta_gd[k] = theta_gd[k] - learning_rate * ts[k]

    Js_gd.append(J(h(x_train,theta_gd),y_train))

print('{:d}: Solution (Gradient descent): J={:.3f}'.format(i, J(h(x_train,theta_gd),y_train)))

f = plt.figure()
drawGraph(f, np.arange(tot_iterations), Js_gd, 'erro_p_epoca_gradiente.png')
#print("Thetas GD")
print(theta_gd)


pred = h(x_test, theta_gd)

# calcular ROC, Accuracy, Precision, Recal
roc.getROC(pred, y_test, 'roc_gradiente.png')