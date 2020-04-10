import tensorflow as tf
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import uniform
from sklearn.preprocessing import MinMaxScaler


def J(X, y, theta):
    preds = np.squeeze(np.matmul(X, theta))
    temp =  preds - np.squeeze(y)
    return np.sqrt(np.sum(np.matmul(np.transpose(temp), temp)))

#########################################################################
#   Read Data
#########################################################################


with open('wines.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) # to skip the header file
    X = []
    y = []
    for row in csv_reader:
        y.append(float(row[0] == 1))
        temp= [float(i) for i in row[1:]]
        X.append(temp)


X = np.asarray(X)
y = np.asarray(y)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X = np.append(X, np.ones((X.shape[0],1), np.float64), axis=1)



def h(X, theta):
    return 1.0 / (1.0 + np.exp(-np.matmul(X, theta)))

def J(hs, y):
    return  (np.matmul(np.transpose(y), np.log(hs))+np.matmul(np.transpose(1.0 - y), np.log(1.0 - hs)))/ - len(y)


#########################################################################
#   Gradient Descent method
#########################################################################

learning_rate = 0.01
tot_iterations = 100

theta_gd = np.random.uniform(-0.5, 0.5, size=(len(X[0]),))
theta_gd_initial = theta_gd

Js_gd=[]
for i in range(tot_iterations):
    ts = np.zeros(len(X[0]), dtype=np.float64)
    predictions = h(X, theta_gd)

    for k in range(len(X[0])):
        ts[k] =  np.matmul(np.transpose(X[:,k]), predictions - y) / len(y)
        theta_gd[k] = theta_gd[k] - learning_rate * ts[k]

    print('{:d}: Solution (Gradient descent): J={:.3f}'.format(i, J(h(X,theta_gd),y)))
    Js_gd.append(J(h(X,theta_gd),y))


#########################################################################
#   Tensor flow
#########################################################################

sess = tf.Session()

# Graph Definition

x_data = tf.placeholder(shape=[None, len(X[0])], dtype=tf.float64)
y_target = tf.placeholder(shape=[None], dtype=tf.float64)
weights = tf.Variable(tf.convert_to_tensor(theta_gd_initial))

# Define the Model
with tf.variable_scope('model_definition') as scope:
    model_output = 1.0/(1.0 + tf.exp(- tf.matmul(x_data, tf.expand_dims(weights,1))))
    scope.reuse_variables()


def J_tf(predict, y):
    return (tf.matmul(tf.transpose(tf.expand_dims(y, 1)), tf.log(predict))+ tf.matmul(tf.transpose(tf.expand_dims(1.0-y, 1)), tf.log(1.0 - predict))) / - tf.cast(tf.size(y), tf.float64)

loss = J_tf(model_output, y_target)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Graph execution

init = tf.global_variables_initializer()
sess.run(init)

Js_tf = []
for i in range(tot_iterations):
    sess.run(train_step, feed_dict={x_data: X, y_target: y})
    l = sess.run(loss, feed_dict={x_data: X, y_target: y})
    Js_tf.append(l[0][0])
    print('{:d}: Solution (Tensorflow): J={:.3f}'.format(i, l[0][0]))



#########################################################################
#   TResults Visualization
#########################################################################

plt.ion()
plt.figure(1)
plt.plot(range(tot_iterations), Js_gd, '-r', label='Gradient Descent')
plt.plot(range(tot_iterations), Js_tf, '-g', label='Tensorflow')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('fig.png')
plt.pause(0.01)

input('Close app?')


