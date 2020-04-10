import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
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


'''
recebe:
x_train
y_train
x_test
y_test
learning_rate
tot_iterations
theta_gd_initial
'''

# --Desenhar Grafico-- #
def drawGraph(f, x, y, nome_graph):
    plt.plot(x, y)
    f.savefig(nome_graph)


learning_rate = 0.01
tot_iterations = 10000

theta_gd = np.random.uniform(-0.5, 0.5, size=(len(x_train[0]),))
theta_gd_initial = theta_gd


def tensor_flow(x_train, y_train, x_test, y_test, learning_rate, tot_iterations, theta_gd_initial):
    sess = tf.Session()

    # Graph Definition

    x_data = tf.placeholder(shape=[None, len(x_train[0])], dtype=tf.float64)
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
        sess.run(train_step, feed_dict={x_data: x_train, y_target: y_train})
        l = sess.run(loss, feed_dict={x_data: x_train, y_target: y_train})
        Js_tf.append(l[0][0])


    print('{:d}: Solution (Tensorflow): J={:.3f}'.format(i, l[0][0]))

    f = plt.figure()
    drawGraph(f, np.arange(tot_iterations), Js_tf, 'erro_p_epoca_tensorFlow.png')

    pred = sess.run(model_output, feed_dict={x_data: x_test, y_target: y_test})

    # calcular ROC, Accuracy, Precision, Recal
    roc.getROC(pred, y_test, 'roc_tensorFlow.png')