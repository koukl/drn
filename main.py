import tensorflow as tf
import numpy as np
from drn import *
import config
import time
import utils

np.random.seed(config.seed)
tf.set_random_seed(config.seed)

q, hidden_q = config.q, config.hidden_q
# training data
data_dir = config.data_dir
train_x = np.loadtxt(data_dir+'train_x.dat')
train_y = np.loadtxt(data_dir+'train_y.dat')
test_x = np.loadtxt(data_dir+'test_x.dat')
test_y = np.loadtxt(data_dir+'test_y.dat')
train_x = train_x[:config.Ntrain].reshape((-1, 1, q))
train_y = train_y[:config.Ntrain].reshape((-1, 1, q))
test_x = test_x[:config.Ntest].reshape((-1, 1, q))
test_y = test_y[:config.Ntest].reshape((-1, 1, q))

utils.ensure_dir('./run'+str(config.run_id))
Lfile = './run'+str(config.run_id)+'/L.dat'

# DRN fully-connected network
x, y, yout, W, B = build_network(config)
# loss
mse_loss = tf.losses.mean_squared_error(yout, y)
djs_loss = Ldjs(yout, y)
num_batches = int(train_x.shape[0]/config.batch_size)
train = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(mse_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = time.time()
    for epoch in xrange(config.Nepoch):
        avg_mse_loss = 0.0
        avg_djs_loss = 0.0
        # Loop over all batches
        for i in range(0, train_x.shape[0], config.batch_size):
            j = i + config.batch_size
            if j <= train_x.shape[0]:
                fd = {x: train_x[i:j], y: train_y[i:j]}
                _, l_, l2_ = sess.run([train, mse_loss, djs_loss], feed_dict=fd)
                avg_mse_loss += l_
                avg_djs_loss += l2_
        if epoch % 1 == 0:
            fd = {x: test_x, y: test_y}
            ltest_djs_ = sess.run(djs_loss, feed_dict=fd)
            print('epoch: ' + str(epoch) + ', mse: ' + str(avg_mse_loss / num_batches) )
            printop(Lfile,epoch, avg_mse_loss / num_batches, avg_djs_loss / num_batches, ltest_djs_, time.time()-t0)
    print ('training time: ' + str(time.time() - t0))

    # evaluate train and test
    fd = {x: train_x, y: train_y}
    ltrain_mse_, ltrain_djs_, predict_train = sess.run([mse_loss, djs_loss, yout], feed_dict=fd)
    fd = {x: test_x, y: test_y}
    ltest_mse_, ltest_djs_, predict_test = sess.run([mse_loss, djs_loss, yout], feed_dict=fd)
    print('ltrain mse: ' + str(ltrain_mse_))
    print('ltest mse: ' + str(ltest_mse_))
    print('ltrain djs: ' + str(ltrain_djs_))
    print('ltest djs: ' + str(ltest_djs_))
    np.savetxt('./run'+str(config.run_id)+'/trainop', predict_train.reshape((-1,q)))
    np.savetxt('./run'+str(config.run_id)+'/testop', predict_test.reshape((-1,q)))

    # save weights
    W_, B_ = sess.run([W, B])
    np.save('./run'+str(config.run_id)+'/W.npy', W_)
    np.save('./run'+str(config.run_id)+'/B.npy', B_)
    print('yay')
