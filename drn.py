import numpy as np
import tensorflow as tf

def build_network(config):
    n_in, n_layers, n_nodes, n_out, q, hidden_q = config.n_in, config.n_layers, config.n_nodes, config.n_out, config.q, config.hidden_q
    x = tf.placeholder(tf.float32, [None, n_in, q])
    y = tf.placeholder(tf.float32, [None, n_out, q])
    W = []
    B = []

    if config.load_W_from_file:
        loadW = np.load('./run' + str(config.run_id) + '/W.npy')
        loadB = np.load('./run' + str(config.run_id) + '/B.npy')

    if n_layers == 0:
        with tf.variable_scope("l1"):
            if config.load_W_from_file:
                yout, Wi, Bi = fully_connected(x, n_in, n_out, q, q, loadparam=True, loadW=loadW[0], loadB=loadB[0])
            else:
                yout, Wi, Bi = fully_connected(x, n_in, n_out, q, q)
            W.append(Wi)
            B.append(Bi)
    else:
        with tf.variable_scope("l1"):
            if config.load_W_from_file:
                h, Wi, Bi = fully_connected(x, n_in, n_nodes, q, hidden_q, loadparam=True, loadW=loadW[0], loadB=loadB[0])
            else:
                h, Wi, Bi = fully_connected(x, n_in, n_nodes, q, hidden_q)
            W.append(Wi)
            B.append(Bi)
        for layer in xrange(2, config.n_layers + 1):
            with tf.variable_scope("l" + str(layer)):
                if config.load_W_from_file:
                    h, Wi, Bi = fully_connected(h, n_nodes, n_nodes, hidden_q, hidden_q, loadparam=True,
                                                loadW=loadW[layer - 1], loadB=loadB[layer - 1])
                else:
                    h, Wi, Bi = fully_connected(h, n_nodes, n_nodes, hidden_q, hidden_q)
                W.append(Wi)
                B.append(Bi)
        with tf.variable_scope("l" + str(config.n_layers + 1)):
            if config.load_W_from_file:
                yout, Wi, Bi = fully_connected(h, n_nodes, n_out, hidden_q, q, loadparam=True,
                                               loadW=loadW[config.n_layers], loadB=loadB[config.n_layers])
            else:
                yout, Wi, Bi = fully_connected(h, n_nodes, n_out, hidden_q, q)
            W.append(Wi)
            B.append(Bi)

    return x, y, yout, W, B


def cal_mult_bias(B, q):
    ba, bq, lama, lamq = B  # each contains multiple nodes bias values, of size nu x 1
    s0_np = np.arange(q, dtype=np.float32).reshape((1, q))
    s0 = tf.constant(s0_np)       # 1 x q
    # need account for multiple nodes in layer
    # s0 - b : (1 x q) x (nu x 1) = nu x q
    expB = tf.exp(-bq * tf.pow(s0 / q - lamq, 2) - ba * tf.abs(s0 / q - lama))  # nu x q
    return expB


def cal_logexp_bias(B, q):
    # returns log(exp(B)) which is B
    ba, bq, lama, lamq = B  # each contains multiple nodes bias values, of size nu x 1
    s0_np = np.arange(q, dtype=np.float32).reshape((1, q))
    s0 = tf.constant(s0_np)       # 1 x q
    # need account for multiple nodes in layer
    # s0 - b : (1 x q) x (nu x 1) = nu x q
    B = -bq * tf.pow(s0 / q - lamq, 2) - ba * tf.abs(s0 / q - lama)  # nu x q
    return B


def initD(ql, qu):
    # initialize constant matrix D(s1,s0), only depends on q, s1 is upper, s0 is lower
    D_np = np.zeros((qu, ql))
    for s1 in xrange(qu):
        for s0 in xrange(ql):
            D_np[s1, s0] = np.exp(-((float(s0)/ql - float(s1)/qu) ** 2))
    return D_np


def fully_connected(P, nl, nu, ql, qu, loadparam=False, loadW=None, loadB=None, fixWb=False, winit=0.1, winit_method='uniform'):
    # P is bs x nl x ql
    # returns ynorm: bs x nu x qu
    Dnp = initD(ql, qu).reshape((qu, ql, 1, 1))
    D = tf.constant(Dnp, dtype=tf.float32)
    D = tf.tile(D, [1, 1, nu, nl])
    if loadparam:
        if fixWb:
            ba, bq, lama, lamq = loadB
            W = tf.constant(loadW, name="W")
            ba = tf.constant(ba, name="ba")
            bq = tf.constant(bq, name="bq")
            lama = tf.constant(lama, name="lama")
            lamq = tf.constant(lamq, name="lamq")
        else:
            ba, bq, lama, lamq = loadB
            W = tf.Variable(loadW, "W")
            ba = tf.Variable(ba, "ba")
            bq = tf.Variable(bq, "bq")
            lama = tf.Variable(lama, "lama")
            lamq = tf.Variable(lamq, "lamq")
    else:
        if winit_method == 'uniform':
            W = tf.get_variable("W", [nu, nl], initializer=tf.random_uniform_initializer(-winit, winit))
            ba = tf.get_variable("ba", [nu, 1], initializer=tf.random_uniform_initializer(-winit, winit))
            bq = tf.get_variable("bq", [nu, 1], initializer=tf.random_uniform_initializer(-winit, winit))
            lama = tf.get_variable("lama", [nu, 1], initializer=tf.random_uniform_initializer(0.0,1.0))
            lamq = tf.get_variable("lamq", [nu, 1], initializer=tf.random_uniform_initializer(0.0,1.0))
        elif winit_method == 'normal':
            W = tf.get_variable("W", [nu, nl], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
            ba = tf.get_variable("ba", [nu, 1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
            bq = tf.get_variable("bq", [nu, 1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
            lama = tf.get_variable("lama", [nu, 1], initializer=tf.random_uniform_initializer(0.0, 1.0))
            lamq = tf.get_variable("lamq", [nu, 1], initializer=tf.random_uniform_initializer(0.0, 1.0))
        elif winit_method == 'glorot_normal':
            W = tf.get_variable("W", [nu, nl], initializer=tf.glorot_normal_initializer())
            ba = tf.get_variable("ba", [nu, 1], initializer=tf.glorot_normal_initializer())
            bq = tf.get_variable("bq", [nu, 1], initializer=tf.glorot_normal_initializer())
            lama = tf.get_variable("lama", [nu, 1], initializer=tf.random_uniform_initializer(0.0, 1.0))
            lamq = tf.get_variable("lamq", [nu, 1], initializer=tf.random_uniform_initializer(0.0, 1.0))

    B = [ba, bq, lama, lamq]
    Ptile = tf.tile(tf.reshape(P,[-1, 1, nl, ql, 1]), [1, nu, 1, 1, 1])  # bs x nu x nl x ql x 1
    T = tf.transpose(tf.pow(D, W), [2, 3, 0, 1])  # nu x nl x qu x ql
    Pw_unclipped = tf.squeeze(tf.einsum('jklm,ijkmn->ijkln', T, Ptile), axis=[4])   # bs x nu x nl x qu x 1 -> bs x nu x nl x qu
    # clip Pw by value to prevent zeros when weight is large
    Pw = tf.clip_by_value(Pw_unclipped, 1e-15, 1e+15)
    # perform underflow handling (product of probabilities become small as no. neighbors increase)
    # 1. log each term in Pw
    logPw = tf.log(Pw)  # bs x nu x nl x qu
    # 2. sum over neighbors
    logsum = tf.reduce_sum(logPw, axis=[2])       # bs x nu x qu
    # 3. log of exp of bias terms: log(expB) = exponent_B
    exponent_B = cal_logexp_bias(B, qu)  # nu x q
    # 4. add B to logsum
    logsumB = tf.add(logsum, exponent_B)          # bs x nu x qu
    # 5. find max over s0
    max_logsum = tf.reduce_max(logsumB, axis=[2], keep_dims=True)    # bs x nu x qu
    # 6. subtract max_logsum and exponentiate (the max term will have a result of exp(0) = 1, preventing underflow)
    # Now all terms will have been multiplied by exp(-max)
    expm_P = tf.exp(tf.subtract(logsumB, max_logsum))        # bs x nu x qu
    # normalize
    Z = tf.reduce_sum(expm_P, 2, keep_dims=True)
    ynorm = tf.divide(expm_P, Z)
    return ynorm, W, B


def KLdiv(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 1e-15, 1.0)
    y_pred = tf.clip_by_value(y_pred, 1e-15, 1.0)
    return tf.reduce_sum(y_true * tf.log(y_true / y_pred), axis=2)


def Ldjs(P,Q):
    M = (P+Q)/2.0
    l = 0.5*KLdiv(P,M)+0.5*KLdiv(Q,M)
    return tf.reduce_mean(l)/tf.log(2.0)


def printop(Lfile, epoch, Lmse, Ldjs, test_Ldjs, timing):
    f = open(Lfile, 'a')
    f.write(str(epoch) + '\t' + str(Lmse) + '\t' + str(Ldjs) +'\t' + str(test_Ldjs) +'\t' + str(timing) + '\n')
    f.close()