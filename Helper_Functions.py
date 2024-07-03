from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import os.path
import pickle
from accountant import GaussianMomentsAccountant
import math
import os
import tensorflow as tf

class PrivAgent:
    def __init__(self, N, Name):
        self.N = N
        self.Name = Name
        if N == 100:
            self.m = [30]*100
            self.Sigma = [1]*24
            self.bound = 0.001
        if N == 1000:
            self.m = [100]*10
            self.Sigma = [1]*24
            self.bound = 0.00001
        if N == 10000:
            self.m = [300]*10
            self.Sigma = [1]*24
            self.bound = 0.000001
        if(N != 100 and N != 1000 and N != 10000 ):
            print('!!!!!!! YOU CAN ONLY USE THE PRIVACY AGENT FOR N = 100, 1000 or 10000 !!!!!!!')

    def get_m(self, r):
        return self.m[r]

    def get_Sigma(self, r):
        return self.Sigma[r]

    def get_bound(self):
        return self.bound

def Assignements(dic):
    return [tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()]

def Vname_to_Pname(var):
    return var.name[:var.name.find(':')] + '_placeholder'

def Vname_to_FeedPname(var):
    return var.name[:var.name.find(':')] + '_placeholder:0'

def Vname_to_Vname(var):
    return var.name[:var.name.find(':')]

class WeightsAccountant:
    def __init__(self, sess, model, Sigma, real_round):
        self.Weights = [np.expand_dims(sess.run(v), -1) for v in tf.trainable_variables()]
        self.keys = [Vname_to_FeedPname(v) for v in tf.trainable_variables()]

        self.global_model = [model[k] for k in self.keys]
        self.Sigma = Sigma
        self.Updates = []
        self.median = []
        self.Norms = []
        self.ClippedUpdates = []
        self.m = 0.0
        self.num_weights = len(self.Weights)
        self.round = real_round

    def save_params(self, save_dir):
        filehandler = open(save_dir + '/weights_accountant_round_' + str(self.round) + '.pkl', "wb")
        pickle.dump(self, filehandler)
        filehandler.close()

    def allocate(self, sess):
        self.Weights = [np.concatenate((self.Weights[i], np.expand_dims(sess.run(tf.trainable_variables()[i]), -1))
                        for i in range(self.num_weights)]

    def compute_updates(self):
        self.Updates = [self.Weights[i]-np.expand_dims(self.global_model[i], -1) for i in range(self.num_weights)]
        self.Weights = None

    def compute_norms(self):
        self.Norms = [np.sqrt(np.sum(np.square(self.Updates[i]), axis=tuple(range(self.Updates[i].ndim)[:-1]), keepdims=True)) for i in range(self.num_weights)]

    def clip_updates(self):
        self.compute_updates()
        self.compute_norms()

        self.median = [np.median(self.Norms[i], axis=-1, keepdims=True) for i in range(self.num_weights)]
        factor = [self.Norms[i]/self.median[i] for i in range(self.num_weights)]
        for i in range(self.num_weights):
            factor[i][factor[i] > 1.0] = 1.0
        self.ClippedUpdates = [self.Updates[i]/factor[i] for i in range(self.num_weights)]

    def Update_via_GaussianMechanism(self, sess, Acc, FLAGS, Computed_deltas):
        self.clip_updates()
        self.m = float(self.ClippedUpdates[0].shape[-1])
        MeanClippedUpdates = [np.mean(self.ClippedUpdates[i], -1) for i in range(self.num_weights)]
        GaussianNoise = [(1.0/self.m * np.random.normal(loc=0.0, scale=float(self.Sigma * self.median[i]), size=MeanClippedUpdates[i].shape)) for i in range(self.num_weights)]
        Sanitized_Updates = [MeanClippedUpdates[i] + GaussianNoise[i] for i in range(self.num_weights)]
        New_weights = [self.global_model[i] + Sanitized_Updates[i] for i in range(self.num_weights)]
        New_model = dict(zip(self.keys, New_weights))

        t = Acc.accumulate_privacy_spending(0, self.Sigma, self.m)
        delta = 1
        if FLAGS.record_privacy:
            if not FLAGS.relearn:
                for j in range(len(self.keys)):
                    sess.run(t)
                r = Acc.get_privacy_spent(sess, [FLAGS.eps])
                delta = r[0][1]
            else:
                delta = Computed_deltas[self.round]
        return New_model, delta

def create_save_dir(FLAGS):
    raw_directory = FLAGS.save_dir + '/'
    gm_str = 'Dp/' if FLAGS.gm else 'non_Dp/'
    if FLAGS.priv_agent:
        model = gm_str + 'N_' + str(FLAGS.n) + '/Epochs_' + str(int(FLAGS.e)) + '_Batches_' + str(int(FLAGS.b))
        return raw_directory + str(model) + '/' + FLAGS.PrivAgentName
    else:
        model = gm_str + 'N_' + str(FLAGS.n) + '/Sigma_' + str(FLAGS.Sigma) + '_C_' + str(FLAGS.m) + '/Epochs_' + str(int(FLAGS.e)) + '_Batches_' + str(int(FLAGS.b))
        return raw_directory + str(model)

def load_from_directory_or_initialize(directory, FLAGS):
    Accuracy_accountant = []
    Delta_accountant = [0]
    model = []
    real_round = 0
    Acc = GaussianMomentsAccountant(FLAGS.n)
    FLAGS.loaded = False
    FLAGS.relearn = False
    Computed_Deltas = []

    if not os.path.isfile(directory + '/model.pkl'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('No loadable model found. All updates stored at: ' + directory)
        print('... Initializing a new model ...')
    else:
        if os.path.isfile(directory + '/specs.csv'):
            with open(directory + '/specs.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                Lines = [list(map(float, line)) for line in reader]
                Accuracy_accountant = Lines[-1]
                Delta_accountant = Lines[1]

            if math.isnan(Delta_accountant[-1]):
                Computed_Deltas = Delta_accountant
                FLAGS.relearn = True
                if math.isnan(Accuracy_accountant[-1]):
                    print('A model identical to that specified was already learned. Another one is learned and appended')
                    Accuracy_accountant = []
                    Delta_accountant = [0]
                else:
                    print('A model identical to that specified was already learned. For a second one learning is resumed')
                    Delta_accountant = Delta_accountant[:len(Accuracy_accountant)]
                    with open(directory + '/model.pkl', 'rb') as fil:
                        model = pickle.load(fil)
                    FLAGS.loaded = True
                return model, Accuracy_accountant, Delta_accountant, Acc, real_round, FLAGS, Computed_Deltas
            else:
                real_round = len(Accuracy_accountant) - 1
                with open(directory + '/model.pkl', 'rb') as fil:
                    model = pickle.load(fil)
                FLAGS.loaded = True
        else:
            print('there seems to be a model, but no saved progress. Fix that.')
            raise KeyboardInterrupt
    return model, Accuracy_accountant, Delta_accountant, Acc, real_round, FLAGS, Computed_Deltas

def save_progress(save_dir, model, Delta_accountant, Accuracy_accountant, PrivacyAgent, FLAGS):
    with open(save_dir + '/model.pkl', "wb") as filehandler:
        pickle.dump(model, filehandler)

    if not FLAGS.relearn:
        with open(save_dir + '/specs.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if FLAGS.priv_agent:
                writer.writerow([0] + [PrivacyAgent.get_m(r) for r in range(len(Delta_accountant)-1)])
            else:
                writer.writerow([0] + [FLAGS.m] * (len(Delta_accountant)-1))
            writer.writerow(Delta_accountant)
            writer.writerow(Accuracy_accountant)
    else:
        if len(Accuracy_accountant) > 1 or (len(Accuracy_accountant) == 1 and FLAGS.loaded):
            with open(save_dir + '/specs.csv', 'r') as csvfile:
                csvReader = csv.reader(csvfile)
                lines = [list(map(float, row)) for row in csvReader]
                lines = lines[:-1]

            with open(save_dir + '/specs.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(lines)

        with open(save_dir + '/specs.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(Accuracy_accountant)

def global_step_creator():
    global_step = [v for v in tf.global_variables() if v.name == "global_step:0"][0]
    global_step_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name='global_step_placeholder')
    one = tf.constant(1, dtype=tf.float32, name='one')
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return increase_global_step, set_global_step

def bring_Accountant_up_to_date(Acc, sess, rounds, PrivAgent, FLAGS):
    print('Bringing the accountant up to date....')
    for r in range(rounds):
        if FLAGS.priv_agent:
            Sigma = PrivAgent.get_Sigma(r)
            m = PrivAgent.get_m(r)
        else:
            Sigma = FLAGS.sigma
            m = FLAGS.m
        print('Completed ' + str(r+1) + ' out of ' + str(rounds) + ' rounds')
        t = Acc.accumulate_privacy_spending(0, Sigma, m)
        sess.run(t)
        sess.run(t)
        sess.run(t)
    print('The accountant is up to date!')

def print_loss_and_accuracy(global_loss, accuracy):
    print(' - Current Model has a loss of:           %s' % global_loss)
    print(' - The Accuracy on the validation set is: %s' % accuracy)
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')

def print_new_comm_round(real_round):
    print('--------------------------------------------------------------------------------------')
    print('------------------------ Communication round %s ---------------------------------------' % str(real_round))
    print('--------------------------------------------------------------------------------------')

def check_validaity_of_FLAGS(FLAGS):
    FLAGS.priv_agent = True
    if not FLAGS.m == 0:
        if FLAGS.sigma == 0:
            print('\n \n -------- If m is specified the Privacy Agent is not used, then Sigma has to be specified too. --------\n \n')
            raise NotImplementedError
    if not FLAGS.sigma == 0:
        if FLAGS.m == 0:
            print('\n \n-------- If Sigma is specified the Privacy Agent is not used, then m has to be specified too. -------- \n \n')
            raise NotImplementedError
    if not FLAGS.sigma == 0 and not FLAGS.m == 0:
        FLAGS.priv_agent = False
    return FLAGS

class Flag:
    def __init__(self, n, b, e, record_privacy, m, sigma, eps, save_dir, log_dir, max_comm_rounds, gm, PrivAgent):
        if not save_dir:
            save_dir = os.getcwd()
        if not log_dir:
            log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed')
        if tf.io.gfile.exists(log_dir):
            tf.io.gfile.rmtree(log_dir)
        tf.io.gfile.makedirs(log_dir)
        self.n = n
        self.sigma = sigma
        self.eps = eps
        self.m = m
        self.b = b
        self.e = e
        self.record_privacy = record_privacy
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.max_comm_rounds = max_comm_rounds
        self.gm = gm
        self.PrivAgentName = PrivAgent.Name
