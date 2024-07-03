from Helper_Functions import Vname_to_FeedPname, Vname_to_Pname, check_validaity_of_FLAGS, create_save_dir, \
    global_step_creator, load_from_directory_or_initialize, bring_Accountant_up_to_date, save_progress, \
    WeightsAccountant, print_loss_and_accuracy, print_new_comm_round, PrivAgent, Flag

def run_differentially_private_federated_averaging(loss, train_op, eval_correct, data, data_placeholder,
                                                   label_placeholder, privacy_agent=None, b=10, e=4,
                                                   record_privacy=True, m=0, sigma=0, eps=8, save_dir=None,
                                                   log_dir=None, max_comm_rounds=3000, gm=True,
                                                   saver_func=create_save_dir, save_params=False):
    """
    This function will simulate a federated learning setting and enable differential privacy tracking.
    """

    # If no privacy agent was specified, the default privacy agent is used.
    if not privacy_agent:
        privacy_agent = PrivAgent(len(data.client_set), 'default_agent')

    # A Flags instance is created that will fuse all specified parameters and default those that are not specified.
    FLAGS = Flag(len(data.client_set), b, e, record_privacy, m, sigma, eps, save_dir, log_dir, max_comm_rounds, gm,
                 privacy_agent)

    # Check whether the specified parameters make sense.
    FLAGS = check_validaity_of_FLAGS(FLAGS)

    # At this point, FLAGS.save_dir specifies both; where we save progress and where we assume the data is stored
    save_dir = saver_func(FLAGS)

    # This function will retrieve the variable associated to the global step and create nodes that serve to
    # increase and reset it to a certain value.
    increase_global_step, set_global_step = global_step_creator()

    # model_placeholder : a dictionary in which there is a placeholder stored for every trainable variable defined
    # in the tensorflow graph.
    model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.compat.v1.trainable_variables()],
                                 [tf.keras.Input(shape=var.shape, dtype=tf.float32, name=Vname_to_Pname(var))
                                  for var in tf.compat.v1.trainable_variables()]))

    # assignments : Is a list of nodes. when run, all trainable variables are set to the value specified through
    # the placeholders in 'model_placeholder'.
    assignments = [tf.compat.v1.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in
                   tf.compat.v1.trainable_variables()]

    # load_from_directory_or_initialize checks whether there is a model at 'save_dir' corresponding to the one we
    # are building. If so, training is resumed, if not, it returns:  - model = []
    #                                                                - accuracy_accountant = []
    #                                                                - delta_accountant = []
    #                                                                - real_round = 0
    # And initializes a Differential_Privacy_Accountant as acc
    model, accuracy_accountant, delta_accountant, acc, real_round, FLAGS, computed_deltas = \
        load_from_directory_or_initialize(save_dir, FLAGS)

    m = int(FLAGS.m)
    sigma = float(FLAGS.sigma)
    # - m : amount of clients participating in a round
    # - sigma : variable for the Gaussian Mechanism.
    # Both will only be used if no Privacy_Agent is deployed.

    ################################################################################################################

    # Usual Tensorflow...

    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)

    ################################################################################################################

    # If there was no loadable model, we initialize a new model:
    if not model:
        model = dict(zip([Vname_to_FeedPname(var) for var in tf.compat.v1.trainable_variables()],
                         [sess.run(var) for var in tf.compat.v1.trainable_variables()]))
        model['global_step_placeholder:0'] = 0
        real_round = 0
        weights_accountant = []

    # If a model is loaded, and we are not relearning it, we have to get the privacy accountant up to date.
    if not FLAGS.relearn and real_round > 0:
        bring_Accountant_up_to_date(acc, sess, real_round, privacy_agent, FLAGS)

    ################################################################################################################

    # This is where the actual communication rounds start:

    data_set_asarray = np.asarray(data.sorted_x_train)
    label_set_asarray = np.asarray(data.sorted_y_train)

    for r in range(FLAGS.max_comm_rounds):
        # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
        sess.run(assignments, feed_dict=model)

        # create a feed-dict holding the validation set.
        feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
                     str(label_placeholder.name): np.asarray(data.y_vali)}

        # compute the loss on the validation set.
        global_loss = sess.run(loss, feed_dict=feed_dict)
        count = sess.run(eval_correct, feed_dict=feed_dict)
        accuracy = float(count) / float(len(data.y_vali))
        accuracy_accountant.append(accuracy)

        print_loss_and_accuracy(global_loss, accuracy)

        if delta_accountant[-1] > privacy_agent.get_bound() or math.isnan(delta_accountant[-1]):
            print('************** The last step exhausted the privacy budget **************')
            if not math.isnan(delta_accountant[-1]):
                try:
                    None
                finally:
                    save_progress(save_dir, model, delta_accountant + [float('nan')],
                                  accuracy_accountant + [float('nan')], privacy_agent, FLAGS)
                return accuracy_accountant, delta_accountant, model
        else:
            try:
                None
            finally:
                save_progress(save_dir, model, delta_accountant, accuracy_accountant, privacy_agent, FLAGS)

        ############################################################################################################
        # Start of a new communication round

        real_round = real_round + 1

        print_new_comm_round(real_round)

        if FLAGS.priv_agent:
            m = int(privacy_agent.get_m(int(real_round)))
            sigma = privacy_agent.get_Sigma(int(real_round))

        print('Clients participating: ' + str(m))

        # Randomly choose a total of m (out of n) client-indices that participate in this round
        perm = np.random.permutation(FLAGS.n)

        # Use the first m entries of the permuted list to decide which clients (and their sets) will participate in
        # this round. participating_clients is therefore a nested list of length m.
        s = perm[0:m].tolist()
        participating_clients = [data.client_set[k] for k in s]

        # For each client c (out of the m chosen ones):
        for c in range(m):
            # Assign the global model and set the global step.
            sess.run(assignments + [set_global_step], feed_dict=model)

            # allocate a list, holding data indices associated to client c and split into batches.
            data_ind = np.split(np.asarray(participating_clients[c]), FLAGS.b, 0)

            # e = Epoch
            for e in range(int(FLAGS.e)):
                for step in range(len(data_ind)):
                    # increase the global_step count (it's used for the learning rate.)
                    real_step = sess.run(increase_global_step)
                    # batch_ind holds the indices of the current batch
                    batch_ind = data_ind[step]

                    # Fill a feed dictionary with the actual set of data and labels using the data and labels associated
                    # to the indices stored in batch_ind:
                    feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j) for j in batch_ind]],
                                 str(label_placeholder.name): label_set_asarray[[int(j) for j in batch_ind]]}

                    # Run one optimization step.
                    _ = sess.run([train_op], feed_dict=feed_dict)

            if c == 0:
                # If we just trained the first client in a comm_round, We override the old weights_accountant
                weights_accountant = WeightsAccountant(sess, model, sigma, real_round)
            else:
                # Allocate the client update, if this is not the first client in a communication round
                weights_accountant.allocate(sess)

        # End of a communication round
        ############################################################################################################

        print('......Communication round %s completed' % str(real_round))
        # Compute a new model according to the updates and the Gaussian mechanism specifications from FLAGS
        model, delta = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, computed_deltas)
        delta_accountant.append(delta)

        # Set the global_step to the current step of the last client, such that the next clients can feed it into
        # the learning rate.
        model['global_step_placeholder:0'] = real_step

        # PRINT the progress and stage of affairs.
        print(' - Epsilon-Delta Privacy:' + str([FLAGS.eps, delta]))

        if save_params:
            weights_accountant.save_params(save_dir)

    return [], [], []
