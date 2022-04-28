import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np
import logging
import codecs

from transformer.common_attention import *
from transformer.common_layers import layer_norm, conv_hidden_relu, smoothing_cross_entropy

from utils import deal_generated_samples
from utils import score
from utils import remove_pad_tolist

from utils import summarize_sequence
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



class Model():
    def __init__(self, config, graph=None, sess=None):
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        if sess is None:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.allow_soft_placement = True
            self.sess = tf.Session(config=sess_config, graph=self.graph)
        else:
            self.sess = sess

        self.config = config
        self.filter_sizes = [i for i in range(1, config.train.max_length, 4)]
        self.num_filters = [(50 + i * 10) for i in range(1, config.train.max_length, 4)]
        self.num_filters_total = sum(self.num_filters)
        self.logger = logging.getLogger("model")
        self.prepared = False
        self.summary = True
        self.context_num = 2
        with self.graph.as_default():
            self.p = tf.Variable(0.5, dtype=tf.float32, name="gate_portion")

        self.tvars = tf.trainable_variables()

    def prepare(self, is_training):
        assert not self.prepared
        self.is_training = is_training
        devices = self.config.train.devices if is_training else self.config.test.devices
        # self.devices = ['/gpu:' + i for i in devices.split(',')] or ['/cpu:0']
        self.devices=['/gpu:0']
        # self.devices = ['/cpu:0']
        # If we have multiple devices (typically GPUs), we set /cpu:0 as the sync device.
        #self.sync_device = self.devices[0] if len(self.devices) == 1 else '/cpu:0'
        self.sync_device = '/cpu:0'

        if is_training:
            with self.graph.as_default():
                with tf.device(self.sync_device):
                    self.global_step = tf.get_variable('global_step', [], tf.int32, trainable=False, initializer=tf.zeros_initializer)
                    #self.global_step = tf.train.get_or_create_global_step()
                    self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)
                    if self.config.train.optimizer == "adam":
                        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    elif self.config.train.optimizer == 'adam_decay':
                        self.learning_rate = learning_rate_decay(self.config, self.global_step)
                        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.9, 0.99, 1e-9)
                    elif self.config.train.optimizer == "sgd":
                        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                    elif self.config.train.optimizer == 'mom':
                        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
                    else:
                        logging.info("No optimizer type")
                        raise ValueError
        self.initializer = init_ops.variance_scaling_initializer(scale=1, mode='fan_avg', distribution='uniform')
        self.prepared = True

    def build_train_model(self, pretrain=True):
        self.prepare(is_training=True)  ## for score
        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(tf.int32, [None, None, None], "src_pl") # B x N x L
                self.dst_pl = tf.placeholder(tf.int32, [None, None, None], "dst_pl") # B x N x L
  
                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                acc_list, loss_list, gv_list = [], [], []
                acc_list_s, loss_list_s, gv_list_s = [], [], []
                preds_list = []
                portion_list = []
                for i, (X, Y, device) in enumerate(zip(Xs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self.logger.info('Build model on %s.' % device)
                        X_shape, Y_shape = tf.shape(X), tf.shape(Y) # BxNxL
                        X_s = tf.reshape(X, [-1, X_shape[-1]])  #X_s (BxN)xL
                        Y_s = tf.reshape(Y, [-1, Y_shape[-1]])  #X_s (BxN)xL
                        self.batch_size = X_shape[0] 

                        print("batch_size:",self.batch_size,'\n')

                        print("sen_enc_inp:",X_s.shape)

                        sentence_encoder_output = self.encoder(X_s, "sentence_encoder", tf.AUTO_REUSE)
                        print("sen_enc_out:",sentence_encoder_output.shape,'\n')

                        context_encoder_output = self.encoder_c(sentence_encoder_output, "context_encoder", tf.AUTO_REUSE)
                        print("doc_enc_out:",context_encoder_output.shape,'\n')

                        decoder_output= self.decoder_context(shift_right(Y_s), sentence_encoder_output, context_encoder_output, "decoder", tf.AUTO_REUSE)
                        acc, loss, preds = self.train_output(decoder_output, Y_s, scope="output_c", reuse=tf.AUTO_REUSE) # B x (N x L) x D
                        acc_list.append(acc)
                        loss_list.append(loss)
                        gv_list.append([g_and_v for g_and_v in self.optimizer.compute_gradients(loss) if g_and_v[0] is not None])
                
                self.acc = tf.reduce_mean(acc_list)
                self.loss = tf.reduce_mean(loss_list)
                grads_and_vars = average_gradients(gv_list)

                self.preds_list = preds_list

                if self.summary:
                    for g, v in grads_and_vars:
                        tf.summary.histogram('variables/'+v.name.split(":")[0], v)
                        tf.summary.histogram('gradients/'+g.name.split(":")[0], g)

                grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars], clip_norm=self.config.train.grads_clip)
                grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
                self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)


                if pretrain:
                    tf.summary.scalar('acc', self.acc)
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.scalar('learning_rate', self.learning_rate)
                    tf.summary.scalar('grads_and_norm', self.grads_norm)
                    self.summary_op = tf.summary.merge_all()

    def build_generate(self, max_len, generate_devices, optimizer='rmsprop'):
        with self.graph.as_default():
            with tf.device(self.sync_device):
                logging.info("using "+optimizer+" for g_loss")
                
                if optimizer == 'adam':
                    logging.info("using adam for g_loss")
                    optimizer = tf.train.AdamOptimizer(self.config.generator.learning_rate)
                elif optimizer == 'adadelta':
                    logging.info("using adadelta for g_loss")
                    optimizer = tf.train.AdadeltaOptimizer()
                elif optimizer == "sgd":
                    logging.info("using adam decay for g_loss")
                    optimizer = tf.train.GradientDescentOptimizer(self.config.generator.learning_rate)
                else:
                    logging.info("using rmsprop for g_loss")
                    optimizer = tf.train.RMSPropOptimizer(self.config.train.learning_rate)
                

                src_pl = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='gene_src_pl') # B x N x L
                dst_pl = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='gene_dst_pl') # B x N x L
                reward_pl = tf.placeholder(dtype=tf.float32, shape=[None, None], name='gene_reward')


                generate_devices = ['/gpu:' + i for i in generate_devices.split(',')] or ['/cpu:0']
                # generate_devices = ['/cpu:0']


                Xs = split_tensor(src_pl, len(generate_devices)) 
                Ys = split_tensor(dst_pl, len(generate_devices))
                Rs = split_tensor(reward_pl, len(generate_devices))

                batch_size_list = [tf.shape(X)[0] for X in Xs]  ##?? 
                con_len_list = [tf.shape(X)[1] for X in Xs]     ##??

                encoder_outputs = [None] * len(generate_devices)
                context_outputs = [None] * len(generate_devices)
                for i, (X, device) in enumerate(zip(Xs, generate_devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self.logger.info('Build generate model on %s'%device)
                        X_shape = tf.shape(X)
                        X_s = tf.reshape(X, [-1, X_shape[-1]])
                        sentence_encoder_output = self.encoder(X_s, "sentence_encoder", reuse=True)
                        context_encoder_output = self.encoder_c(sentence_encoder_output, "context_encoder", reuse=True)
                        encoder_outputs[i] = sentence_encoder_output
                        context_outputs[i] = context_encoder_output

                def recurrency(i, cur_y, encoder_output, context_output):
                    #decoder_output = self.decoder_context4(shift_right(cur_y), encoder_output, context_output, "decoder", reuse=True)
                    # (B x N) x L x D
                    decoder_output = self.decoder_context(shift_right(cur_y),encoder_output ,context_output, "decoder", reuse=True)

                    next_logits = top(body_output=decoder_output,
                                      vocab_size=self.config.dst_vocab_size,
                                      dense_size=self.config.hidden_units,
                                      shared_embedding=self.config.train.shared_embedding,
                                      reuse=True) # (BxN)x?xD

                    next_logits = next_logits[:, i, :] # (BxN)x1xD
                    next_logits = tf.reshape(next_logits, [-1, self.config.dst_vocab_size]) # (BxN)xD
                    next_probs = tf.nn.softmax(next_logits)
                    next_sample = tf.argmax(next_probs, 1) # (BxN)
                    next_sample = tf.expand_dims(next_sample, -1) # (BxN)x1
                    next_sample = tf.to_int32(next_sample)# (BxN)x1
                    next_y = tf.concat([cur_y[:, :i], next_sample], axis=1) # (BxN)xi
                    next_y = tf.pad(next_y, [[0, 0], [0, max_len - 1 - i]]) # (BxN)xL
                    next_y.set_shape([None, max_len])
                    return i + 1, next_y, encoder_output, context_output

                total_results = [None]*len(generate_devices)
                for i, (device, batch_size, con_len) in enumerate(zip(generate_devices, batch_size_list, con_len_list)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        initial_y = tf.zeros((batch_size*con_len, max_len), dtype=tf.int32)  ##begin with <s>
                        initial_i = tf.constant(0, dtype=tf.int32)
                        _, sample_result, _, _ = tf.while_loop(
                            cond=lambda a, _1, _2, _3: a < max_len,
                            body=recurrency,
                            loop_vars=(initial_i, initial_y, encoder_outputs[i], context_outputs[i]),
                            shape_invariants=(initial_i.get_shape(), initial_y.get_shape(), encoder_outputs[i].get_shape(), context_outputs[i].get_shape())
                        )
                        #sample_result = tf.reshape(sample_result, [batch_size, -1])
                        total_results[i] = sample_result # (BxN)xL

                generate_result = tf.concat(total_results, axis=0) # (BxN)xL
                ### generate over ###
                loss_list = []
                train_list = []
                acc_list = []
                grads_and_vars_list = []
                for i, (Y, reward, device) in enumerate(zip(Ys, Rs, generate_devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        Y_shape = tf.shape(Y)
                        Y_s = tf.reshape(Y, [-1, Y_shape[-1]]) # (BxN)xL
                        Y_c = tf.reshape(Y, [Y_shape[0], -1]) # Bx(NxL)
                        decoder_output = self.decoder_context(shift_right(Y_s), encoder_outputs[i], context_outputs[i], scope="decoder", reuse=True)
                        # (B x N) x L x D
                        g_loss = self.gan_output(decoder_output, Y_s, reward, reuse=True)
                        acc, t_loss, _ = self.train_output(decoder_output, Y_s, reuse=True)
                        grads_and_vars = self.optimizer.compute_gradients(g_loss)
                        loss_list.append(g_loss)
                        acc_list.append(acc)
                        train_list.append(t_loss)
                        grads_and_vars_list.append([g_and_v for g_and_v in grads_and_vars if g_and_v[0] is not None])
                grads_and_vars = average_gradients(grads_and_vars_list)
                loss = tf.reduce_mean(loss_list)
                accuracy = tf.reduce_mean(acc_list)
                train_loss = tf.reduce_mean(train_list)
                g_optm = self.optimizer.apply_gradients(grads_and_vars, self.global_step)

                self.generate_x = src_pl
                self.generate_y = dst_pl
                self.generate_reward = reward_pl

                self.generate_sample = generate_result # (BxN)xL
                self.generate_g_loss = loss
                self.generate_g_grad = grads_and_vars
                self.generate_g_optm = g_optm
                self.generate_t_loss = train_loss
                self.generate_accuracy = accuracy

    def build_rollout_generate(self, max_len, roll_generate_devices):
        src_pl = tf.placeholder(dtype=tf.int32, shape=[None, None], name='gene_src_pl')
        dst_pl = tf.placeholder(dtype=tf.int32, shape=[None, None], name='gene_dst_pl')
        given_num_pl = tf.placeholder(dtype=tf.int32, shape=[], name='give_num_pl')
        src_p_pl = tf.placeholder(tf.int32, [None, None, None], "gene_src_p_pl")
        dst_p_pl = tf.placeholder(tf.int32, [None, None, None], "gene_dst_p_pl")

        devices = ['/gpu:' + i for i in roll_generate_devices.split(',')] or ['/cpu:0']
        Xs = split_tensor(src_pl, len(devices))
        Ys = split_tensor(dst_pl, len(devices))
        X_ps = split_tensor(src_p_pl, len(devices))
        Y_ps = split_tensor(dst_p_pl, len(devices))

        Ms = [given_num_pl] * len(devices)

        batch_size_list = [tf.shape(X)[0] for X in Xs]

        encoder_outputs = [None] * len(devices)
        context_outputs = [None] * len(devices)
        for i, (X, Xp, device) in enumerate(zip(Xs, X_ps, devices)):
            with tf.device(lambda op: self.choose_device(op, device)):
                self.logger.info('Build roll generate model on %s' % device)
                Xc = tf.transpose(Xp, [1, 0, 2])
                context_s_output = self.encoder_doc(Xc, tf.shape(Xc)[0], scope="context_s_encoder", reuse=True)
                encoder_output = self.decoder(X, context_s_output, scope="encoder", reuse=True)

                encoder_outputs[i] = encoder_output
                context_outputs[i] = context_s_output

        def recurrency(given_num, given_y, encoder_output, context_output):
            decoder_output = self.decoder_context(shift_right(given_y), encoder_output, context_output, scope="decoder", reuse=True)
            next_logits = top(body_output=decoder_output,
                         vocab_size = self.config.dst_vocab_size,
                         dense_size = self.config.hidden_units,
                         shared_embedding = self.config.train.shared_embedding,
                         reuse=True)
            next_logits = next_logits[:, given_num, :]
            next_probs = tf.nn.softmax(next_logits)
            log_probs = tf.log(next_probs)
            next_sample = tf.multinomial(log_probs, 1)
            next_sample_flat = tf.cast(next_sample, tf.int32)
            next_y = tf.concat([given_y[:, :given_num], next_sample_flat], axis=1)
            next_y = tf.pad(next_y, [[0,0], [0, max_len-given_num-1]])
            next_y.set_shape([None, max_len])
            return given_num + 1, next_y, encoder_output, context_output

        total_results = [None] * len(devices)
        for i, (Y, given_num, device) in enumerate(zip(Ys, Ms, devices)):
            with tf.device(lambda op: self.choose_device(op, device)):
                given_y = Y[:, :given_num]
                init_given_y = tf.pad(given_y, [[0,0], [0, (max_len-given_num)]])
                _, roll_sample, _, _ = tf.while_loop(
                    cond=lambda a, _1, _2, _3: a < max_len,
                    body=recurrency,
                    loop_vars=(given_num, init_given_y, encoder_outputs[i], context_s_output[i]),
                    shape_invariants=(
                        given_num.get_shape(), init_given_y.get_shape(), encoder_outputs[i].get_shape(), context_s_output[i].get_shape())
                )
                total_results[i] = roll_sample
        sample_result = tf.concat(total_results, axis=0)

        self.roll_x = src_pl
        self.roll_y = dst_pl
        self.roll_xp = src_p_pl
        self.roll_yp = dst_p_pl
        self.roll_given_num = given_num_pl
        self.roll_y_sample = sample_result

    def generate_step(self, sentence_x):
        feed={self.generate_x:sentence_x}
        y_sample = self.sess.run(self.generate_sample, feed_dict=feed)
        return y_sample

    def generate_step_and_update(self, sentence_x, sentence_y, reward):
        feed={self.generate_x:sentence_x, self.generate_y:sentence_y, self.generate_reward:reward}
        gl = tf.summary.scalar('loss', self.generate_g_loss)
        self.merge_summary = tf.summary.merge([gl])
        loss, ms, _, _, lr, acc, t_loss = self.sess.run([self.generate_g_loss, self.merge_summary, self.generate_g_grad, self.generate_g_optm,
                                        self.learning_rate, self.generate_accuracy, self.generate_t_loss], feed_dict=feed)
        return loss, ms, lr, acc, t_loss

    def generate_and_save(self, data_util, infile, generate_batch, outfile):
        outfile = codecs.open(outfile, 'w', 'utf-8')
        for batch in data_util.get_test_batches_doc(infile, generate_batch):
            num = np.shape(batch)[1]   #batch.shape (1,13,50)
            feed = {self.generate_x: batch, self.src_pl:batch}
            out_generate = self.sess.run(self.generate_sample, feed_dict=feed)  #generate_sample (192,50) ##1 3 1 3 
            out_generate_dealed, _ = deal_generated_samples(out_generate, data_util.dst2idx)   ## out_generate(192,50) 1 0 0 
            y_strs = data_util.indices_to_words_del_pad(out_generate_dealed, 'dst')
            count = 0
            for y_str in y_strs:
                count += 1
                if count % num == 0:
                    outfile.write(y_str+'\n')
                else:
                    outfile.write(y_str + '||')
        outfile.close()

    def get_reward(self, x, x_to_maxlen, y_sample, y_sample_mask, rollnum, disc, max_len=50, bias_num=None, data_util=None):
        rewards = []
        x_to_maxlen = np.transpose(x_to_maxlen)

        for i in range(rollnum):
            for given_num in np.arange(1, max_len, dtype='int32'):
                feed = {self.roll_x:x, self.roll_y:y_sample, self.roll_given_num:given_num}
                output = self.sess.run(self.roll_y_sample, feed_dict=feed)
                output = output*y_sample_mask
                output = np.transpose(output)
                feed = {disc.dis_input_x: output, disc.dis_input_xs: x_to_maxlen,
                        disc.dis_dropout_keep_prob: 1.0}
                ypred_for_auc = self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)

                ypred = np.array([item[1] for item in ypred_for_auc])
                # ypred = np.array([item[0] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred
            y_sample_transpose = np.transpose(y_sample)
            feed = {disc.dis_input_x: y_sample_transpose, disc.dis_input_xs: x_to_maxlen,
                    disc.dis_dropout_keep_prob: 1.0}

            ypred_for_auc = self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)
            ypred = np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[max_len - 1] += ypred

        rewards = np.transpose(np.array(rewards))  ## now rewards: batch_size * max_len

        if bias_num is None:
            rewards = rewards * y_sample_mask
            rewards = rewards / (1.0 * rollnum)
        else:
            bias = np.zeros_like(rewards)
            bias += bias_num * rollnum
            rewards = rewards - bias
            rewards = rewards * y_sample_mask
            rewards = rewards / (1.0 * rollnum)
        return  rewards

    def get_reward_Obinforced(self, x, x_to_maxlen, y_sample, y_sample_mask, y_ground, rollnum, disc, max_len=50,
                              bias_num=None, data_util=None, namana=0.7):
        rewards = []
        BLEU = []

        y_ground_removed_pad_list = remove_pad_tolist(y_ground)

        x_to_maxlen = np.transpose(x_to_maxlen)
        y_sample_transpose = np.transpose(y_sample)

        for i in range(rollnum):
            for give_num in np.arange(1, max_len, dtype='int32'):
                feed = {self.roll_x: x, self.roll_y: y_sample, self.roll_given_num: give_num}
                output = self.sess.run(self.roll_y_sample, feed_dict=feed)
                output = output * y_sample_mask
                output_removed_pad_list = remove_pad_tolist(output)
                output = np.transpose(output)
                feed = {disc.dis_input_x: output, disc.dis_input_xs: x_to_maxlen,
                        disc.dis_dropout_keep_prob: 1.0}
                ypred_for_auc = self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)

                BLEU_predict = []
                for hypo_tokens, ref_tokens in zip(output_removed_pad_list, y_ground_removed_pad_list):
                    BLEU_predict.append(score(ref_tokens, hypo_tokens))
                BLEU_predict = np.array(BLEU_predict)

                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                    BLEU.append(BLEU_predict)
                else:
                    rewards[give_num - 1] += ypred
                    BLEU[give_num - 1] += BLEU_predict

        rewards = np.transpose(np.array(rewards)) # batch_size x max_len
        BLEU = np.transpose(np.array(BLEU))
        if bias_num is None:
            rewards = rewards * y_sample_mask
            rewards = rewards / (1.0 * rollnum)
        else:
            bias = np.zeros_like(rewards)
            bias += bias_num * rollnum
            rewards = rewards - bias
            rewards = namana * rewards + (1 - namana) * BLEU
            rewards = rewards * y_sample_mask
            rewards = rewards / (1.0 * rollnum)
        return rewards

    def init_and_restore(self, modelFile=None):
        params = tf.trainable_variables()
        print("params:",params)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(params)

        self.sess.run(init_op)
        self.saver = saver
        if modelFile is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.train.logdir))
        else:
            # self.saver = tf.train.import_meta_graph('./pretrain/model_generator_fren/model_step_18000.meta')
            self.saver.restore(self.sess, modelFile)


        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            print("  name = %s, shape = %s" % (variable.name, variable.shape))

            for dim in shape:
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters     
        # print(total_parameters)

    def build_test_model(self):
        """Build model for testing."""

        self.prepare(is_training=False)

        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='src_pl')
                self.dst_pl = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='dst_pl')
                # self.decoder_input = shift_right(self.dst_pl)
                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                dec_inputs = split_tensor(self.dst_pl, len(self.devices))

                encoder_output_list = []
                context_output_list = []
                for i, (X, device) in enumerate(zip(Xs, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        X_shape = tf.shape(X) # BxNxL
                        X_s = tf.reshape(X, [-1, X_shape[-1]])  #X_s (BxN)xL
                        self.batch_size = X_shape[0]

                        sentence_encoder_output = self.encoder(X_s, "sentence_encoder", tf.AUTO_REUSE)
                        context_encoder_output = self.encoder_c(sentence_encoder_output, "context_encoder", tf.AUTO_REUSE)
                        
                        encoder_output_list.append(sentence_encoder_output)
                        context_output_list.append(context_encoder_output)
                self.encoder_output = tf.concat(encoder_output_list, axis=0)
                self.context_output = tf.concat(context_output_list, axis=0)

                # Decode
                enc_outputs = split_tensor(self.encoder_output, len(self.devices))
                context_outputs = split_tensor(self.context_output, len(self.devices))
                preds_list, k_preds_list, k_scores_list = [], [], []
                self.loss_sum = 0.0
                for i, (X, enc_output, context_output, dec_input, Y, device) in enumerate(
                        zip(Xs, enc_outputs, context_outputs, dec_inputs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self.logger.info('Build model on %s.' % device)
                        dec_input_shape =  tf.shape(dec_input) # BxNxL
                        Y_shape = tf.shape(Y)   
                        dec_input_s = tf.reshape(dec_input, [-1, dec_input_shape[-1]])  #X_s (BxN)xL (1x13)x50
                        Y_s = tf.reshape(Y, [-1, Y_shape[-1]])
                        decoder_output= self.decoder_context(shift_right(dec_input_s), enc_output, context_output, "decoder", reuse=tf.AUTO_REUSE)
                        
                        # Predictions
                        preds, k_preds, k_scores = self.test_output(decoder_output, reuse=tf.AUTO_REUSE)
                        preds_list.append(preds)
                        k_preds_list.append(k_preds)
                        k_scores_list.append(k_scores)
                        # Loss

                        loss = self.test_loss(decoder_output, Y_s, reuse=True)
                        self.loss_sum += loss

                self.preds = tf.concat(preds_list, axis=0)
                self.k_preds = tf.concat(k_preds_list, axis=0)
                self.k_scores = tf.concat(k_scores_list, axis=0)

    def choose_device(self, op, device):
        """Choose a device according the op's type."""
        if op.type.startswith('Variable'):
            return self.sync_device
        return device

    def encoder(self, encoder_input, scope, reuse):
        encoder_padding = tf.equal(encoder_input, 0)
        """Transformer encoder."""
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = bottom(encoder_input,
                                    vocab_size=self.config.src_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)
            encoder_output += add_timing_signal_1d(encoder_output)
            encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate, training=self.is_training)
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output, # [batch, length_q, channels]
                                                  memory=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding),
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                # Mask padding part to zeros.
            encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
            return encoder_output

    def encoder_s(self, encoder_input, context_input, scope, cnn=False, reuse=None):
        input_shape = tf.shape(encoder_input)
        if not cnn:
            context = tf.reshape(context_input, [input_shape[0], input_shape[1], self.config.hidden_units])
        else:
            context = tf.reshape(context_input, [input_shape[0], 1, self.config.hidden_units])
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context), axis=-1), 0.0)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        encoder_padding = tf.equal(encoder_input, 0)
        """Transformer encoder."""
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = bottom(encoder_input,
                                    vocab_size=self.config.src_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)
            # positional encoding
            encoder_output += add_timing_signal_1d(encoder_output)
            encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate, training=self.is_training)
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output, 
                                                  memory=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding),
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
                    encoder_output_s = encoder_output
                    # Multihead Attention (vanilla attention)
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output,
                                                  memory=context,
                                                  bias=context_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                # Mask padding part to zeros.
            encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
            return encoder_output, encoder_output_s

    def encoder_s_c(self, encoder_input, scope, reuse=None):
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_input), axis=-1), 0.0)
        """Transformer encoder."""
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = encoder_input
            # positional encoding
            #encoder_output += add_timing_signal_1d(encoder_output)
            #encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate, training=self.is_training)
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (vanilla attention)
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output,
                                                  memory=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding),
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="sentence_context_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                # Mask padding part to zeros.
            encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
            return encoder_output

    def encoder_s_cnn(self, encoder_input, context_input, scope, cnn=False, reuse=None):
        input_shape = tf.shape(encoder_input)
        if not cnn:
            context = tf.reshape(context_input, [input_shape[0], input_shape[1], self.config.hidden_units])
        else:
            context = tf.reshape(context_input, [input_shape[0], 1, self.config.hidden_units])
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context), axis=-1), 0.0)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_input), axis=-1), 0.0)
        """Transformer encoder."""
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = encoder_input
            # positional encoding
            encoder_output += add_timing_signal_1d(encoder_output)
            encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate, training=self.is_training)
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (vanilla attention)
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output,
                                                  memory=context,
                                                  bias=context_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                # Mask padding part to zeros.
            encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
            return encoder_output

    def encoder_d(self, encoder_input, scope, reuse):
        encoder_padding = tf.equal(encoder_input, 0)
        self_attention_bias = attention_bias_lower_traingle_l(tf.shape(encoder_input)[1], self.config.train.max_length)
        """Transformer encoder."""
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = bottom(encoder_input,
                                    vocab_size=self.config.src_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)
            encoder_output += add_timing_signal_1d(encoder_output)
            encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate, training=self.is_training)
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output,
                                                  memory=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding)+self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                # Mask padding part to zeros.
            encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
            return encoder_output

    def encoder_c(self, encoder_input, scope, reuse):
        """Transformer encoder."""
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = tf.transpose(encoder_input, [0, 2, 1])
            encoder_output = tf.reshape(encoder_output, [-1, self.config.hidden_units, self.config.train.max_length])
            encoder_output = tf.layers.dense(encoder_output, 1, activation="relu") # (bxn)xdx1
            encoder_output = tf.squeeze(encoder_output, -1)
            encoder_output = tf.reshape(encoder_output, [self.batch_size, -1, self.config.hidden_units])
            encoder_output += add_timing_signal_1d(encoder_output)
            encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            
            print("doc-enc-input:",encoder_output.shape,'\n')

            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output,
                                                  memory=None,
                                                  bias=None,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                # Mask padding part to zeros.
            #encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
            return encoder_output

    def decoder(self, decoder_input, encoder_output, scope, reuse):
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
            encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
            decoder_output = target(decoder_input,
                                    vocab_size=self.config.dst_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)
            # Positional Encoding
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  output_depth=self.config.hidden_units,
                                                  name="decoder_self_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Multihead Attention (vanilla attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=encoder_output,
                                                  bias=encoder_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
                    # Feed Forward
                    decoder_output = residual(decoder_output,
                                              conv_hidden_relu(
                                                  decoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
        return decoder_output

    def decoder_context(self, decoder_input, encoder_output, context_output, scope, reuse):
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            batch_size = tf.shape(context_output)[0]  # BxNxD
            decoder_output = target(decoder_input,
                                    vocab_size=self.config.dst_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)  # (BxN)xLxD

            # Positional Encoding
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  output_depth=self.config.hidden_units,
                                                  name="decoder_self_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training) # (BxN)x?xD
  
                    with tf.variable_scope("context_sentence"):
                        decoder_output_c = context_output
                        decoder_output_c = tf.reshape(decoder_output_c, [-1, 1, self.config.hidden_units])
                        decoder_output_c = tf.transpose(decoder_output_c, [0, 2, 1])
                        decoder_output_c = tf.layers.dense(decoder_output_c, self.config.train.max_length, activation='relu')
                        decoder_output_c = tf.transpose(decoder_output_c, [0, 2, 1]) # (BxN)xLxD

                    with tf.variable_scope("sentence_attention"):
                        # Multihead Attention (vanilla attention)

                        decoder_padding = tf.equal(tf.reduce_sum(tf.abs(decoder_output_c), axis=-1), 0.0)
                        decoder_attention_bias = attention_bias_ignore_padding(decoder_padding)
                        encoder_attention = residual(encoder_output,
                                                  multihead_attention(
                                                      query=encoder_output,
                                                      memory=decoder_output_c,
                                                      bias=decoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="encoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training) # (BxN)x?xD

                        with tf.variable_scope("ff"):
                            encoder_attention = residual(encoder_attention,
                                                        conv_hidden_relu(
                                                            encoder_attention,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)

                        #c = tf.constant(1.0, dtype=tf.float32)
                        #encoder_attention = self.p*encoder_output + (c-self.p)*decoder_output_c
                        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_attention), axis=-1), 0.0)
                        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_attention,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD


                    with tf.variable_scope("feedforward"):
                        decoder_output = residual(decoder_output,
                                                    conv_hidden_relu(
                                                        decoder_output,
                                                        hidden_size=4 * self.config.hidden_units,
                                                        output_size=self.config.hidden_units,
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)
        return decoder_output

    def decoder_context4(self, decoder_input, encoder_output, context_output, scope, reuse):
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            batch_size = tf.shape(context_output)[0]  # BxNxD
            decoder_output = target(decoder_input,
                                    vocab_size=self.config.dst_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)  # (BxN)xLxD

            # Positional Encoding
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  output_depth=self.config.hidden_units,
                                                  name="decoder_self_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training) # (BxN)x?xD
                    '''
                    with tf.variable_scope("context_attention"):
                        decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
                        # Multihead Attention (vanilla attention)
                        decoder_padding = tf.equal(tf.reduce_sum(tf.abs(decoder_output_c), axis=-1), 0.0)
                        decoder_attention_bias = attention_bias_ignore_padding(decoder_padding)
                        decoder_output_c = residual(context_output,
                                                  multihead_attention(
                                                      query=context_output,
                                                      memory=decoder_output_c,
                                                      bias=decoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="context_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # BxNxD

                        with tf.variable_scope("ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)
                    '''

                    with tf.variable_scope("context_sentence"):
                        decoder_output_c = context_output
                        decoder_output_c = tf.reshape(decoder_output_c, [-1, 1, self.config.hidden_units])
                        decoder_output_c = tf.transpose(decoder_output_c, [0, 2, 1])
                        decoder_output_c = tf.layers.dense(decoder_output_c, self.config.train.max_length, activation='relu')
                        decoder_output_c = tf.transpose(decoder_output_c, [0, 2, 1]) # (BxN)xLxD
                        context_padding = tf.equal(tf.reduce_sum(tf.abs(decoder_output_c), axis=-1), 0.0)
                        context_attention_bias = attention_bias_ignore_padding(context_padding)
                        decoder_output = residual(decoder_output,
                                                     multihead_attention(
                                                         query=decoder_output,
                                                         memory=decoder_output_c,
                                                         bias=context_attention_bias,
                                                         total_key_depth=self.config.hidden_units,
                                                         total_value_depth=self.config.hidden_units,
                                                         output_depth=self.config.hidden_units,
                                                         num_heads=self.config.num_heads,
                                                         dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                         name="encoder_vanilla_attention",
                                                         summaries=self.summary),
                                                     dropout_rate=self.config.residual_dropout_rate,
                                                     is_training=self.is_training)  # (BxN)x?xD
                        with tf.variable_scope("ff"):
                            decoder_output = residual(decoder_output,
                                                         conv_hidden_relu(
                                                             decoder_output,
                                                             hidden_size=4 * self.config.hidden_units,
                                                             output_size=self.config.hidden_units,
                                                             summaries=self.summary),
                                                         dropout_rate=self.config.residual_dropout_rate,
                                                         is_training=self.is_training)

                    with tf.variable_scope("sentence_attention"):
                        # Multihead Attention (vanilla attention)
                        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
                        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_output,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        with tf.variable_scope("ff"):
                                decoder_output = residual(decoder_output,
                                                            conv_hidden_relu(
                                                                decoder_output,
                                                                hidden_size=4 * self.config.hidden_units,
                                                                output_size=self.config.hidden_units,
                                                                summaries=self.summary),
                                                            dropout_rate=self.config.residual_dropout_rate,
                                                            is_training=self.is_training)

                        #c = tf.constant(1.0, dtype=tf.float32)
                        #encoder_attention = self.p*encoder_output + (c-self.p)*decoder_output_c

                    '''
                    with tf.variable_scope("final"):
                        final_padding = tf.equal(tf.reduce_sum(tf.abs(context_attention), axis=-1), 0.0)
                        final_attention_bias = attention_bias_ignore_padding(final_padding)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=context_attention,
                                                      bias=final_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        decoder_output = residual(decoder_output,
                                                    conv_hidden_relu(
                                                        decoder_output,
                                                        hidden_size=4 * self.config.hidden_units,
                                                        output_size=self.config.hidden_units,
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)
                    '''
        return decoder_output

    def decoder_context_new(self, decoder_input, encoder_output, context_output, scope, cnn, reuse):
        input_shape = tf.shape(encoder_output) #(BxN)xLxD
        batch_size = tf.shape(context_output)[0] #Bx(Nx?)xD
        if not cnn:
            context = tf.reshape(context_output, [input_shape[0], input_shape[1], self.config.hidden_units]) # （BxN)xLxD
        else:
            context = tf.reshape(context_output, [input_shape[0], 1, self.config.hidden_units])
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        #context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0) # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            if not cnn:
                self_attention_bias2 = attention_bias_lower_traingle_l(tf.shape(context_output)[1],
                                                                       self.config.train.max_length)
            else:
                self_attention_bias2 = attention_bias_lower_triangle(tf.shape(context_output)[1])
                self_attention_bias2 = tf.tile(self_attention_bias2, [1, 1, self.config.train.max_length, 1])
                #self_attention_bias2 = attention_bias_lower_traingle_cnn(self.config.train.max_length, tf.shape(context_output)[1])
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  output_depth=self.config.hidden_units,
                                                  name="decoder_self_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training) # (BxN)x?xD


                    # Multihead Attention (vanilla attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=encoder_output,
                                                  bias=encoder_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)  # (BxN)x?xD

                    with tf.variable_scope("sentence_ff"):
                        decoder_output = residual(decoder_output,
                                                  conv_hidden_relu(
                                                      decoder_output,
                                                      hidden_size=4 * self.config.hidden_units,
                                                      output_size=self.config.hidden_units,
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)
                    # context adjustment
                    decoder_output_c = tf.reshape(decoder_output,
                                                  [batch_size, -1, self.config.hidden_units])  # (BxN)xLxD


                    '''
                    decoder_output_c = residual(decoder_output_c,
                                                multihead_attention(
                                                    query=decoder_output_c,
                                                    memory=None,
                                                    bias=None,
                                                    total_key_depth=self.config.hidden_units,
                                                    total_value_depth=self.config.hidden_units,
                                                    output_depth=self.config.hidden_units,
                                                    num_heads=self.config.num_heads,
                                                    dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                    name="decoder_context_self_attention",
                                                    summaries=self.summary),
                                                dropout_rate=self.config.residual_dropout_rate,
                                                is_training=self.is_training)
                    '''


                    decoder_output_c = residual(decoder_output_c,
                                              multihead_attention(
                                                  query=decoder_output_c,
                                                  memory=context_output,
                                                  bias=context_attention_bias+self_attention_bias2,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_context_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    decoder_output_c = tf.reshape(decoder_output_c, [-1, self.config.train.max_length, self.config.hidden_units])

                    '''
                    decoder_output_c = residual(decoder_output_c,
                                              multihead_attention(
                                                  query=decoder_output_c,
                                                  memory=encoder_output,
                                                  bias=encoder_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)  # (BxN)x?xD
                    '''

                    with tf.variable_scope("context_ff"):
                        decoder_output_c = residual(decoder_output_c,
                                                  conv_hidden_relu(
                                                      decoder_output_c,
                                                      hidden_size=4 * self.config.hidden_units,
                                                      output_size=self.config.hidden_units,
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)

        return decoder_output_c, decoder_output

    def decoder_context_gate(self, decoder_input, encoder_output, context_output, document_attention, scope, reuse):
        input_shape = tf.shape(encoder_output) #(BxN)xLxD
        batch_size = tf.shape(context_output)[0] #Bx(Nx?)xD

        context_attention = tf.reshape(context_output, [input_shape[0], input_shape[1], self.config.hidden_units]) # （BxN)xLxD
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_attention), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        #context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0) # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            # context adjustment
            #decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            #decoder_output_c += add_timing_signal_1d(decoder_output_c)
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            #self_attention_bias2 = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length)
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    with tf.variable_scope("sentence_level"):
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=None,
                                                      bias=self_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      output_depth=self.config.hidden_units,
                                                      name="decoder_self_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training) # (BxN)x?xD

                        # Multihead Attention context
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=context_attention,
                                                      bias=context_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_context_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])

                        # Multihead Attention (vanilla attention)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_output,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD



                        with tf.variable_scope("sentence_ff"):
                            decoder_output = residual(decoder_output,
                                                      conv_hidden_relu(
                                                          decoder_output,
                                                          hidden_size=4 * self.config.hidden_units,
                                                          output_size=self.config.hidden_units,
                                                          summaries=self.summary),
                                                      dropout_rate=self.config.residual_dropout_rate,
                                                      is_training=self.is_training)

                    with tf.variable_scope("context_level"):
                        decoder_output_c = residual(decoder_output_c,
                                                  multihead_attention(
                                                      query=decoder_output_c,
                                                      memory=document_attention,
                                                      bias=None,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_context_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)

                        with tf.variable_scope("context_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)

                    with tf.variable_scope("gate_level"):
                        decoder_output_c = tf.reshape(decoder_output_c, [-1, self.config.train.max_length, self.config.hidden_units])
                        p = tf.Variable(0.5, dtype=tf.float32, name="gate_portion")
                        c = tf.constant(1.0, dtype=tf.float32)
                        decoder_output_c = (c-p)*decoder_output_c + p*decoder_output

        return decoder_output_c, decoder_output, p

    def decoder_context_gate_2(self, decoder_input, encoder_output, context_output, document_attention, scope, reuse):
        input_shape = tf.shape(encoder_output)  # (BxN)xLxD
        batch_size = tf.shape(context_output)[0]  # Bx(Nx?)xD


        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        # context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)  # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            # context adjustment
            # decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # decoder_output_c += add_timing_signal_1d(decoder_output_c)
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # self_attention_bias2 = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length)
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    with tf.variable_scope("sentence_level"):
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=None,
                                                      bias=self_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      output_depth=self.config.hidden_units,
                                                      name="decoder_self_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD



                        # Multihead Attention (vanilla attention)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_output,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        with tf.variable_scope("sentence_ff"):
                            decoder_output = residual(decoder_output,
                                                      conv_hidden_relu(
                                                          decoder_output,
                                                          hidden_size=4 * self.config.hidden_units,
                                                          output_size=self.config.hidden_units,
                                                          summaries=self.summary),
                                                      dropout_rate=self.config.residual_dropout_rate,
                                                      is_training=self.is_training)

                    with tf.variable_scope("context_level"):
                        decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
                        # Multihead Attention context
                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=document_attention,
                                                        bias=None,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)

                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=context_output,
                                                        bias=context_attention_bias,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_vanilla_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)  # (BxN)x?xD

                        with tf.variable_scope("context_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)

                    with tf.variable_scope("gate_level"):
                        decoder_output_c = tf.reshape(decoder_output_c,
                                                      [-1, self.config.train.max_length, self.config.hidden_units])
                        p = tf.Variable(0.5, dtype=tf.float32, name="gate_portion")
                        c = tf.constant(1.0, dtype=tf.float32)
                        decoder_output_c = (c - p) * decoder_output_c + p * decoder_output

        return decoder_output_c, decoder_output, p

    def decoder_context_gate_h(self, decoder_input, encoder_output, context_output, document_attention, scope, reuse):
        input_shape = tf.shape(encoder_output)  # (BxN)xLxD
        batch_size = tf.shape(context_output)[0]  # Bx(Nx?)xD


        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        # context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)  # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            # context adjustment
            # decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # decoder_output_c += add_timing_signal_1d(decoder_output_c)
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # self_attention_bias2 = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length)
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    with tf.variable_scope("sentence_level"):
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=None,
                                                      bias=self_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      output_depth=self.config.hidden_units,
                                                      name="decoder_self_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD



                        # Multihead Attention (vanilla attention)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_output,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        with tf.variable_scope("sentence_ff"):
                            decoder_output_s = residual(decoder_output,
                                                      conv_hidden_relu(
                                                          decoder_output,
                                                          hidden_size=4 * self.config.hidden_units,
                                                          output_size=self.config.hidden_units,
                                                          summaries=self.summary),
                                                      dropout_rate=self.config.residual_dropout_rate,
                                                      is_training=self.is_training)

                    with tf.variable_scope("context_level"):
                        # Multihead Attention context
                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=document_attention,
                                                        bias=None,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)

                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=context_output,
                                                        bias=context_attention_bias,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_vanilla_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)  # (BxN)x?xD
                        '''
                        with tf.variable_scope("context_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)
                        '''

                    with tf.variable_scope("gate_level"):
                        decoder_output_c = tf.reshape(decoder_output_c,
                                                      [-1, self.config.train.max_length, self.config.hidden_units])
                        c = tf.constant(1.0, dtype=tf.float32)
                        decoder_output_c = (c - self.p) * decoder_output_c + self.p * decoder_output
                        with tf.variable_scope("gate_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)


        return decoder_output_c, decoder_output_s, self.p

    def decoder_context_gate_h_pool(self, decoder_input, encoder_output, context_output, document_attention, table, scope, reuse):
        input_shape = tf.shape(encoder_output)  # (BxN)xLxD
        batch_size = tf.shape(context_output)[0]  # Bx(Nx?)xD


        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        # context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)  # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            # context adjustment
            # decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # decoder_output_c += add_timing_signal_1d(decoder_output_c)
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            #decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # self_attention_bias2 = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length)
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    with tf.variable_scope("sentence_level"):
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=None,
                                                      bias=self_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      output_depth=self.config.hidden_units,
                                                      name="decoder_self_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units]) # Bx(Nx?)xD

                        # Multihead Attention (vanilla attention)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_output,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD
                        '''
                        with tf.variable_scope("sentence_ff"):
                            decoder_output_s = residual(decoder_output,
                                                      conv_hidden_relu(
                                                          decoder_output,
                                                          hidden_size=4 * self.config.hidden_units,
                                                          output_size=self.config.hidden_units,
                                                          summaries=self.summary),
                                                      dropout_rate=self.config.residual_dropout_rate,
                                                      is_training=self.is_training)
                        '''
                        decoder_output_s = decoder_output
                    with tf.variable_scope("context_level"):
                        # Multihead Attention context
                        document_attention = tf.matmul(tf.expand_dims(table, -1), tf.expand_dims(document_attention, axis=2))
                        document_attention = tf.reshape(document_attention, [batch_size, -1, self.config.hidden_units])

                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=document_attention,
                                                        bias=None,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)

                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=context_output,
                                                        bias=context_attention_bias,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_vanilla_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)  # (BxN)x?xD
                        '''
                        with tf.variable_scope("context_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)
                        '''
                    with tf.variable_scope("gate_level"):
                        decoder_output_c = tf.reshape(decoder_output_c,
                                                      [-1, self.config.train.max_length, self.config.hidden_units])
                        c = tf.constant(1.0, dtype=tf.float32)
                        decoder_output_c = (c - self.p) * decoder_output_c + self.p * decoder_output_s
                        with tf.variable_scope("gate_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)


        return decoder_output_c, decoder_output_s, self.p

    def decoder_context_gate_h_pool_new(self, decoder_input, encoder_output, context_output, document_attention, sentence_output, table, scope, reuse):
        input_shape = tf.shape(encoder_output)  # (BxN)xLxD
        batch_size = tf.shape(context_output)[0]  # Bx(Nx?)xD


        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        # context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)  # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            # context adjustment
            # decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # decoder_output_c += add_timing_signal_1d(decoder_output_c)
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            #decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            # self_attention_bias2 = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length)
            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    with tf.variable_scope("sentence_level"):
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=None,
                                                      bias=self_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      output_depth=self.config.hidden_units,
                                                      name="decoder_self_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD

                        decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units]) # Bx(Nx?)xD
                        #decoder_output_c = decoder_output
                        # Multihead Attention (vanilla attention)
                        decoder_output = residual(decoder_output,
                                                  multihead_attention(
                                                      query=decoder_output,
                                                      memory=encoder_output,
                                                      bias=encoder_attention_bias,
                                                      total_key_depth=self.config.hidden_units,
                                                      total_value_depth=self.config.hidden_units,
                                                      output_depth=self.config.hidden_units,
                                                      num_heads=self.config.num_heads,
                                                      dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                      name="decoder_vanilla_attention",
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)  # (BxN)x?xD
                        '''
                        with tf.variable_scope("sentence_ff"):
                            decoder_output_s = residual(decoder_output,
                                                      conv_hidden_relu(
                                                          decoder_output,
                                                          hidden_size=4 * self.config.hidden_units,
                                                          output_size=self.config.hidden_units,
                                                          summaries=self.summary),
                                                      dropout_rate=self.config.residual_dropout_rate,
                                                      is_training=self.is_training)
                        '''
                        decoder_output_s = decoder_output
                    with tf.variable_scope("context_level"):
                        # Multihead Attention context
                        document_attention = tf.matmul(tf.expand_dims(table, -1), tf.expand_dims(document_attention, axis=2)) # BxNxLxD

                        document_attention = tf.reshape(document_attention, [batch_size, -1, self.config.hidden_units])
                        #document_attention = tf.reshape(document_attention, [-1, self.config.train.max_length, self.config.hidden_units])
                        sentence_output = tf.reshape(sentence_output, [batch_size, -1, self.config.hidden_units])
                        document_attention = residual(document_attention, sentence_output,
                                                      dropout_rate=self.config.residual_dropout_rate,
                                                      is_training=self.is_training)
                        decoder_output_c = residual(decoder_output_c,
                                                    multihead_attention(
                                                        query=decoder_output_c,
                                                        memory=document_attention,
                                                        bias=None,
                                                        total_key_depth=self.config.hidden_units,
                                                        total_value_depth=self.config.hidden_units,
                                                        output_depth=self.config.hidden_units,
                                                        num_heads=self.config.num_heads,
                                                        dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                        name="decoder_context_attention",
                                                        summaries=self.summary),
                                                    dropout_rate=self.config.residual_dropout_rate,
                                                    is_training=self.is_training)
                        '''
                        with tf.variable_scope("context_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)
                        '''

                    with tf.variable_scope("gate_level"):
                        decoder_output_c = tf.reshape(decoder_output_c,[-1, self.config.train.max_length, self.config.hidden_units])
                        c = tf.constant(1.0, dtype=tf.float32)
                        decoder_output_c = (c - self.p) * decoder_output_c + self.p * decoder_output
                        with tf.variable_scope("gate_ff"):
                            decoder_output_c = residual(decoder_output_c,
                                                        conv_hidden_relu(
                                                            decoder_output_c,
                                                            hidden_size=4 * self.config.hidden_units,
                                                            output_size=self.config.hidden_units,
                                                            summaries=self.summary),
                                                        dropout_rate=self.config.residual_dropout_rate,
                                                        is_training=self.is_training)


        return decoder_output_c, decoder_output_s, self.p


    def decoder_context_gate_new(self, decoder_input, encoder_output, context_output, scope, cnn, reuse):
        input_shape = tf.shape(encoder_output) #(BxN)xLxD
        batch_size = tf.shape(context_output)[0] #Bx(Nx?)xD
        if not cnn:
            context = tf.reshape(context_output, [input_shape[0], input_shape[1], self.config.hidden_units]) # （BxN)xLxD
        else:
            context = tf.reshape(context_output, [input_shape[0], 1, self.config.hidden_units])
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        context_padding = tf.equal(tf.reduce_sum(tf.abs(context_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        #context_attention_bias = attention_bias_lower_traingle_l(tf.shape(context_output)[1], self.config.train.max_length) + \
        #                         attention_bias_ignore_padding(context_padding)
        context_attention_bias = attention_bias_ignore_padding(context_padding)
        decoder_output = target(decoder_input,
                                vocab_size=self.config.dst_vocab_size,
                                dense_size=self.config.hidden_units,
                                shared_embedding=self.config.train.shared_embedding,
                                reuse=reuse,
                                multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0) # (BxN)xLxD

        decoder_output_c = target(tf.reshape(decoder_input, [batch_size, -1]),
                                  vocab_size=self.config.dst_vocab_size,
                                  dense_size=self.config.hidden_units,
                                  shared_embedding=self.config.train.shared_embedding,
                                  reuse=reuse,
                                  multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0) # (BxN)xLxD

        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            # Positional Encoding
            # context adjustment
            #decoder_output_c += add_timing_signal_1d(decoder_output_c)
            decoder_output += add_timing_signal_1d(decoder_output)
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])
            if cnn:
                self_attention_bias2 = attention_bias_lower_triangle(tf.shape(context_output)[1])
            else:
                self_attention_bias2 = attention_bias_lower_triangle(tf.shape(context_output)[1])
                self_attention_bias2 = tf.tile(self_attention_bias2, [1, 1, self.config.train.max_length, 1])

            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  output_depth=self.config.hidden_units,
                                                  name="decoder_self_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training) # (BxN)x?xD

                    #decoder_output_c = tf.reshape(decoder_output, [batch_size, -1, self.config.hidden_units])

                    # Multihead Attention (vanilla attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=encoder_output,
                                                  bias=encoder_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)  # (BxN)x?xD

                    with tf.variable_scope("sentence_ff"):
                        decoder_output = residual(decoder_output,
                                                  conv_hidden_relu(
                                                      decoder_output,
                                                      hidden_size=4 * self.config.hidden_units,
                                                      output_size=self.config.hidden_units,
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)
                    if cnn:
                        decoder_output_c = tf.reshape(
                            self.document_cnn(tf.reshape(decoder_output_c, [input_shape[0], input_shape[1], self.config.hidden_units]), "decoder_cnn",
                                              reuse=reuse),
                            [batch_size, -1, self.config.hidden_units])

                    decoder_output_c = residual(decoder_output_c,
                                              multihead_attention(
                                                  query=decoder_output_c,
                                                  memory=context_output,
                                                  bias=self_attention_bias2,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_context_atthention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    decoder_output_c = tf.reshape(decoder_output_c, [input_shape[0], -1, self.config.hidden_units])
                    decoder_output_final = residual(decoder_output,
                                              multihead_attention(
                                                  query=decoder_output,
                                                  memory=decoder_output_c,
                                                  bias=None,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)  # (BxN)x?xD

                    with tf.variable_scope("context_ff"):
                        decoder_output_final = residual(decoder_output_final,
                                                  conv_hidden_relu(
                                                      decoder_output_final,
                                                      hidden_size=4 * self.config.hidden_units,
                                                      output_size=self.config.hidden_units,
                                                      summaries=self.summary),
                                                  dropout_rate=self.config.residual_dropout_rate,
                                                  is_training=self.is_training)

        return decoder_output_final, decoder_output

    def encoder_doc(self, context, num, scope, reuse):

        def recurrency(i, context, attention):
            attention = tf.cond(i>0, lambda: self.encoder_c(context[num - 1 - i], attention, scope, reuse),
                                lambda:self.encoder_c(context[num - 1 - i], None, scope, reuse))
            return i+1, context, attention

        attention = tf.zeros([tf.shape(context)[1], tf.shape(context)[2], self.config.hidden_units,], tf.float32)
        initial_i = 0
        _, sentence, attention = tf.while_loop(
            cond=lambda a, _1, _2: a < num,
            body=recurrency,
            loop_vars=(initial_i, context, attention),
        )

        attention = self.encoder_c(context[-1], None, scope, reuse)
        return attention

    def test_output(self, decoder_output, reuse):
        last_logits = top(body_output=decoder_output[:, -1],
                          vocab_size=self.config.dst_vocab_size,
                          dense_size=self.config.hidden_units,
                          shared_embedding=self.config.train.shared_embedding,
                          reuse=reuse)
        with tf.variable_scope("output",initializer=self.initializer,  reuse=reuse):
            last_preds = tf.to_float(tf.argmax(last_logits, dimension=-1))
            z = tf.nn.softmax(last_preds)
            last_k_scores, last_k_preds = tf.nn.top_k(z, k=self.config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_loss(self, decoder_output, Y, reuse):
        logits = top(body_output=decoder_output,
                     vocab_size=self.config.dst_vocab_size,
                     dense_size=self.config.hidden_units,
                     shared_embedding=self.config.train.shared_embedding,
                     reuse=reuse)
        with tf.variable_scope("output", initializer=self.initializer, reuse=reuse):
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=self.config.dst_vocab_size)
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = smoothing_cross_entropy(logits=logits, labels=Y, vocab_size=self.config.dst_vocab_size,
                                           confidence=1-self.config.train.label_smoothing)
            loss_sum = tf.reduce_sum(loss * mask)  / tf.reduce_sum(mask)
        return loss_sum

    def gan_output(self, decoder_output, Y, reward, reuse):
        logits = top(body_output=decoder_output,
                     vocab_size=self.config.dst_vocab_size,
                     dense_size=self.config.hidden_units,
                     shared_embedding=self.config.train.shared_embedding,
                     reuse=reuse)
        with tf.variable_scope("output", initializer=self.initializer, reuse=reuse):
            l_shape = tf.shape(logits)
            probs = tf.nn.softmax(tf.reshape(logits, [-1, self.config.dst_vocab_size]))
            probs = tf.reshape(probs, [l_shape[0], l_shape[1], l_shape[2]])
            sample = tf.to_float(l_shape[0])
            '''
            g_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(
                tf.reduce_sum(tf.one_hot(tf.reshape(Y, [-1]), self.config.dst_vocab_size, 1.0, 0.0) *
                              tf.reshape(probs, [-1, self.config.dst_vocab_size]), 1) *
                tf.reshape(reward, [-1]), 1e-20, 1.0)), 0
            ) / sample
            '''
            g_loss = -tf.reduce_sum(
                tf.reduce_sum(tf.one_hot(tf.reshape(Y, [-1]), self.config.dst_vocab_size, 1.0, 0.0) *
                              tf.log(tf.clip_by_value(tf.reshape(probs, [-1, self.config.dst_vocab_size]), 1e-5, 1.0)), 1) *
                tf.reshape(reward, [-1]), 0
            ) / sample
        return g_loss

    def train_output(self, decoder_output, Y, reuse, scope="output"):
        logits = top(body_output=decoder_output,
                     vocab_size=self.config.dst_vocab_size,
                     dense_size=self.config.hidden_units,
                     shared_embedding=self.config.train.shared_embedding,
                     reuse=reuse)
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            preds = tf.to_int32(tf.argmax(logits, dimension=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)
            loss = smoothing_cross_entropy(logits=logits, labels=Y, vocab_size=self.config.dst_vocab_size,
                                           confidence=1-self.config.train.label_smoothing)
            mean_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

        return acc, mean_loss, preds

    def document_cnn(self, encoder_input, scope, reuse):
        if len(encoder_input.get_shape().as_list()) == 2:
            context = bottom(encoder_input,
                                    vocab_size=self.config.src_vocab_size,
                                    dense_size=self.config.hidden_units,
                                    shared_embedding=self.config.train.shared_embedding,
                                    reuse=reuse,
                                    multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0)
        elif len(encoder_input.get_shape().as_list()) == 3:
            context = encoder_input
        else:
            raise AssertionError
        context = tf.expand_dims(context, -1)
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable('W', initializer=tf.truncated_normal(
                [self.num_filters_total, self.config.hidden_units], stddev=0.1))
            b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.config.hidden_units]))

            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                # print('the filter size is ', filter_size)
                scope = "conv_maxpool-%s" % filter_size
                filter_shape = [filter_size, self.config.hidden_units, 1, num_filter]
                strides = [1, 1, 1, 1]
                conv = cnn_layer(filter_size, self.config.hidden_units, num_filter, scope=scope, reuse_var=reuse)
                is_train = True
                conv_out = conv.conv_op(context, strides, is_train=is_train)
                pooled = tf.nn.max_pool(conv_out, ksize=[1, (self.config.train.max_length - filter_size + 1), 1, 1],
                                        strides=strides, padding='VALID', name='pool')
                # print('the shape of the pooled is ', pooled.get_shape())
                pooled_outputs.append(pooled)

            h_pool = tf.concat(axis=3, values=pooled_outputs)
            # print('the shape of h_pool is ', h_pool.get_shape())
            # print('the num_filters_total is ', self.num_filters_total)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

            # print('the shape of h_pool_flat is ', h_pool_flat.get_shape())

            h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0, reuse_var=reuse)
            h_drop = tf.nn.dropout(h_highway, 1.0)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
            return scores # (BxN)xD

    def encoder_cnn(self, input, scope, reuse):
        with tf.variable_scope(scope, initializer=self.initializer, reuse=reuse):
            encoder_output = input + add_timing_signal_1d(input)
            encoder_output = tf.layers.dropout(encoder_output, rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            self_attention_bias = attention_bias_lower_triangle(tf.shape(input)[1])
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query=encoder_output,
                                                  memory=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self.summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
            return encoder_output

    def encoder_context(self, encoder_input, X_shape, scope, reuse=None):
        with tf.variable_scope(scope):
            context = self.encoder_d(encoder_input, "context_s", reuse=reuse)  # B x (N x L) x D
            context_s = tf.reshape(context, [-1, self.config.train.max_length, self.config.hidden_units])
            context_s_t = tf.transpose(context_s, [0, 2, 1])  # (BxN)xDxL
            context_t = tf.layers.dense(context_s_t, 1, name="context_dense", reuse=reuse)  # (BxN)xDx1
            #context_t = linear_c(context_s_t, 1, True, scope="context_dense")
            context_t = tf.reshape(tf.squeeze(context_t, -1), [X_shape[0], X_shape[1], self.config.hidden_units])  # BxNxD
            #context_encoder_output = self.encoder_s_c(context_t, "context_c", reuse)  # BxNxD
            context_encoder_output = context_t
        return context, context_encoder_output

    def encoder_context_pool(self, encoder_input, X_shape, scope, reuse=None):
        with tf.variable_scope(scope):
            context = self.encoder_d(encoder_input, "context_s", reuse=reuse)  # B x (N x L) x D
            context_s = tf.reshape(context, [-1, self.config.train.max_length, self.config.hidden_units]) # (BxN)xLxD
            context_s_t = tf.transpose(context_s, [1, 0, 2])  # Lx(BxN)xD
            context_t, table = self.get_pooled_out(context_s_t, "attn")  # (BxN)xD, (BxN)xL
            table = tf.reshape(table, [X_shape[0], X_shape[1], self.config.train.max_length])
            table = self.make_hash_table(table, 3)
            #context_t = linear_c(context_s_t, 1, True, scope="context_dense")
            context_t = tf.reshape(context_t, [X_shape[0], X_shape[1], self.config.hidden_units])  # BxNxD
            context_encoder_output = self.encoder_s_c(context_t, "context_c", reuse)  # BxNxD
            context = tf.multiply(context, tf.reshape(table, [X_shape[0], -1, 1]))
        return context, context_encoder_output, table # Bx(N x L)xD, BxNxD, BxNxL

    def get_pooled_out(self, output, summary_type, use_summ_proj=True):
        """
        Args:
          summary_type: str, "last", "first", "mean", or "attn". The method
            to pool the input to get a vector representation.
          use_summ_proj: bool, whether to use a linear projection during pooling.

        Returns:
          float32 Tensor in shape [bsz, d_model], the pooled representation.
        """
        input_mask = None

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            summary, table = summarize_sequence(
                summary_type=summary_type,
                hidden=output,
                d_model=self.config.hidden_units,
                n_head=self.config.num_heads,
                d_head=self.config.hidden_units/self.config.num_heads,
                dropout=self.config.residual_dropout_rate,
                dropatt=self.config.attention_dropout_rate,
                is_training=self.is_training,
                input_mask=input_mask,
                initializer=tf.initializers.random_normal(stddev=0.1, seed=None),
                summaries=self.summary,
                use_proj=use_summ_proj)

            table = tf.transpose(table)

        return summary, table

    def make_hash_table(self, table, n):
        """
        :param table: [BxNxL]
        :return:
        """
        '''
        W = tf.get_variable("threshold_kernel", shape=[self.config.train.max_length, 1], trainable=True)
        threshold = tf.tile(tf.einsum("bnl,li->bni", table, W), [1, 1, self.config.train.max_length])
        M = tf.where(tf.nn.relu(table-threshold))
        return M
        '''
        a_top, a_top_idx = tf.nn.top_k(table, n, sorted=False)
        kth = tf.reduce_min(a_top, axis=2, keepdims=True)
        top2 = tf.greater_equal(table, kth)
        #mask = tf.cast(top2, dtype=tf.float32)
        mask = tf.where(top2, tf.ones_like(top2, dtype=tf.float32), 1e-5*tf.ones_like(top2, dtype=tf.float32))
        mask = tf.nn.softmax(mask, axis=-1)
        '''
        v = tf.multiply(table, mask)
        sum = tf.reciprocal(tf.reduce_sum(v, axis=2))  # 取倒数
        v = tf.transpose(v, [0, 2, 1])
        norms = tf.transpose(tf.multiply(v, sum), [0, 2, 1])
        '''
        return mask

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads

def residual(inputs, outputs, dropout_rate, is_training):
    output = inputs + tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
    output = layer_norm(output)
    return output

def split_tensor(input, n):
    batch_size = tf.shape(input)[0]
    ls = tf.cast(tf.lin_space(0.0, tf.cast(batch_size, tf.float32), n + 1), tf.int32)  ##type change
    return [input[ls[i]:ls[i + 1]] for i in range(n)]

def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.learning_rate_warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5
    )

def shift_right(input, pad=2):
    return tf.concat((tf.ones_like(input[:, :1])*pad, input[:,:-1]), 1)

def get_weight(vocab_size, dense_size, name=None):
    weights = tf.get_variable("kernel", [vocab_size, dense_size],
                              initializer=tf.random_normal_initializer(0.0, 512 ** -0.5))
    return weights

def bottom(x, vocab_size, dense_size, shared_embedding=True, reuse=None, multiplier=1.0):
    with tf.variable_scope("embedding", reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=reuse):
               embedding_var = get_weight(vocab_size, dense_size)
               emb_x = tf.gather(embedding_var, x)
               if multiplier != 1.0:
                   emb_x *= multiplier
        else:
            with tf.variable_scope("src_embedding", reuse=reuse):
                embedding_var = get_weight(vocab_size, dense_size)
                emb_x = tf.gather(embedding_var, x)
                if multiplier !=1.0:
                    emb_x *= multiplier
    return emb_x

def target(x, vocab_size, dense_size, shared_embedding=True, reuse=None, multiplier=1.0):
    with tf.variable_scope("embedding", reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
               embedding_var = get_weight(vocab_size, dense_size)
               emb_x = tf.gather(embedding_var, x)
               if multiplier != 1.0:
                   emb_x *= multiplier
        else:
            with tf.variable_scope("dst_embedding", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                emb_x = tf.gather(embedding_var, x)
                if multiplier !=1.0:
                    emb_x *= multiplier
    return emb_x

def top(body_output, vocab_size, dense_size, shared_embedding=True, reuse=None):
    with tf.variable_scope('embedding', reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
                shape=tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size])
                embedding_var = get_weight(vocab_size, dense_size)
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
        else:
            with tf.variable_scope("softmax", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                shape=tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size])
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
    return logits

def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
        name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        emb_x = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            emb_x *= multiplier
        return emb_x


class cnn_layer(object):
    def __init__(self, filter_size, dim_word, num_filter, scope='cnn_layer', init_device='/cpu:0', reuse_var=False):
        self.filter_size = filter_size
        self.dim_word = dim_word
        self.num_filter = num_filter
        self.scope = scope
        self.reuse_var = reuse_var
        with tf.variable_scope(self.scope or 'cnn_layer', reuse=reuse_var):
            with tf.variable_scope('self_model', reuse=reuse_var):
                with tf.device(init_device):
                    filter_shape = [filter_size, dim_word, 1, num_filter]
                    b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[num_filter]))
                    W = tf.get_variable('W', initializer=tf.truncated_normal(filter_shape, stddev=0.1))

    ## convolutuon with batch normalization
    def conv_op(self, input_sen, stride, is_train, padding='VALID', is_batch_norm=True, f_activation=tf.nn.relu):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('self_model'):
                tf.get_variable_scope().reuse_variables()
                b = tf.get_variable('b')
                W = tf.get_variable('W')
                conv = tf.nn.conv2d(
                    input_sen,
                    W,
                    stride,
                    padding,
                    name='conv')
                bias_add = tf.nn.bias_add(conv, b)

            if is_batch_norm:
                with tf.variable_scope('conv_batch_norm'):
                    conv_bn = conv_batch_norm(bias_add, is_train=is_train, scope='bn', reuse_var=self.reuse_var)
                h = f_activation(conv_bn, name='relu')
            else:
                h = f_activation(bias_add, name='relu')

        return h

def conv_batch_norm(x, is_train, scope='bn', decay=0.9, reuse_var=False):
    out = batch_norm(x,
                     decay=decay,
                     center=True,
                     scale=True,
                     updates_collections=None,
                     is_training=is_train,
                     reuse=reuse_var,
                     trainable=True,
                     scope=scope)
    return out


def linear(inputs, output_size, use_bias, scope='linear'):
    if not scope:
        scope = tf.get_variable_scope()

    input_size = inputs.get_shape()[1].value
    dtype = inputs.dtype

    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', [input_size, output_size], dtype=dtype)
        res = tf.matmul(inputs, weights)
        if not use_bias:
            return res
        biases = tf.get_variable('biases', [output_size], dtype=dtype)
    return tf.add(res, biases)

def linear_c(inputs, output_size, use_bias, scope='linear'):
    if not scope:
        scope = tf.get_variable_scope()

    input_size = inputs.get_shape()[-1].value
    dtype = inputs.dtype

    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', [input_size, output_size], dtype=dtype)
        res = tf.matmul(inputs, weights)
        if not use_bias:
            return res
        biases = tf.get_variable('biases', [output_size], dtype=dtype)
    return tf.add(res, biases)

def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu, reuse_var=False):
    output = input_
    if reuse_var == True:
        tf.get_variable_scope().reuse_variables()
    for idx in range(layer_size):
        output = f(linear(output, size, 0, scope='output_lin_%d' % idx))
        transform_gate = tf.sigmoid(linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_
    return output