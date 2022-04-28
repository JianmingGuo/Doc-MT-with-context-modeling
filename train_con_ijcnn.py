import yaml
import time
import os
import sys
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model_con_ijcnn import Model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train(config):
    logger = logging.getLogger('')
    du = DataUtil(config=config)
    du.load_vocab()  ## build dict

    model = Model(config=config)


    model.build_train_model(True)



    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with model.graph.as_default():
        saver = tf.train.Saver(var_list=tf.global_variables())
        summary_writer = tf.summary.FileWriter(config.train.logdir, graph=model.graph)
        with tf.Session(config=sess_config) as sess:

            sess.run(tf.global_variables_initializer())
            try:
                saver.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
            except:
                logger.info('Failed to reload model.')

            # model.init_and_restore()
            # sys.exit()

            for epoch in range(1, config.train.num_epochs+1):
                for batch in du.get_training_batches_doc():  ##get batches
                    start_time = time.time()
                    step = sess.run(model.global_step)
                    # Summary
                    if step % config.train.summary_freq == 0:
                        step, lr, gnorm, loss, acc, summary, _ = sess.run([model.global_step, model.learning_rate,
                                                                          model.grads_norm, model.loss, model.acc,
                                                                        #   model.summary_op],
                                                                           model.summary_op, model.train_op],
                                                                          feed_dict={model.src_pl:batch[0],
                                                                                     model.dst_pl: batch[1]})
                        summary_writer.add_summary(summary, global_step=step)
                    else:

                        step, lr, gnorm, loss, acc, preds, _ = sess.run([model.global_step, model.learning_rate,
                                                                  model.grads_norm, model.loss, model.acc, model.preds_list, model.train_op],
                                                                feed_dict={model.src_pl: batch[0], model.dst_pl: batch[1]})
                        
                        # var = sess.graph.get_tensor_by_name('sentence_encoder/block_0/encoder_self_attention/output_transform_single/kernel:0')
                        # print(sess.run(var).shape)
            

                    logger.info('epoch: {0}\tstep: {1}\tlr: {2:.6f}\tgnorm: {3:.4f}\tloss: {4:.4f}\tacc: {5:.4f}\ttime: {6:.4f}'.
                        format(epoch, step, lr, gnorm, loss, acc, time.time() - start_time))

                    # Save model
                    if step % config.train.save_freq == 0:
                        mp = config.train.logdir + '/model_step_%d'%(step)
                        saver.save(sess, mp)
                        logger.info('Save model in %s.'%mp)
            logger.info("Finish training.")

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.safe_load(open(args.config)))
    # Logger
    if not os.path.exists(config.train.logdir):
        os.makedirs(config.train.logdir)
    logging.basicConfig(filename=config.train.logdir+'/train.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # Train
    train(config)