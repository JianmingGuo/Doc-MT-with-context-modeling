import numpy as np
import six
from six.moves import zip
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoints", "", "Comma-separated list of checkpoints to average.")
flags.DEFINE_string("prefix", "", "Prefix (e.g., directory) to append to each checkpoint.")
flags.DEFINE_string("output_path", "/tmp/averaged.ckpt", "Path to output the averaged checkpoint to.")

def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path+".meta") or tf.gfile.Exists(path+".index"))

def main():
    checkpoints = [c.strip() for c in FLAGS.checkpoints.split(",")]
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")
    if FLAGS.prefix:
        checkpoints = [FLAGS.prefix + c for c in checkpoints]
    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
    if not checkpoints:
        raise ValueError("None of the provided checkpoints exists. %s" % FLAGS.checkpoints)
    tf.logging.info("Reading variables and averagin checkpoints:")
    for c in checkpoints:
        tf.logging.info("%s"%c)
    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in  var_list:
        if not name.startswith("global_step"):
            var_values[name] = np.zeros(shape)
    for checkpoint in checkpoints:
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor
        tf.logging.info("Read from checkpoint %s"%checkpoint)
    for name in var_values:
        var_values[name] /= len(checkpoints)

    tf_vars = [tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v]) for v in var_values]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(
        0, name="global_step", trainable=False, dtype=tf.int64)
    saver = tf.train.Saver(tf.all_variables())

    # Build a model consisting only of variables, set them to the average values.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               six.iteritems(var_values)):
            sess.run(assign_op, {p: value})
        # Use the built saver to save the averaged checkpoint.
        saver.save(sess, FLAGS.output_path, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", FLAGS.output_path)


if __name__ == "__main__":
    tf.app.run()