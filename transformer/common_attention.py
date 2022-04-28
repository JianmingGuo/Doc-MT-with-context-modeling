import math
import tensorflow as tf
from transformer import common_layers
from .DiSAN import disan

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1e4):
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1)
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return x + signal

def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1e4):
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in range(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x

def add_positional_embedding_nd(x, max_length, name):
    static_shape = x.get_shape().as_list()
    dynamic_shape = tf.shape(x)
    num_dims = len(static_shape) -2
    depth = static_shape[-1]
    base_shape = [1] * (num_dims + 1) + [depth]
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [1] * num_dims + [depth]
    for i in range(num_dims):
        shape = base_shape[:]
        start = base_start[:]
        size = base_size[:]
        shape[i+1] = max_length
        size[i+1] = dynamic_shape[i+1]
        var = (tf.get_variable(name+"_%d"%i, shape, initializer=tf.random_normal_initializer(0, depth**-0.5))*(depth**0.5))
        x += tf.slice(var, start, size)
    return x

def embedding_to_padding(emb):
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.equal(emb_sum, 0.0)

def attention_bias_lower_triangle(length):
    lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    ret = -1e9 * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])

def attention_bias_lower_traingle_l(length, l):
    lower_triangle = tf.zeros([length, length])
    def recurrency(i, lower_triangle, l):
        new_block = tf.concat([tf.ones([l, i+l]), tf.zeros([l, length-(i+l)])], 1)
        lower_triangle = tf.concat([lower_triangle[:i,:], new_block, lower_triangle[i+l:,:]], 0)
        return i+l, lower_triangle, l

    initial_i = 0
    _, lower_triangle, _ = tf.while_loop(
        cond=lambda a, _1, _2: a < length,
        body=recurrency,
        loop_vars=(initial_i, lower_triangle, l),
    )

    ret = -1e9 * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])

def attention_bias_lower_traingle_cnn(length, l): #k n
    lower_triangle = tf.zeros([length*l, length])
    def recurrency(i, lower_triangle, l):
        new_block = tf.concat([tf.ones([length, i+1]), tf.zeros([length, l-(i+1)])], 1)
        lower_triangle = tf.concat([lower_triangle[:i*length,:], new_block, lower_triangle[(i+1)*length:,:]], 0)
        return i+1, lower_triangle, l

    initial_i = 0
    _, lower_triangle, _ = tf.while_loop(
        cond=lambda a, _1, _2: a < l,
        body=recurrency,
        loop_vars=(initial_i, lower_triangle, l),
    )

    ret = -1e9 * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length*l, length])

def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
    Args:
        memory_padding: a boolean `Tensor` with shape [batch, memory_length].
    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
    """
    ret = tf.to_float(memory_padding) * -1e9
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)

def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret

def combine_last_two_dimensions(x):
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a*b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret

def split_heads(x, num_heads):
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

def combine_heads(x):
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def attention_image_summary(attn, image_shapes=None):
    num_heads = attn.get_shape().as_list()[1]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])
    image = split_last_dimension(image, 3)
    image = tf.reduce_max(image, 4)
    if image_shapes is not None:
        q_rows, q_cols, m_rows, m_cols = list(image_shapes)
        image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
        image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
        image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
    tf.summary.image("attention", image, max_outputs=1)

def dot_product_attention(q, k, v, bias, dropout_rate=0.0, summaries=False, image_shapes=None, name=None):
    """dot-product attention.
      Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        dropout_rate: a floating point number
        summaries: a boolean
      Returns:
        A Tensor
    """
    with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.nn.dropout(weights, 1.0-dropout_rate)
        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)
        return tf.matmul(weights, v)

def multihead_attention(query, memory, bias, total_key_depth, total_value_depth, output_depth, num_heads, dropout_rate,
                        summaries=False, image_shapes=None, name=None):
    """
    Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
    Returns:
        A Tensor
    """
    with tf.variable_scope(name, default_name="multihead_attention", values=[query, memory]):
        if memory is None:
            combined = common_layers.conv1d(query, total_key_depth*2+total_value_depth, 1, name="qkv_transform")
            q, k, v = tf.split(combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
        else:
            q = common_layers.conv1d(query, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(memory, total_key_depth+total_value_depth, 1, name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x = dot_product_attention(q, k, v, bias, dropout_rate, summaries, image_shapes, name)
        x = combine_heads(x)
        x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
        print("attention:",x.shape,'\n')
        return x

def threshold_m(m, dim1, dim2, filter_size, num_filter):
    """
    :param m:  a Tensor with shape [batch, heads, length_q, length_kv]
    :return: a Tensor with shape [batch, heads, length_q, length_kv]
    """
    shape = tf.shape(m)
    m = tf.reshape(m, [-1, dim1, dim2])
    value = tf.layers.conv1d(m, num_filter, filter_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    value = tf.layers.max_pooling1d(value, dim1-filter_size+1, 1)
    value = tf.layers.dense(value, 1, "sigmoid") #(bxh)
    value = tf.reshape(value, [shape[0], shape[1], 1, 1])
    value = tf.tile(value, [1, 1, shape[2], shape[3]])

    return value

def threshold_t(m, bias, bias_start=0):
    """
    :param m:  a Tensor with shape [batch, heads, length_q, length_kv]
    :return: a Tensor with shape [batch, heads, length_q, length_kv]
    """
    shape = tf.shape(m)
    bsz = shape[0]
    value = disan(m, None, "disan", is_train=True, activation="sigmoid")
    input_size = value.get_shape()[-1]
    output_size = 1
    W = tf.get_variable('W', shape=[input_size, output_size], dtype=tf.float32,)
    if bias:
        bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float32,
                               initializer=tf.constant_initializer(bias_start))
        out = tf.matmul(value, W) + bias
    else:
        out = tf.matmul(value, W)
    return out


def dot_product_attention_combiner(q, k, v, m, dim1, dim2, filter_size, num_filter, bias, dropout_rate=0.0, summaries=False, image_shapes=None, name=None):
    """dot-product attention.
      Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        m: a Tensor with shape [batch, heads, length_q, length_kv]
        bias: bias Tensor (see attention_bias())
        dropout_rate: a floating point number
        summaries: a boolean
      Returns:
        A Tensor
    """
    with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        logits = tf.matmul(q, k, transpose_b=True)
        tm = threshold_m(logits, dim1, dim2, filter_size, num_filter)
        #tm = threshold_t(logits, True)
        logits = tf.nn.relu(logits - tm, "M")
        if bias is not None:
            logits += bias
        if m is None:
            m = tf.zeros_like(logits)
        weights = tf.nn.softmax(logits+m, name="attention_weights")
        weights = tf.nn.dropout(weights, 1.0-dropout_rate)
        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)
        return tf.matmul(weights, v), weights

def multihead_attention_combiner(query, memory, m, bias, dim1, dim2, filter_size, num_filter, total_key_depth, total_value_depth, output_depth, num_heads, dropout_rate,
                        summaries=False, image_shapes=None, name=None):
    """
    Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
    Returns:
        A Tensor
    """
    with tf.variable_scope(name, default_name="multihead_attention", values=[query, memory]):
        if memory is None:
            combined = common_layers.conv1d(query, total_key_depth*2+total_value_depth, 1, name="qkv_transform")
            q, k, v = tf.split(combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
        else:
            q = common_layers.conv1d(query, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(memory, total_key_depth+total_value_depth, 1, name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        if m is not None:
            m = split_heads(m, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x, c = dot_product_attention_combiner(q, k, v, m, dim1, dim2, filter_size, num_filter,bias, dropout_rate, summaries, image_shapes, name)
        x = combine_heads(x)
        c = combine_heads(c)
        x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
        return x, c

