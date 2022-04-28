import math
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import function

def NoistyTopKGatingParams():
    return tf.contrib.training.HParmas(
        gating_class=NoisyTopKGating,
        num_experts=16,  # The number of experts
        k=2,  # 'The number of experts to use per example
        input_size=None,  # size of input to MoE.  Set by MoE class
        dtype=tf.float32,  # floating point data type
        initializer=tf.zeros_initializer(),  # initializer for weight matrices
        noisy_gating=True,  # Add tunable noise (necessary for load-balancing)
        noise_epsilon=1e-2,  # Added to noise stddev for numerical stability
    )

def FeedForwardExpertParams():
    return tf.contrib.training.HParams(
        # The class that implements the expert network
        expert_class=FeedForwardExpert,
        input_size=None,  # Size of input to MoE.  Set by MoE class.
        # List of hidden layer sizes, or None for no hidden layers.
        # The length of this list determines the number of hidden layers
        hidden_layer_sizes=None,
        output_size=None,  # Size of output from MoE.  Set by MoE class.
        dtype=tf.float32,  # Floating point data type)
        # Activation function applied at each hidden layer)
        hidden_activation=tf.nn.relu,
        initializer=None,  # Optional initializer for weight matrices.)
        # If autoscale=True, At each hidden/output layer, multiply by
        # rsqrt(prev_layer_size / input_size).  This scaling happens
        # before application of hidden_activation)
        autoscale=True, )

def SetInputOutputSizes(hp, input_size, output_size):
    if hp.input_size is None:
        hp.input_size = input_size
    else:
        assert hp.input_size == input_size
    if output_size is not None:
        if hp.output_size is None:
            hp.output_size = output_size
        else:
            assert hp.output_size == output_size

class FeedForwardExpert():
    def __init__(self, hp, name):
        self.hp = hp
        hidden_layer_sizes = hp.hidden_layer_sizes or []
        num_layers = 1 + len(hidden_layer_sizes)
        layer_sizes = [hp.input_size] + hidden_layer_sizes + [hp.output_size]
        self.layer_sizes = layer_sizes
        self.w = []
        for layer in range(num_layers):
            shape = layer_sizes[layer:layer+2]
            self.w.append(tf.get_variable('%s_layer_%d'%(name, layer), shape, hp.dtype, hp.initializer))

    def eval(self, x):
        hp = self.hp
        num_layers = len(self.w)
        for i in range(num_layers):
            x = tf.matmul(x, self.w[i])
            if hp.autoscale and self.layer_sizes[i] != hp.input_size:
                x *= (self.layer_sizes[i] / hp.input_size)**-0.5
            if i+1 < num_layers and hp.hidden_activation:
                x = hp.hidden_activation(x)
        return x

    @property
    def vars(self):
        return self.w

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def ConvertGradientToTensor(x):
    return x

class Parallelism():
    def __init__(self, device_names_or_functions, reuse=None, caching_devices=None, daisy_chain_variables=False):
        assert device_names_or_functions
        self.devices = device_names_or_functions
        self.n = len(device_names_or_functions)
        self.reuse = reuse
        self.caching_devices = self.MaybeRepeat(caching_devices)
        self.daisy_chain_variables = daisy_chain_variables

    def __call__(self, fn, *args, **kwargs):
        if args:
            my_args = TransposeListOfLists([self.MaybeRepeat(arg) for arg in args])
        else:
            my_args = [[] for _ in xrange(self.n)]
        my_kwargs = [{} for _ in xrange(self.n)]
        for k, v in six.iteritems(kwargs):
            vals = self.MaybeRepeat(v)
            for i in xrange(self.n):
                my_kwargs[i][k] = vals[i]

        # Construct lists of functions.
        fns = self.MaybeRepeat(fn)

        # Now make the parallel call.
        outputs = []
        cache = {}
        for i in xrange(self.n):

            def DaisyChainGetter(getter, name, *args, **kwargs):
                """Get a variable and cache in a daisy chain."""
                device_var_key = (self.devices[i], name)
                if device_var_key in cache:
                    # if we have the variable on the correct device, return it.
                    return cache[device_var_key]
                if name in cache:
                    # if we have it on a different device, copy it from the last device
                    v = tf.identity(cache[name])
                else:
                    var = getter(name, *args, **kwargs)
                    v = tf.identity(var._ref())  # pylint: disable=protected-access
                # update the cache
                cache[name] = v
                cache[device_var_key] = v
                return v

            # Variable scope will not reset caching_device on reused variables,
            # so we make a custom getter that uses identity to cache the variable.
            # pylint: disable=cell-var-from-loop
            def CachingGetter(getter, name, *args, **kwargs):
                v = getter(name, *args, **kwargs)
                key = (self.caching_devices[i], name)
                if key in cache:
                    return cache[key]
                with tf.device(self.caching_devices[i]):
                    ret = tf.identity(v._ref())  # pylint: disable=protected-access
                cache[key] = ret
                return ret

            if self.daisy_chain_variables:
                custom_getter = DaisyChainGetter
            elif self.caching_devices:
                custom_getter = CachingGetter
            else:
                custom_getter = None
            # pylint: enable=cell-var-from-loop
            with tf.name_scope('parallel_%d' % i):
                with tf.variable_scope(
                        tf.get_variable_scope(),
                        reuse=True if i > 0 and self.reuse else None,
                        caching_device=self.caching_devices[i],
                        custom_getter=custom_getter):
                    with tf.device(self.devices[i]):
                        outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
        if isinstance(outputs[0], tuple):
            outputs = list(zip(*outputs))
            outputs = tuple([list(o) for o in outputs])
        return outputs

    @property
    def n(self):
        return self.n

    @property
    def devices(self):
        return self.devices

    def MaybeRepeat(self, x):
        if isinstance(x, list):
            assert len(x) == self.n
            return x
        else:
            return [x] * self.n

def Parallel(device_names_or_functions, fn, *args):
    return Parallelism(device_names_or_functions)(fn, *args)

def RowiseUnsortedSegmentSum(values, indices, n):
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])

def NormalDistributionCDF(x, stddev):
    return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))

def _ProbInTopK(clean_values, noisy_values, noise_stddev, noisy_top_values, k):
    batch = tf.shape(clean_values)[0]
    m = tf.shape(noisy_top_values)[1]
    top_values_flat = tf.reshape(noisy_top_values, [-1])
    # we want to compute the threshold that a particular value would have to
    # exceed in order to make the top k.  This computation differs depending
    # on whether the value is already in the top k.
    threshold_positions_if_in = tf.range(batch) * m + k
    threshold_if_in = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_in), 1)
    is_in = tf.greater(noisy_values, threshold_if_in)
    if noise_stddev is None:
        return tf.to_float(is_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = NormalDistributionCDF(clean_values - threshold_if_in,
                                        noise_stddev)
    prob_if_out = NormalDistributionCDF(clean_values - threshold_if_out,
                                         noise_stddev)
    prob = tf.where(is_in, prob_if_in, prob_if_out)
    return prob

def CVSquared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
        x: a `Tensor`.
    Returns:
        a `Scalar`.
    """
    epsilon = 1e-10
    float_size = tf.to_float(tf.size(x)) + epsilon
    mean = tf.reduce_sum(x) / float_size
    variance = tf.reduce_sum(tf.square(x - mean)) / float_size
    return variance / (tf.square(mean) + epsilon)

def MaxOverload(load):
    per_device_load = tf.reduce_sum(tf.reshape(load, [tf.shape(load)[0], -1]), 1)
    return (tf.reduce_max(per_device_load) /
            (tf.reduce_mean(per_device_load) + 1e-10))

def GatesToLoad(gates):
    return tf.reduce_sum(tf.to_float(gates > 0), 0)

def MyTopK(x, k):
    if k > 10:
        return tf.nn.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[1]
    for i in range(k):
        values.append(tf.reduce_max(x, 1))
        argmax = tf.argmax(x, 1)
        indices.append(argmax)
        if i+1 < k:
            x += tf.one_hot(argmax, depth, -1e9)
    return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))

class NoisyTopKGating():
    def __init__(self, hp, name):
        self.vars = []
        self.hp = hp
        self.w_gate = tf.get_variable('%s_gate' % name,
                                       [hp.input_size,
                                        hp.num_experts], hp.dtype, hp.initializer)
        self.vars.append(self.w_gate)
        if hp.noisy_gating:
            self.w_noise = tf.get_variable('%s_noise' % name,
                                            [hp.input_size, hp.num_experts], hp.dtype,
                                            hp.initializer)
            self.vars.append(self.w_noise)

    def Eval(self, x, train=True, summaries=False):
        with tf.variable_scope('NoisyTopKGating'):
            hp = self.hp
            clean_logits = tf.matmul(x, self.w_gate)
            if hp.noisy_gating:
                raw_noise_stddev = tf.matmul(x, self.w_noise)
                noise_stddev = ((tf.nn.softplus(raw_noise_stddev)+hp.noise_epsilon)*(tf.to_float(train)))
                noisy_logits = clean_logits + (tf.random_normal(tf.shape(clean_logits))*noise_stddev)
                logits = noisy_logits
                if summaries:
                    tf.summary.histogram('noisy_logits', noisy_logits)
                    tf.summary.histogram('noisy_stddev', noise_stddev)
            else:
                logits = clean_logits
            top_loits, top_indices = MyTopK(logits, min(hp.k+1, hp.num_experts))
            top_k_logits = tf.slice(top_loits, [0, 0], [-1, hp.k])
            top_k_indices = tf.slice(top_indices, [0, 0], [-1, hp.k])
            top_k_gates = tf.nn.softmax(top_k_logits)
            gates = RowiseUnsortedSegmentSum(top_k_gates, top_k_indices, hp.num_experts)
            if hp.noisy_gating and hp.k < hp.num_experts:
                load = tf.reduce_sum(ProbInTopK(clean_logits, noisy_logits, noisy_stddev, top_logits, hp.k), 0)
            else:
                load = GatesToLoad(gates)
            if summaries:
                tf.summary.histogram('importance', tf.reduce_sum(gates, 0))
                tf.summary.histogram('load', load)

        return gates, load

    @property
    def vars(self):
        return self.vars

class LocalMixtureOfExperts():
    def __init__(self, gating_hp, expert_hp, input_size, output_size, name):
        """Create a LocalMixtureOfExperts.
        Args:
          gating_hp: hyperparameters for the gating network.
            e.g. NoisyTopKGatingParams()
          expert_hp: hyperparameters for the expert networks.
            e.g. FeedForwardExpertParams()
          input_size: an integer.
          output_size: an integer.
          name: a string.
        """
        self._name = name
        SetInputOutputSizes(gating_hp, input_size, None)
        SetInputOutputSizes(expert_hp, input_size, output_size)
        self.gating_hp = gating_hp
        self.gating = gating_hp.gating_class(gating_hp, name + '_gating')
        self.expert_hp = expert_hp
        self.experts = [
            expert_hp.expert_class(expert_hp, name + '_%d' % i)
            for i in xrange(gating_hp.num_experts)
        ]

    def Eval(self, x, train=True, per_example_multiplier=None, summaries=False, identifiers=None):
        gating_hp = self.gating_hp
        gates, load = self.gating.Eval(x, train, summaries)
        if per_example_multiplier is not None:
            gates *= tf.expand_dims(per_example_multiplier, 1)
        dispatcher = SparseDispatcher(gating_hp.num_experts, gates)
        expert_input = dispatcher.Dispatch(x)
        expert_output = [
            self.experts[i].Eval(expert_input[i])
            for i in xrange(gating_hp.num_experts)
        ]
        y = dispatcher.Combine(expert_output)
        if identifiers is not None:
            expert_to_identifiers = dispatcher.Dispatch(identifiers)
        else:
            expert_to_identifiers = None
        return (y, tf.reduce_sum(gates, 0), load, expert_to_identifiers,
                dispatcher.ExpertToGates())

    @property
    def vars(self):
        ret = []
        for x in self.experts:
            ret.extend(x.vars)
        ret.extend(self.gating_vars)
        return ret

class DistributedMixtureOfExperts():
    def __init__(self, primary_gating_hp, secondary_gating_hp, expert_hp,
                 input_size, output_size, expert_devices, name):
        self.name = name
        # fill in the missing values in the hyperparameters
        SetInputOutputSizes(primary_gating_hp, input_size, None)
        SetInputOutputSizes(expert_hp, input_size, output_size)
        self.is_hierarchical = secondary_gating_hp is not None
        self.primary_gating_hp = primary_gating_hp
        self.primary_gating = primary_gating_hp.gating_class(
            primary_gating_hp, name + '_primary_gating')
        n1 = self.primary_gating_hp.num_experts
        # round robin assignment of experts to devices.
        expert_devices = [
            expert_devices[i % len(expert_devices)] for i in xrange(n1)
        ]
        self.expert_devices = expert_devices
        self.all_vars = []
        self.all_vars.extend(self.primary_gating.vars)
        if self.is_hierarchical:
            # hierarchical MoE
            self.secondary_moe = []
            for i in xrange(n1):
                with tf.device(expert_devices[i]):
                    secondary_moe = LocalMixtureOfExperts(secondary_gating_hp, expert_hp,
                                                          input_size, output_size,
                                                          '%s_secondary_%d' % (name, i))
                    self.secondary_moe.append(secondary_moe)
                    self.all_vars.extend(secondary_moe.vars)
        else:
            # flat MoE
            self.experts = []
            for i in xrange(n1):
                with tf.device(expert_devices[i]):
                    expert = expert_hp.expert_class(expert_hp, name + '_%d' % i)
                    self.experts.append(expert)
                    self.all_vars.extend(expert.vars)

    def Eval(self, datashard_devices, xs, train=True, summaries=False, identifiers=None, shadow_xs=None):
        n1 = self.primary_gating_hp.num_experts
        epsilon = 1e-10
        assert len(datashard_devices) == len(xs)
        num_datashards = len(xs)
        expert_devices = self.expert_devices
        has_identifiers = identifiers is not None
        # pylint: disable=unbalanced-tuple-unpacking
        primary_gates, primary_smooth_load = Parallel(
            datashard_devices, self.primary_gating.Eval, xs, train,
            [summaries] + [False] * (num_datashards - 1))
        primary_importance = tf.add_n(
            Parallel(datashard_devices, tf.reduce_sum, primary_gates, 0))
        primary_smooth_load = tf.add_n(primary_smooth_load)
        primary_true_load = tf.add_n(
            Parallel(datashard_devices, GatesToLoad, primary_gates))
        primary_dispatcher = DistributedSparseDispatcher(
            datashard_devices, expert_devices, primary_gates)

        if shadow_xs is None:
            secondary_input = primary_dispatcher.Dispatch(xs)
        else:
            secondary_input = primary_dispatcher.Dispatch(shadow_xs)

        primary_expert_to_identifiers = (primary_dispatcher.Dispatch(identifiers)
                                         if has_identifiers else None)
        primary_expert_to_gates = primary_dispatcher.ExpertToGates()
        if not self.is_hierarchical:
            # one-level distributed mixture of experts
            secondary_output = Parallel(expert_devices, lambda a, b: a.Eval(b),
                                        self.experts, secondary_input)
            ys = primary_dispatcher.Combine(secondary_output)
            return (ys, primary_importance, primary_smooth_load,
                    primary_expert_to_identifiers, primary_expert_to_gates)
        # two-level hierarchical MoE
        (secondary_output, secondary_importance, secondary_load,
         secondary_expert_to_identifiers, secondary_expert_to_gates) = (Parallel(
            expert_devices, [m.Eval for m in self.secondary_moe], secondary_input,
            train, primary_expert_to_gates, [summaries] + [False] * (n1 - 1),
            primary_expert_to_identifiers))
        # pylint: enable=unbalanced-tuple-unpacking
        ys = primary_dispatcher.Combine(secondary_output, multiply_by_gates=False)
        importance = tf.stack(secondary_importance)
        load = tf.stack(secondary_load) * tf.expand_dims(primary_smooth_load / (
                primary_true_load + epsilon), 1)
        expert_to_identifiers = []
        if identifiers is not None:
            for el in secondary_expert_to_identifiers:
                expert_to_identifiers.extend(el)
        expert_to_gates = []
        for el in secondary_expert_to_gates:
            expert_to_gates.extend(el)
        return (ys, importance, load, expert_to_identifiers, expert_to_gates)

    @property
    def vars(self):
        return self.all_vars

class SparseDispatcher():
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher.
        Args:
          num_experts: an integer.
          gates: a `Tensor` of shape `[batch_size, num_experts]`.
        Returns:
          a SparseDispatcher
        """
        self.gates = gates
        self.num_experts = num_experts

        where = tf.to_int32(tf.where(tf.transpose(gates) > 0))
        self.expert_index, self.batch_index = tf.unstack(where, num=2, axis=1)
        self.part_sizes_tensor = tf.reduce_sum(tf.to_int32(gates > 0), [0])
        self.nonzero_gates = tf.gather(
            tf.reshape(self.gates, [-1]),
            self.batch_index * num_experts + self.expert_index)

    def Dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape '[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        inp = tf.gather(inp, self.batch_index)
        return tf.split(inp, self.part_sizes_tensor, 0)

    def Combine(self, expert_out, multiply_by_gates=True):
        stitched = ConvertGradientToTensor(tf.concat(expert_out, 0))
        if multiply_by_gates:
            stitched *= tf.expand_dims(self.nonzero_gates, 1)
        combined = tf.unsorted_segment_sum(stitched, self.batch_index,
                                           tf.shape(self.gates)[0])
        return combined

    def ExpertToGates(self):
        return tf.split(self.nonzero_gates, self.part_sizes_tensor, 0)

    @property
    def part_sizes(self):
        return self.part_sizes_tensor

class DistributedSparseDispatcher():
    def __init__(self, datashard_devices, expert_devices, gates):
        self.gates = gates
        self.num_experts = len(expert_devices)
        assert len(gates) == len(datashard_devices)
        self.num_datashards = len(gates)
        self.datashard_devices = datashard_devices
        self.expert_devices = expert_devices
        self.dispatchers = Parallel(self.datashard_devices, SparseDispatcher, self.num_experts, gates)

    def Dispatch(self, inp):
        dispatched = Parallel(self.datashard_devices, lambda a, b: a.Dispatch(b),
                              self.dispatchers, inp)
        ret = Parallel(self.expert_devices, tf.concat,
                       TransposeListOfLists(dispatched), 0)
        if ret[0].dtype == tf.float32:
            # see comments on ConvertGradientToTensor
            ret = Parallel(self.expert_devices, ConvertGradientToTensor, ret)
        return ret

    def Combine(self, expert_out, multiply_by_gates=True):
        expert_part_sizes = tf.unstack(
            tf.stack([
                self.dispatchers[d].part_sizes
                for d in xrange(self.num_datashards)
            ]),
            num=self.num_experts,
            axis=1)
        # list of lists of shape [num_experts][num_datashards]
        expert_output_parts = Parallel(self.expert_devices, tf.split, expert_out,
                                       expert_part_sizes)
        expert_output_parts_t = TransposeListOfLists(expert_output_parts)
        ret = []
        for d in xrange(self.num_datashards):
            with tf.device(self.datashard_devices[d]):
                ret.append(self.dispatchers[d].Combine(
                    # see comments on ConvertGradientToTensor
                    ConvertGradientToTensor(tf.concat(expert_output_parts_t[d], 0)),
                    multiply_by_gates=multiply_by_gates))
        return ret

    def ExpertToGates(self):
        return Parallel(self.expert_devices, tf.concat,
                        TransposeListOfLists(
                            Parallel(self.datashard_devices, [
                                self.dispatchers[d].ExpertToGates
                                for d in xrange(self.num_datashards)
                            ])), 0)

def TransposeListofLists(lol):
    assert lol, 'cannot pass the empty list'
    return [list(x) for x in zip(*lol)]

class DistributedSingleDispathcer():
    def __init__(self, data_parallelism, model_parallelism, gates):
        gates = data_parallelism(tf.to_int32, gates)
        self._gates = gates
        self._data_parallelism = data_parallelism
        self._model_parallelism = model_parallelism

        # Compute the sizes number of examples going from each datashard to each
        # expert.
        def _PartSizes(gates):
            return tf.unsorted_segment_sum(
                tf.ones_like(gates), gates, model_parallelism.n)

        part_sizes_by_datashard = data_parallelism(_PartSizes, gates)
        self._part_sizes_by_expert = tf.unstack(
            tf.stack(part_sizes_by_datashard), num=model_parallelism.n, axis=1)

        # These indices will be used to combine the output on the datashards.
        def _StitchIndices(gates):
            return tf.dynamic_partition(
                tf.range(tf.size(gates)), gates, model_parallelism.n)

        self._stitch_indices = data_parallelism(_StitchIndices, gates)

    def Dispatch(self, d_tensors):
        parts = self._data_parallelism(tf.dynamic_partition, d_tensors, self._gates,
                                       self._model_parallelism.n)
        parts_by_expert = TransposeListOfLists(parts)
        x_tensors = self._model_parallelism(tf.concat, parts_by_expert, 0)
        return x_tensors

    def Combine(self, x_tensors):
        parts = self._model_parallelism(tf.split, x_tensors,
                                        self._part_sizes_by_expert)
        d_tensors = self._data_parallelism(tf.dynamic_stitch, self._stitch_indices,
                                           TransposeListOfLists(parts))
        return d_tensors

def ParallelEmbeddingLookup(params, ids, data_parallelism):
    param_devices = [x.device for x in params]
    model_parallelism = Parallelism(param_devices)
    num_shards = len(param_devices)
    # pylint: disable=unbalanced-tuple-unpacking
    ids, unique_idx = data_parallelism(tf.unique, ids)
    # pylint: enable=unbalanced-tuple-unpacking
    gates = data_parallelism(tf.mod, ids, num_shards)
    ids_div = data_parallelism(tf.div, ids, num_shards)
    dispatcher = DistributedSingleDispatcher(data_parallelism, model_parallelism,
                                             gates)
    x_ids_div = dispatcher.Dispatch(ids_div)
    params = model_parallelism(ConvertGradientToTensor, params)
    x_emb = model_parallelism(tf.gather, params, x_ids_div)
    r_emb = dispatcher.Combine(x_emb)
    r_emb = data_parallelism(tf.gather, r_emb, unique_idx)
    return r_emb

def SampledSoftmaxLoss(features, sampler, num_classes, target_classes, target_params, sampled_classes, sampled_params):
    sampled_logits = (tf.matmul(features, sampled_params, transpose_b=True) -
                      sampler.log_expected_count(sampled_classes))
    target_logits = (tf.reduce_sum(target_params * features, 1) -
                     sampler.log_expected_count(target_classes))
    sampled_log_denominator = tf.reduce_logsumexp(
        sampled_logits, [1], name='SampledLogDenominator')
    sampled_classes_mask = tf.unsorted_segment_sum(
        tf.fill(tf.shape(sampled_classes), float('-inf')), sampled_classes,
        num_classes)
    target_log_denominator = (
            target_logits + tf.gather(sampled_classes_mask, target_classes))
    combined_log_denominator = tf.reduce_logsumexp(
        tf.stack([sampled_log_denominator, target_log_denominator]), [0])
    loss = combined_log_denominator - target_logits
    return loss

def ParallelSampledSoftmaxLoss(params, features, target_classes, sampler, num_classes, data_parallelism, target_weights=None):
    sampled_classes = data_parallelism(sampler.sample)
    sampled_params = ParallelEmbeddingLookup(params, sampled_classes,
                                             data_parallelism)
    target_params = ParallelEmbeddingLookup(params, target_classes,
                                            data_parallelism)
    ret = data_parallelism(SampledSoftmaxLoss, features, sampler, num_classes,
                           target_classes, target_params, sampled_classes,
                           sampled_params)
    if target_weights is not None:
        ret = data_parallelism(tf.multiply, ret, target_weights)
    ret = data_parallelism(tf.reduce_sum, ret)
    ret = tf.add_n(ret)
    return ret