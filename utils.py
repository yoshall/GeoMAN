import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import numpy as np

def Linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError(
                "linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            "kernel", [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(
                    0.0, dtype=dtype)
            biases = vs.get_variable(
                "bias", [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)

def basic_hyperparams():
    return tf.contrib.training.HParams(
        # GPU arguments
        gpu_id='0',

        # model parameters
        learning_rate=1e-3,
        lambda_l2_reg=1e-3,
        gc_rate=2.5,  # to avoid gradient exploding
        dropout_rate=0.3,
        n_stacked_layers=2,
        s_attn_flag=2,
        ext_flag=True,

        # encoder parameter
        n_sensors=35,
        n_input_encoder=19,
        n_steps_encoder=12,  # time steps
        n_hidden_encoder=64,  # size of hidden units

        # decoder parameter
        n_input_decoder=1,
        n_external_input=83,
        n_steps_decoder=6,
        n_hidden_decoder=64,
        n_output_decoder=1  # size of the decoder output
    )

def count_total_params():
    """ count the parameters in the model """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def load_data(input_path, mode, n_steps_encoder, n_steps_decoder):
    """ load training/validation data 
    Args:
        input_path:
        mode: "train" or "test"
        n_steps_encoder: length of encoder, i.e., how many historical time steps we use for predictions
        n_steps_decoder: length of decoder, i.e., how many future time steps we predict
    Return:
        a list
    """
    mode_local_inp = np.load(
        input_path + "GeoMAN-{}-{}-{}-local_inputs.npy".format(n_steps_encoder, n_steps_decoder, mode))
    global_attn_index = np.load(
        input_path + "GeoMAN-{}-{}-{}-global_attn_state_indics.npy".format(n_steps_encoder, n_steps_decoder, mode))
    global_inp_index = np.load(
        input_path + "GeoMAN-{}-{}-{}-global_input_indics.npy".format(n_steps_encoder, n_steps_decoder, mode))
    mode_ext_inp = np.load(
        input_path + "GeoMAN-{}-{}-{}-external_inputs.npy".format(n_steps_encoder, n_steps_decoder, mode))
    mode_labels = np.load(
        input_path + "GeoMAN-{}-{}-{}-decoder_gts.npy".format(n_steps_encoder, n_steps_decoder, mode))
    return [mode_local_inp, global_inp_index, global_attn_index, mode_ext_inp, mode_labels]

def shuffle_data(training_data):
    """ shuffle data"""
    shuffle_index = np.random.permutation(training_data[0].shape[0])
    new_training_data = []
    for inp in training_data:
        new_training_data.append(inp[shuffle_index])
    return new_training_data

def get_batch_feed_dict(model, k, batch_size, training_data, global_inputs, global_attn_states):
    """ get feed_dict of each batch in a training epoch"""
    train_local_inp = training_data[0]
    train_global_inp = training_data[1]
    train_global_attn_ind = training_data[2]
    train_ext_inp = training_data[3]
    train_labels = training_data[4]
    n_steps_encoder = train_local_inp.shape[1]

    batch_local_inp = train_local_inp[k:k + batch_size]
    batch_ext_inp = train_ext_inp[k:k + batch_size]
    batch_labels = train_labels[k:k + batch_size]
    batch_labels = np.expand_dims(batch_labels, axis=2)
    batch_global_inp = train_global_inp[k:k + batch_size]
    batch_global_attn = train_global_attn_ind[k:k + batch_size]
    tmp = []
    for j in batch_global_inp:
        tmp.append(
            global_inputs[j: j + n_steps_encoder, :])
    tmp = np.array(tmp)
    feed_dict = {model.phs['local_inputs']: batch_local_inp,
                 model.phs['global_inputs']: tmp,
                 model.phs['local_attn_states']: np.swapaxes(batch_local_inp, 1, 2),
                 model.phs['global_attn_states']: global_attn_states[batch_global_attn],
                 model.phs['external_inputs']: batch_ext_inp,
                 model.phs['labels']: batch_labels}
    return feed_dict

def load_global_inputs(input_path, n_steps_encoder, n_steps_decoder):
    """ load global inputs"""
    global_inputs = np.load(
        input_path + "GeoMAN-{}-{}-global_inputs.npy".format(n_steps_encoder, n_steps_decoder))
    global_attn_states = np.load(
        input_path + "GeoMAN-{}-{}-global_attn_state.npy".format(n_steps_encoder, n_steps_decoder))
    return global_inputs, global_attn_states

def get_valid_batch_feed_dict(model, valid_indexes, k, valid_data, global_inputs, global_attn_states):
    """ get feed_dict of each batch in the validation set"""
    valid_local_inp = valid_data[0]
    valid_global_inp = valid_data[1]
    valid_global_attn_ind = valid_data[2]
    valid_ext_inp = valid_data[3]
    valid_labels = valid_data[4]
    n_steps_encoder = valid_local_inp.shape[1]

    batch_local_inp = valid_local_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_ext_inp = valid_ext_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_labels = valid_labels[valid_indexes[k]:valid_indexes[k + 1]]
    batch_labels = np.expand_dims(batch_labels, axis=2)
    batch_global_inp = valid_global_inp[valid_indexes[k]:valid_indexes[k + 1]]
    batch_global_attn = valid_global_attn_ind[valid_indexes[k]:valid_indexes[k + 1]]
    tmp = []
    for j in batch_global_inp:
        tmp.append(
            global_inputs[j: j + n_steps_encoder, :])
    tmp = np.array(tmp)
    feed_dict = {model.phs['local_inputs']: batch_local_inp,
                 model.phs['global_inputs']: tmp,
                 model.phs['local_attn_states']: np.swapaxes(batch_local_inp, 1, 2),
                 model.phs['global_attn_states']: global_attn_states[batch_global_attn],
                 model.phs['external_inputs']: batch_ext_inp,
                 model.phs['labels']: batch_labels}
    return feed_dict