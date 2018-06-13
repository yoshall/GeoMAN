# -*- coding: utf-8 -*-
"""
Yuxuan Liang, Songyu Ke, Junbo Zhang, Xiuwen Yi, Yu Zheng
GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction
27th International Joint Conference on Artificial Intelligence (IJCAI 2018)
-------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from base_model import BaseModel
from tensorflow.contrib import rnn
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from six.moves import xrange
from six.moves import zip
from utils import Linear

def input_transform(local_inputs,
                    global_inputs,
                    external_inputs,
                    local_attn_states,
                    global_attn_states,
                    labels):
    n_steps_decoder = labels.get_shape()[1].value
    n_output_decoder = labels.get_shape()[2].value
    n_sensors = global_inputs.get_shape()[2].value
    n_steps_encoder = local_inputs.get_shape()[1].value
    n_input_encoder = local_inputs.get_shape()[2].value
    n_external_input = external_inputs.get_shape()[2].value

    # a tuple composed of the local and global attention states
    encoder_attention_states = (local_attn_states,
                                global_attn_states)

    # transform the inputs from local and global view into encoder_inputs
    _local_inputs = tf.transpose(local_inputs, [1, 0, 2])
    _local_inputs = tf.reshape(_local_inputs, [-1, n_input_encoder])
    _local_inputs = tf.split(_local_inputs, n_steps_encoder, 0)
    _global_inputs = tf.transpose(global_inputs, [1, 0, 2])
    _global_inputs = tf.reshape(_global_inputs, [-1, n_sensors])
    _global_inputs = tf.split(_global_inputs, n_steps_encoder, 0)
    encoder_inputs = (_local_inputs, _global_inputs)

    # transform the variables into lists as the input of different function
    _labels = tf.transpose(labels, [1, 0, 2])
    _labels = tf.reshape(_labels, [-1, n_output_decoder])
    _labels = tf.split(_labels, n_steps_decoder, 0)
    _external_inputs = tf.transpose(external_inputs, [1, 0, 2])
    _external_inputs = tf.reshape(_external_inputs, [-1, n_external_input])
    _external_inputs = tf.split(_external_inputs, n_steps_decoder, 0)

    # not useful when the loop function is employed
    decoder_inputs = [tf.zeros_like(
        _labels[0], dtype=tf.float32, name="GO")] + _labels[:-1]
    return encoder_attention_states, encoder_inputs, _labels, _external_inputs, decoder_inputs

class GeoMAN(BaseModel):
    def __init__(self, hps, mode='train'):
        super(GeoMAN, self).__init__(hps, mode)
        preds = self.mod_fn()
        self.phs['preds'] = preds
        self.phs['loss'] = self.get_loss()  # see at eq.[11]
        tf.add_to_collection('loss', self.phs['loss'])
        self.phs['train_op'] = self.train_op()
        self.phs['summary'] = self.summary()

    def spatial_attention(self,
                          encoder_inputs,
                          attention_states,
                          cell,
                          s_attn_flag=2,
                          output_size=None,
                          dtype=dtypes.float32,
                          scope=None):
        """ Spaital attention in GeoMAN
            Args:
                encoder_inputs: A tuple consisting of
                  1) local_inputs: the inputs of local spatial attention, i.e., a list of 2D tensors with the shape of
                    [batch_size, n_inputs_encoder]
                  2) global_inputs: the inputs of local spatial attention, i.e., a list of 2D tensors with the shape of
                    [batch_size, n_sensors]
                attention_states: A tuple consisting of
                  1) local_attention_states: 3D tensor [batch_size, n_input_encoder, n_steps_encoder]
                  2) global_attention_states: 4D tensor [batch_size, n_sensors, n_input_encoder, n_steps_encoder]
                cell: core_rnn_cell.RNNCell defining the cell function and size.
                s_attn_flag: 0: only local. 1: only global. 2: local + global.
                output_size: Size of the output vectors; if None, we use cell.output_size.
                loop_function: the loop function we use.
                dtype: The dtype to use for the RNN initial state (default: tf.float32).
                scope: VariableScope for the created subgraph; default: "spatial_attention".
            Return:
                A tuple of the form (outputs, state), where:
            Raises:
                ValueError: when num_heads is not positive, there are no inputs, shapes
                  of attention_states are not set, or input size cannot be inferred from the
                  input.
        """
        # check inputs
        if not encoder_inputs:
            raise ValueError(
                "Must provide at least 1 input to attention encoder.")
        local_inputs = encoder_inputs[0]
        global_inputs = encoder_inputs[1]
        if output_size is None:
            output_size = cell.output_size
        local_attention_states = attention_states[0]
        global_attention_states = attention_states[1]
        if not local_attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError("Shape[1] and [2] of local_attention_states must be known: %s"
                             % local_attention_states.get_shape())
        if not global_attention_states.get_shape()[1:3].is_fully_defined():
            raise ValueError("Shape[1] to [3] of global_attention_states must be known: %s"
                             % global_attention_states.get_shape())
        batch_size = array_ops.shape(local_inputs[0])[0]

        # decide whether to use local/global attention
        # s_attn_flag: 0: only local. 1: only global. 2: local + global
        local_flag = True
        global_flag = True
        if s_attn_flag == 0:
            global_flag = False
        elif s_attn_flag == 1:
            local_flag = False

        with vs.variable_scope(scope or "spatial_attn"):
            if local_flag:
                # implement of local spatial attention
                with tf.variable_scope('local_spatial_attn'):
                    local_attn_length = local_attention_states.get_shape()[
                        1].value  # n_input_encoder
                    local_attn_size = local_attention_states.get_shape()[
                        2].value  # n_steps_encoder

                    # A trick: to calculate U_l * x^{i,k} by a 1-by-1 convolution
                    # refer: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
                    # See at eq.[1] in the paper
                    local_hidden = array_ops.reshape(
                        local_attention_states, [-1, local_attn_length, 1, local_attn_size])
                    # Size of query vectors for attention.
                    local_attention_vec_size = local_attn_size
                    local_u = vs.get_variable("AttnUl",
                                              [1, 1, local_attn_size, local_attention_vec_size])  # U_l
                    local_hidden_features = nn_ops.conv2d(
                        local_hidden, local_u, [1, 1, 1, 1], "SAME")  # U_l * x^{i,k}
                    local_v = vs.get_variable(
                        "AttnVl", [local_attention_vec_size])  # v_l
                    batch_attn_size = array_ops.stack(
                        [batch_size, local_attn_length])
                    local_attn = array_ops.zeros(batch_attn_size, dtype=dtype)

                    def local_attention(query):
                        """Put attention masks on local_hidden using local_hidden_features and query."""
                        # If the query is a tuple (when stacked RNN/LSTM), flatten it
                        if nest.is_sequence(query):
                            query_list = nest.flatten(query)
                            for q in query_list:
                                ndims = q.get_shape().ndims
                                if ndims:
                                    assert ndims == 2
                            query = array_ops.concat(query_list, 1)
                        with tf.variable_scope("AttnWl"):
                            # linear map
                            y = Linear(query, local_attention_vec_size, True)
                            y = array_ops.reshape(
                                y, [-1, 1, 1, local_attention_vec_size])
                            # Attention mask is a softmax of v_l^{\top} * tanh(...)
                            s = math_ops.reduce_sum(
                                local_v * math_ops.tanh(local_hidden_features + y), [2, 3])
                            # Now calculate the attention-weighted vector, i.e., alpha in eq.[2]
                            a = nn_ops.softmax(s)
                        return a

            # implement of global spatial attention
            if global_flag:
                with tf.variable_scope('global_spatial_attn'):
                    global_attn_length = global_attention_states.get_shape()[
                        1].value  # n_sensor
                    global_n_input = global_attention_states.get_shape()[
                        2].value  # n_input_dim
                    global_attn_size = global_attention_states.get_shape()[
                        3].value  # n_steps_encoder, T

                    # This implement is a little bit different from the paper at IJCAI-18.
                    # See at eq.[3.5] (we have no place to label the equation T-T) in the paper
                    # Note that our global input X^l include the target series y^l at l-th sensor,
                    # we calculate W'_g * X^l * u_g by a convolution and omit U_g * y^l for simplicity.
                    # You can easily add U_g * y^l here, where y^l is the first column of local inputs.
                    global_hidden = array_ops.reshape(global_attention_states,
                                                      [-1, global_attn_length, global_n_input, global_attn_size])
                    # Size of query vectors for attention.
                    global_attention_vec_size = global_attn_size
                    global_k = vs.get_variable("AttnW_and_u",
                                               [1, global_n_input, global_attn_size,
                                                global_attention_vec_size])
                    # global_hidden_features with the shape (batch_size, global_attn_length, 1, global_attn_size)
                    global_hidden_features = nn_ops.conv2d(
                        global_hidden, global_k, [1, 1, 1, 1], "SAME")
                    global_v = vs.get_variable(
                        "AttnVg", [global_attention_vec_size])
                    batch_attn_size = array_ops.stack(
                        [batch_size, global_attn_length])
                    global_attn = array_ops.zeros(batch_attn_size, dtype=dtype)

                    def global_attention(query):
                        """Put attention masks on global_hidden using global_hidden_features and query."""
                        # If the query is a tuple (when stacked RNN/LSTM), flatten it
                        if nest.is_sequence(query):
                            query_list = nest.flatten(query)
                            for q in query_list:  # Check that ndims == 2 if specified.
                                ndims = q.get_shape().ndims
                                if ndims:
                                    assert ndims == 2
                            query = array_ops.concat(query_list, 1)
                        with tf.variable_scope("AttnWg"):
                            # linear map
                            y = Linear(query, global_attention_vec_size, True)
                            y = array_ops.reshape(
                                y, [-1, 1, 1, global_attention_vec_size])
                            # Attention mask is a softmax of v_g^{\top} * tanh(...)
                            s = math_ops.reduce_sum(
                                global_v * math_ops.tanh(global_hidden_features + y), [2, 3])
                            # Sometimes it's not easy to find a measurement to denote similarity between sensors,
                            # here we omit such prior knowledge in eq.[4].
                            # You can use "a = nn_ops.softmax((1-lambda)*s + lambda*sim)" to encode similarity info,
                            # where:
                            #     sim: a vector with length n_sensors, describing the sim between the target sensor and the others
                            #     lambda: a trade-off.
                            a = nn_ops.softmax(s)
                            # a = nn_ops.softmax((1 - lambda) * s + lambda * sim)
                        return a

            # how to get the initial_state
            initial_state_size = array_ops.stack([batch_size, output_size])
            initial_state_one = [array_ops.zeros(
                initial_state_size, dtype=dtype) for _ in xrange(2)]
            initial_state = [
                initial_state_one for _ in range(len(cell._cells))]
            state = initial_state

            outputs = []
            attn_weights = []
            i = 0
            # i is the index of the which time step
            # local_inp is numpy.array and the shape of local_inp is (batch_size, n_feature)
            for local_inp, global_inp in zip(local_inputs, global_inputs):
                if i > 0:
                    vs.get_variable_scope().reuse_variables()
                input_size = local_inp.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError(
                        "Could not infer input size from input: %s" % local_inp.name)

                if local_flag and global_flag:
                    # multiply attention weights with the original input
                    local_x = local_attn * local_inp
                    global_x = global_attn * global_inp
                    # Run the BasicLSTM with the newly input
                    cell_output, state = cell(
                        tf.concat([local_x, global_x], axis=1), state)
                    # Run the attention mechanism.
                    with tf.variable_scope('local_spatial_attn'):
                        local_attn = local_attention(state)
                    with tf.variable_scope('global_spatial_attn'):
                        global_attn = global_attention(state)
                    attn_weights.append((local_attn, global_attn))
                elif local_flag:
                    local_x = local_attn * local_inp
                    cell_output, state = cell(local_x, state)
                    with tf.variable_scope('local_spatial_attn'):
                        local_attn = local_attention(state)
                    attn_weights.append(local_attn)
                elif global_flag:
                    global_x = global_attn * global_inp
                    cell_output, state = cell(global_x, state)
                    with tf.variable_scope('global_spatial_attn'):
                        global_attn = global_attention(state)
                    attn_weights.append(global_attn)
                # Attention output projection
                with vs.variable_scope("AttnOutputProjection"):
                    output = cell_output
                outputs.append(output)
                i += 1
        return outputs, state, attn_weights

    def temporal_attention(self,
                           decoder_inputs,
                           external_inputs,
                           initial_state,
                           attention_states,
                           cell,
                           output_size=None,
                           loop_function=None,
                           dtype=tf.float32,
                           scope=None,
                           initial_state_attention=False,
                           external_flag=True):
        """ Temporal attention in GeoMAN
        Args:
            decoder_inputs: A list (length: n_steps_decoder) of 2D Tensors [batch_size, n_input_decoder].
            external_inputs: A list (length: n_steps_decoder) of 2D Tensors [batch_size, n_external_input].
            initial_state: 2D Tensor [batch_size, cell.state_size].
            attention_states: 3D Tensor [batch_size, n_step_encoder, n_hidden_encoder].
            cell: core_rnn_cell.RNNCell defining the cell function and size.
            output_size: Size of the output vectors; if None, we use cell.output_size.
            loop_function: the loop function we use. 
            dtype: The dtype to use for the RNN initial state (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "tempotal_attention".
            initial_state_attention: If False (default), initial attentions are zero.
            external_flag: whether to use external factors

        Return:
            A tuple of the form (outputs, state), where:
                outputs: A list of the same length as the inputs of decoder of 2D Tensors of
                         shape [batch_size x output_size]
                state: The state of each decoder cell the final time-step.
        """
        # check inputs
        if not decoder_inputs:
            raise ValueError(
                "Must provide at least 1 input to attention decoder.")
        if not external_inputs:
            raise ValueError(
                "Must provide at least 1 ext_input to attention decoder.")
        if attention_states.get_shape()[2].value is None:
            raise ValueError("Shape[2] of attention_states must be known: %s" %
                             attention_states.get_shape())
        if output_size is None:
            output_size = cell.output_size

        # implement of temporal attention
        with vs.variable_scope(
                scope or "temporal_attn", dtype=dtype) as scope:
            dtype = scope.dtype
            # Needed for reshaping.
            batch_size = array_ops.shape(decoder_inputs[0])[0]
            attn_length = attention_states.get_shape()[1].value
            if attn_length is None:
                attn_length = array_ops.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value

            # A trick: to calculate W_d * h_o by a 1-by-1 convolution
            # See at eq.[6] in the paper
            hidden = array_ops.reshape(attention_states,
                                       [-1, attn_length, 1, attn_size])  # need to reshape before
            # Size of query vectors for attention.
            attention_vec_size = attn_size
            w = vs.get_variable(
                "Attn_Wd", [1, 1, attn_size, attention_vec_size])  # W_d
            hidden_features = nn_ops.conv2d(
                hidden, w, [1, 1, 1, 1], "SAME")  # W_d * h_o
            v = vs.get_variable("Attn_v", [attention_vec_size])  # v_d
            state = initial_state

            def attention(query):
                """Put attention masks on local_hidden using local_hidden_features and query."""
                # If the query is a tuple (when stacked RNN/LSTM), flatten it
                if nest.is_sequence(query):
                    query_list = nest.flatten(query)
                    for q in query_list:  # Check that ndims == 2 if specified.
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = array_ops.concat(query_list, 1)
                with vs.variable_scope("Attn_Wpd"):
                    # linear map
                    y = Linear(query, attention_vec_size, True)
                    y = array_ops.reshape(
                        y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v_d^{\top} * tanh(...).
                    s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y),
                                            [2, 3])
                    # Now calculate the attention-weighted vector, i.e., gamma in eq.[7]
                    a = nn_ops.softmax(s)
                    # eq. [8]
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                return array_ops.reshape(d, [-1, attn_size])

            if initial_state_attention:
                attn = attention(initial_state)
            else:
                batch_attn_size = array_ops.stack([batch_size, attn_size])
                attn = array_ops.zeros(batch_attn_size, dtype=dtype)
                attn.set_shape([None, attn_size])

            i = 0
            outputs = []
            prev = None
            for inp, ext_inp in zip(decoder_inputs, external_inputs):
                if i > 0:
                    vs.get_variable_scope().reuse_variables()
                # If loop_function is set, we use it instead of decoder_inputs.
                if loop_function is not None and prev is not None:
                    with vs.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                # Merge input and previous attentions into one vector of the right size.
                input_size = inp.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError(
                        "Could not infer input size from input: %s" % inp.name)
                # we map the concatenation to shape [batch_size, input_size]
                if external_flag:
                    x = Linear([inp] + [ext_inp] + [attn], input_size, True)
                else:
                    x = Linear([inp] + [attn], input_size, True)
                # Run the RNN.
                cell_output, state = cell(x, state)
                # Run the attention mechanism.
                if i == 0 and initial_state_attention:
                    with vs.variable_scope(vs.get_variable_scope(), reuse=True):
                        attn = attention(state)
                else:
                    attn = attention(state)
                # Attention output projection
                with vs.variable_scope("AttnOutputProjection"):
                    output = Linear([cell_output] + [attn], output_size, True)
                if loop_function is not None:
                    prev = output
                outputs.append(output)
                i += 1
        return outputs, state

    def _loop_function(self, prev, _):
        """loop function used in the decoder to generate the next inupt"""
        return tf.matmul(prev, self.phs['w_out']) + self.phs['b_out']

    def mod_fn(self):
        encoder_attention_states, encoder_inputs, _labels, _external_inputs, decoder_inputs \
            = input_transform(self.phs['local_inputs'],
                              self.phs['global_inputs'],
                              self.phs['external_inputs'],
                              self.phs['local_attn_states'],
                              self.phs['global_attn_states'],
                              self.phs['labels'])

        n_stacked_layers = self.hps.n_stacked_layers  # num of layer stacked in RNN
        # dimension of encoder hidden/cell state
        n_hidden_encoder = self.hps.n_hidden_encoder
        # dimension of decoder hidden/cell state
        n_hidden_decoder = self.hps.n_hidden_decoder
        dropout_rate = self.hps.dropout_rate  # dropout rate in RNN unit
        n_output_decoder = self.hps.n_output_decoder

        # Define weights in the transformation layer of decoder
        self.phs['w_out'] = tf.get_variable('Weights_out',
                                            [n_hidden_decoder, n_output_decoder],
                                            dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer())

        self.phs['b_out'] = tf.get_variable('Biases_out',
                                            shape=[n_output_decoder],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.))

        with tf.variable_scope('GeoMAN'):
            # the implement of encoder
            with tf.variable_scope('Encoder'):
                cells = []
                for i in range(n_stacked_layers):
                    with tf.variable_scope('LSTM_{}'.format(i)):
                        cell = rnn.BasicLSTMCell(
                            n_hidden_encoder, forget_bias=1.0, state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=1.0 - dropout_rate)
                        cells.append(cell)
                encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
                encoder_outputs, encoder_state, attn_weights = self.spatial_attention(encoder_inputs,
                                                                                      encoder_attention_states,
                                                                                      encoder_cell,
                                                                                      self.hps.s_attn_flag)

            # Calculate a concatenation of encoder outputs to put attention on.
            top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size])
                          for e in encoder_outputs]
            attention_states = tf.concat(top_states, 1)

            # the implement of decoder
            with tf.variable_scope('Decoder'):
                cells = []
                for i in range(n_stacked_layers):
                    with tf.variable_scope('LSTM_{}'.format(i)):
                        cell = rnn.BasicLSTMCell(n_hidden_decoder,
                                                 forget_bias=1.0,
                                                 state_is_tuple=True)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                             output_keep_prob=1.0 - dropout_rate)
                        cells.append(cell)
                decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
                decoder_outputs, states = self.temporal_attention(decoder_inputs,
                                                                  _external_inputs,
                                                                  encoder_state,
                                                                  attention_states,
                                                                  decoder_cell,
                                                                  loop_function=self._loop_function,
                                                                  external_flag=self.hps.ext_flag)
            # generate outputs
            with tf.variable_scope('Prediction'):
                preds = [tf.matmul(i, self.phs['w_out']) +
                         self.phs['b_out'] for i in decoder_outputs]

        return preds

    def get_loss(self):
        """MSE loss"""
        # reshape
        n_steps_decoder = self.phs['labels'].get_shape()[1].value
        n_output_decoder = self.phs['labels'].get_shape()[2].value
        labels = tf.transpose(self.phs['labels'], [1, 0, 2])
        labels = tf.reshape(labels, [-1, n_output_decoder])
        labels = tf.split(labels, n_steps_decoder, 0)

        # compute empirical loss
        empirical_loss = 0
        # Extra: we can also get separate error at each future time slot
        for _y, _Y in zip(self.phs['preds'], labels):
            empirical_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))
        self.phs['empirical_loss'] = empirical_loss
        return empirical_loss

    def get_l2reg_loss(self):
        """l2 reg loss"""
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'kernel:' in tf_var.name or 'bias:' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        return self.lambda_l2_reg * reg_loss

    def train_op(self):
        # Training optimizer
        with tf.variable_scope('Optimizer'):
            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP,
                             tf.GraphKeys.GLOBAL_VARIABLES])
            optimizer = tf.contrib.layers.optimize_loss(
                loss=self.phs['loss'],
                learning_rate=self.hps.learning_rate,
                global_step=global_step,
                optimizer="Adam",
                clip_gradients=self.hps.gc_rate)
        return optimizer
    
    def summary(self):
        tf.summary.scalar("loss", self.phs['loss'])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        return tf.summary.merge_all()

