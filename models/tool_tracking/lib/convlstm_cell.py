#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weakly-supervised ConvLSTM Surgical Tool Tracker
================
------
**A re-implementation of surgical tool tracker in**<br>
<i>Nwoye, C. I., Mutter, D., Marescaux, J., & Padoy, N. (2019). 
    Weakly supervised convolutional LSTM approach for tool tracking in laparoscopic videos. 
    International journal of computer assisted radiology and surgery, 14(6), 1059-1067.<br></i>
    <b> Cite this paper, if you use this code or part of it.. </b>
    
    
(c) Research Group CAMMA, University of Strasbourg, France<br>
Website: http://camma.u-strasbg.fr<br>
Code author: Chinedu Nwoye <br>
    
-----
"""
# Lib built by TensorFlow Authors with slight modification from CAMMA Researchers
# ===============================================================================

import tensorflow as tf
import resnet_utils as utils

# TensorFlow 2.x compatibility
TF_VERSION = int(tf.__version__.split('.')[0])
if TF_VERSION >= 2:
    try:
        # Try to use tf_keras (Keras 2 for TF 2.16+)
        import tf_keras
        RNNCellBase = tf_keras.layers.AbstractRNNCell
        # Use tf_keras for LSTMStateTuple to avoid Keras 3 issues
        try:
            LSTMStateTuple = tf_keras.layers.LSTMCell.LSTMStateTuple
        except:
            # Create a simple namedtuple as fallback
            from collections import namedtuple
            LSTMStateTuple = namedtuple('LSTMStateTuple', ['c', 'h'])
    except ImportError:
        # Fallback to compat API
        RNNCellBase = tf.compat.v1.nn.rnn_cell.RNNCell
        LSTMStateTuple = tf.compat.v1.nn.rnn_cell.LSTMStateTuple
    tf_variable_scope = tf.compat.v1.variable_scope
    tf_get_variable = tf.compat.v1.get_variable
    tf_get_variable_scope = tf.compat.v1.get_variable_scope
    try:
        variance_scaling_initializer = tf.keras.initializers.VarianceScaling()
    except:
        variance_scaling_initializer = tf.compat.v1.keras.initializers.VarianceScaling()
else:
    RNNCellBase = tf.nn.rnn_cell.RNNCell
    tf_variable_scope = tf.variable_scope
    tf_get_variable = tf.get_variable
    tf_get_variable_scope = tf.get_variable_scope
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
    variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer()

class ConvLSTMCell(RNNCellBase):
    def __init__(self,
                 shape,
                 kernel,
                 depth,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None,
                 normalize=None,
                 dropout=None,
                 reuse=None):
        # Initialize parent - TF2 doesn't use _reuse
        if TF_VERSION >= 2:
            super(ConvLSTMCell, self).__init__()
        else:
            super(ConvLSTMCell, self).__init__(_reuse=reuse) 
        tf_shape = tf.TensorShape(shape + [depth])
        self._output_size = tf_shape
        self._state_size = LSTMStateTuple(tf_shape, tf_shape) 
        self._kernel = kernel
        self._depth = depth
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or tf.nn.tanh
        self._normalize = normalize
        self._dropout = dropout 
        self._w_conv = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None
 
    @property
    def state_size(self):
        return self._state_size
 
    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
            batch_size: int, the batch size
            dtype: data type for the state tensors

        Returns:
            LSTMStateTuple of zero tensors
        """
        # Get shape from state_size
        shape = self._state_size.c  # TensorShape from __init__
        # Create zero tensors with shape [batch_size] + shape
        zeros = tf.zeros([batch_size] + shape.as_list(), dtype=dtype)
        return LSTMStateTuple(zeros, zeros)

    def call(self, inputs, state):
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(4)[3]
        # In TF2, dimensions are ints directly, not Dimension objects
        if input_size is None:
            raise ValueError('Could not infer size from inputs.get_shape()[-1]')

        c_prev, h_prev = state
        inputs = tf.concat([inputs, h_prev], axis=-1)

        if not self._w_conv:
            scope = tf_get_variable_scope()
            # Add conv_lstm_cell scope for TF1 checkpoint compatibility
            with tf_variable_scope(scope, initializer=self._initializer):
                with tf_variable_scope('conv_lstm_cell'):
                    # In TF2, shape dimensions are ints directly
                    input_channels = inputs.shape[-1] if isinstance(inputs.shape[-1], int) else inputs.shape[-1].value
                    kernel_shape = self._kernel + [input_channels, 4 * self._depth]
                    self._w_conv = tf_get_variable('w_conv', shape=kernel_shape, dtype=dtype)
 
        # i = input_gate, j = new_input, f = forget_gate, o = ouput_gate
        conv = tf.nn.conv2d(inputs, self._w_conv, (1, 1, 1, 1), 'SAME')
        i, j, f, o = tf.split(conv, 4, axis=-1)
 
        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = tf_get_variable_scope()
            with tf_variable_scope(scope, initializer=self._initializer):
                self._w_f_diag = tf_get_variable('w_f_diag', c_prev.shape[1:], dtype=dtype)
                self._w_i_diag = tf_get_variable('w_i_diag', c_prev.shape[1:], dtype=dtype)
                self._w_o_diag = tf_get_variable('w_o_diag', c_prev.shape[1:], dtype=dtype)
 
        if self._use_peepholes:
            f = f + self._w_f_diag * c_prev
            i = i + self._w_i_diag * c_prev
        if self._normalize is not None:
            f = self._normalize(f, dim=None)
            i = self._normalize(i, dim=None)
            j = self._normalize(j, dim=None)
 
        j = self._activation(j)
 
        if self._dropout is not None:
            j = tf.nn.dropout(j, self._dropout)
 
        c = tf.nn.sigmoid(f + self._forget_bias) * c_prev + tf.nn.sigmoid(i) * j
 
        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            o = o + self._w_o_diag * c
        if self._normalize is not None:
            o = self._normalize(o, dim=None)
            c = self._normalize(c, dim=None)
 
        h = tf.nn.sigmoid(o) * self._activation(c)

        new_state = LSTMStateTuple(c, h)
        return h, new_state


class ConvLSTM(ConvLSTMCell):
    def __init__(self, filters, kernel=3, strides=1, is_training=tf.constant(True), scope='convlstm'):
        self._filters   = filters
        self._kernel    = kernel
        self._strides   = strides
        self._scope     = scope
        self._is_training = is_training
        self._prev_convlstm_state   = [0] 

        
    def _residual(self, x, seek):
        in_channel = x.get_shape().as_list()[-1]
        with tf_variable_scope(self._scope):
            # Shortcut connection
            if in_channel == self._filters:
                if self._strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, self._strides, self._strides, 1], [1, self._strides, self._strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, self._filters, self._strides, name='conv_1')
            # Residual convlstm           
            x = self._convlstm(x=x, seek=seek)   
            x = utils._bn(x, self._is_training, None, name='bn_1')
            x = x + shortcut
        return x
        

    def _convlstm(self, x, seek):
        with tf_variable_scope('convlstm') as scope: 
            # cell 
            x                   = tf.expand_dims(x, [0])                      
            convlstmcell        = ConvLSTMCell(shape=x.get_shape().as_list()[2:4],  kernel=[self._kernel, self._kernel], depth=self._filters, normalize=tf.nn.l2_normalize,
                                                initializer=variance_scaling_initializer, forget_bias=1.0, activation=tf.nn.tanh)
            # states
            zero_state          = tf.Variable(convlstmcell.zero_state(1, dtype=tf.float32), name='zero_convlstm_state')
            if self._prev_convlstm_state[0]  == 0: 
                self._prev_convlstm_state[0] = tf.Variable(convlstmcell.zero_state(1,  dtype=tf.float32), name='convlstm_state')
            # switch state condition
            convlstm_state      = tf.cond( tf.reduce_any(tf.equal(tf.cast(0,tf.int64), seek)), lambda:zero_state, lambda: self._prev_convlstm_state[0] )
            convlstm_state      = tf.unstack(convlstm_state, axis=0)
            convlstm_state      = LSTMStateTuple(convlstm_state[0], convlstm_state[1])
            # Run
            x, convlstm_state   = tf.compat.v1.nn.dynamic_rnn(cell=convlstmcell, inputs=x, dtype=tf.float32, initial_state=convlstm_state, scope='clstm') 
            # state update
            update_state_ops    = []
            cell_state          = convlstm_state.c
            hidden_state        = convlstm_state.h
            update_state_ops.append( self._prev_convlstm_state[0][0,:,:,:,:].assign(cell_state))
            update_state_ops.append( self._prev_convlstm_state[0][1,:,:,:,:].assign(hidden_state))
            update_state_op     = tf.group(*update_state_ops)            
            with tf.control_dependencies([update_state_op]):                           
                x = tf.squeeze(x, [0]) # reformat output   
            print('\tBuilding units: {}: -> {}'.format( scope.name, x.get_shape() ))
        return x

        
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf_get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x