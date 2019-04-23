from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.ops.rnn_cell_impl import *

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _linear(args,
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
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
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
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
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
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


_MRUStateTuple = collections.namedtuple("MRUStateTuple", ("x_p", "c"))


class MRUStateTuple(_MRUStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (x_p, c) = self
        if x_p.dtype != c.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(x_p.dtype), str(c.dtype)))
        return x_p.dtype


class MRUCell(RNNCell):
    """Basic MRU recurrent network cell.
    """

    def __init__(self, num_units,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None):
        """Initialize the basic MRU cell.
        """
        super(MRUCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer

    @property
    def state_size(self):
        return (MRUStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            x_p, c = state
        else:
            x_p, c = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        with vs.variable_scope("propagate_gate"):
            p_i = _linear([inputs, x_p], 2 * self._num_units, True, kernel_initializer=self._kernel_initializer)
            p, i = array_ops.split(value=p_i, num_or_size_splits=2, axis=1)

        with vs.variable_scope("candidate"):
            candidate_c = _linear(inputs, self._num_units, False, kernel_initializer=self._kernel_initializer)
            u_c_diag = vs.get_variable("u_c_diag", shape=[self._num_units], dtype=inputs.dtype)
            candidate_c = self._activation(candidate_c + u_c_diag * c)

        with vs.variable_scope("content_weight"):
          w_p_diag = vs.get_variable("w_p_diag", shape=[self._num_units], dtype=inputs.dtype)
          w_i_diag = vs.get_variable("w_i_diag", shape=[self._num_units], dtype=inputs.dtype)

        new_c = c * sigmoid(p + self._forget_bias + w_p_diag * c) + sigmoid(i + w_i_diag * c) * candidate_c
        new_c = self._activation(new_c)


        if self._state_is_tuple:
            new_state = MRUStateTuple(inputs, new_c)
        else:
            new_state = array_ops.concat([inputs, new_c], 1)

        return new_c, new_state
