import tensorflow as tf

class DCGAN:
    def __init__(
        self,
        config=None,
        training=None,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if training is None:
            raise ValueError('Set training either to be True or False.')
        else:
            self.training = training

        self._tf_summary = {}
        self._tf_session = None

        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = gpu_memory_allow_growth 
        if gpu_memory_fraction is not None:
            self._tf_config.gpu_options.per_process_gpu_memory_fraction = (
                gpu_memory_fraction
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            with tf.variable_scope('generator'):
                self._build_generator_network()
            with tf.variable_scope('discriminator'):
                self._build_discriminator_network()
            with tf.variable_scope('summary'):
                self._build_summary_ops()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def _build_generator_network(self, minibatch_size=None):
        conf = self._config['generator']
        prev_layer = None

        for layer_name, layer_conf in conf:
            with tf.variable_scope(layer_name):
#                if 'input' in layer_name:
#                    new_layer = tf.placeholder(
#                        dtype=tf.float32,
#                        shape=(None, conf['input']['dim']),
#                        name='Z',
#                    )
#                elif 'fc' in layer_name:
#                    minibatch_size, in_chs = prev_layer.shape.as_list() 
#                    out_size = layer_conf['out_size']
#                    out_chs = layer_conf['out_chs']
#
#                    W = tf.get_variable(
#                        name='W',
#                        shape=(in_chs, out_size * out_size * out_chs),
#                        initializer=self._get_variable_initializer(),
#                    )
#
#                    pre_activation = tf.reshape(
#                        tf.nn.matmul(prev_layer, W),
#                        shape=(minibatch_size, out_size, out_size, out_chs),
#                        name='pre_activation',
#                    )
#
#                    new_layer = tf.reshape(
#                        activation,
#                        shape=(minibatch_size, out_size, out_size, out_chs),
#                        name='output',
#                    )
#                elif 'conv' in layer_name:

                filter_size = layer_conf['filter_size']
                stride = layer_conf['stride']
                out_chs = layer_conf['out_chs']
                

                if 'input' in layer_name:
                    in_chs = conf['input_dim']
                    prev_layer = tf.placeholder(
                        dtype=tf.float32,
                        shape=(minibatch_size, 1, 1, in_chs),
                        name='Z',
                    )
                    out_size = filter_size
                else:
                    _, _, in_size, in_chs = prev_layer.shape.as_list()
                    out_size =  in_size * stride

                W = tf.get_variable(
                    name='W',
                    shape=(filter_size,
                           filter_size,
                           in_chs,
                           out_chs),
                    initializer=self._get_variable_initializer(),
                )

                pre_activation = tf.nn.conv2d_transpose(
                    prev_layer,
                    W,
                    output_shape=(minibatch_size,
                                  out_size,
                                  out_size,
                                  out_chs),
                    strides=(1, stride, stride, 1),
                    padding='SAME',
                    name='pre_activation',
                )

                if 'output' not in layer_name:
                    new_layer = tf.nn.relu(
                        batch_normalization(
                            input_tensor=pre_activation,
                            training=self.training,
                        ),
                        name='activation',
                    )
                else:
                    new_layer = tf.tanh(pre_activation, name='output')
                    
            # End of layer_name variable scope.

            prev_layer = new_layer

        # End of conf for loop.

    def _build_discriminator_network(self, minibatch_size=None):
        conf = self._config['discriminator']
        prev_layer = None

        for layer_name, layer_conf in conf:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    input_size = conf['input_size']
                    in_chs = conf['in_chs']
                    prev_layer = tf.placeholder(
                        dtype=tf.float32,
                        shape=(minibatch_size, input_size, input_size, in_chs),
                        name='input',
                    )
                else:
                    _, _, _, in_chs = prev_layer.shape.as_list()

                filter_size = layer_conf['filter_size']
                stride = layer_conf['stride']
                out_chs = layer_conf['out_chs']

                W = tf.get_variable(
                    name='W',
                    shape=(filter_size,
                           filter_size,
                           in_chs,
                           out_chs),
                    initializer=self._get_variable_initializer(),
                )

                pre_activation = tf.nn.conv2d(
                    prev_layer,
                    W,
                    strides=(1, stride, stride, 1),
                    padding='SAME',
                    name='pre_activation',
                )

                if 'input' in layer_name:
                    new_layer = leaky_relu(pre_activation)
                elif 'output' not in layer_name:
                    new_layer = leaky_relu(
                        batch_normalization(
                            input_tensor=pre_activation,
                            training=self.training,
                        ),
                    )
                else:
                    new_layer = tf.reshape(
                        tf.sigmoid(pre_activation),
                        [],
                        name='output',
                    )

            # End of layer_name variable scope.

            prev_layer = new_layer



def batch_normalization(
    input_tensor,
    momentum=0.99
    epsilon=0.001,
    training=False,
    use_layers_api=False,
):
    x = input_tensor
    x_shape = x.shape.as_list()[1:]

    if use_layers_api:
        name_prefix = 
        y = tf.layers.batch_normalization(
            x,
            momentum=momentum,
            epsilon=epsilon,
            training=training,
            name='layers_batch_normalization',
        )
    else:
        name_prefix = 
        if training:
            name_suffix = 'training'
        else:
            name_suffix = 'inference'

        with tf.variable_scope('nn_batch_normalization'):
            # TODO: Use tf.train.ExponentialMovingAverage(decay=momentum)
            inf_mean = tf.get_variable(
                name='inf_mean',
                shape=x_shape,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            inf_variance = tf.get_variable(
                name='inf_variance',
                shape=x_shape,
                initializer=tf.ones_initializer(),
                trainable=False,
            )
            x_scale = tf.get_variable(
                name='scale',
                shape=x_shape,
                initializer=tf.ones_initializer(),
            )
            x_offset = tf.get_variable(
                name='offset',
                shape=x_shape,
                initializer=tf.zeros_initializer(),
            )
            if training:
                with tf.variable_scope('training'):
                    x_mean, x_variance = tf.nn.moments(x, axes=[0])
                    inf_mean_op = get_exponential_moving_average_op(
                        x_mean,
                        inf_mean,
                        momentum,
                    )
                    inf_variance_op = get_exponential_moving_average_op(
                        x_variance,
                        inf_variance,
                        momentum,
                    )

                    with tf.control_dependencies(
                        [inf_mean_op. inf_variance_op]
                    ):
                        y = tf.nn.batch_normalization(
                            x,
                            x_mean,
                            x_variance,
                            x_offset,
                            x_scale,
                            epsilon,
                        )
            else:
                with tf.variable_scope('inference'):
                    y = tf.nn.batch_normalization(
                        x,
                        inf_mean,
                        inf_variance,
                        x_offset,
                        x_scale,
                        epsilon,
                    )
    return y


def get_exponential_moving_average_op(v_in, v_avg, decay):
    return tf.assign(v_avg, v_avg * decay + v_in * (1 - decay))

def leaky_relu(x, alpha=0.2):
    return tf.maximum(
        -x * alpha,
        x,
        name='leaky_relu',
    )
