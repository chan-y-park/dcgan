import os

import numpy as np
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

        self._config = config
        self._load_data()

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

            if self.training:
                with tf.variable('train'):
                    self._build_train_ops()

                with tf.variable_scope('summary'):
                    self._build_summary_ops()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def train(self):
        if not self.training:
            raise RuntimeError

        minibatch_size = self._config['minibatch_size']

        for i in range(self._config['num_training_iterations']):
            real_inputs = self.get_samples_from_data(minibatch_size)
            fake_inputs = self.generate_samples(minibatch_size)

            fetches = [
                self._tf_graph.get_tensor_by_name(
                    'training/real_D_loss:0',
                ),
                self._tf_graph.get_tensor_by_name(
                    'training/fake_D_loss:0',
                ),
                self._tf_graph.get_operation_by_name(
                    'train/minimize_D_loss',
                ),
            ]

            feed_dict = {
                self._tf_graph.get_tensor_by_name(
                    'training/real_D_logits:0',
                ): self.get_D_logits(real_inputs),
                self._tf_graph.get_tensor_by_name(
                    'training/fake_D_logits:0',
                ): self.get_D_logits(fake_inputs),
            }

            real_D_loss, fake_D_loss, _ = self._tf_session.run(
                fetches=fetches,
                feed_dict=feed_dict,
            )

            fetches = [
                self._tf_graph.get_tensor_by_name(
                    'training/fake_D_logits:0',
                ),
                self._tf_graph.get_tensor_by_name(
                    'training/G_loss:0',
                ),
                self._tf_graph.get_operation_by_name(
                    'train/minimize_G_loss',
                ),
            ]

            feed_dict = {
                self._tf_graph.get_tensor_by_name(
                    'training/fake_D_logits:0',
                ): self.get_D_logits(fake_inputs),
            }

            G_loss, _ = self._tf_session.run(
                fetches=fetches,
                feed_dict=feed_dict,
            )

    def _build_generator_network(self):
        minibatch_size = self._config['minibatch_size']
        prev_layer = None

        for layer_name, layer_conf in self._config['generator']:
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
#                    in_chs = layer_conf['input_dim']
                    in_chs = self._config['generator_input']['size']
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

    def _build_discriminator_network(self):
        minibatch_size = self._config['minibatch_size']
        prev_layer = None

        for layer_name, layer_conf in self._config['discriminator']:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    input_size = self._config['input_size']
                    in_chs = self._config['num_input_chs']
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
                    # XXX Just logits or plus sigmoid?
                    new_layer = tf.reshape(
                        pre_activation,
                        (minibatch_size, 1),
                        name='output_logits',
                    )

            # End of layer_name variable scope.

            prev_layer = new_layer

    def _build_train_ops(self):
        minibatch_size = self._config['minibatch_size']
        input_shape = (minibatch_size, 1)

        real_D_logits = tf.placeholder(
            dtype=tf.float32,
            shape=input_shape,
            name='real_D_logits',
        )

        fake_D_logits = tf.placeholder(
            dtype=tf.float32,
            shape=input_shape,
            name='fake_D_logits',
        )

        real_D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_D_logits),
                logits=real_D_logits,
            ),
            name='real_D_loss',
        )

        fake_D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_D_logits),
                logits=fake_D_logits,
            ),
            name='fake_D_loss',
        )

#        G_loss = -fake_D_loss 

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_D_logits),
                logits=(-fake_D_logits),
            ),
            name='G_loss',
        )

        # TODO: check hyperparameters.
        adam = tf.train.AdamOptimizer(**self._config['adam'])

        train_D_op = adam.minimize(
            loss=(real_D_loss + fake_D_loss),
            name='minimize_D_loss',
        )

        train_G_op = adam.minimize(
            loss=G_loss,
            name='minimize_G_loss',
        )

    def _load_data(self):
        dataset_name=self._config['dataset_name']

        if dataset_name == 'MNIST':
            ((x_train, y_train),
             (x_test, y_test)) = tf.contrib.keras.datasets.mnist.load_data()
            self._mnist_data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'x_size': len(x_train),
                'y_size': len(y_train),
            }
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

    def get_samples_from_data(self, minibatch_size=1):
        dataset_name=self._config['dataset_name']
        
        if dataset_name == 'MNIST':
            samples = self._mnist_data['x_train'][
                np.random.randint(
                    low=0,
                    high=self._mnist_data['x_size'],
                    size=minibatch_size,
                )
            ]
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

        return samples

    def generate_samples(self, minibatch_size=1):
        feed_dict = {
            self._tf_graph.get_tensor_by_name(
                'generator/conv0_input/Z:0',
            ): np.random.uniform(**self._config['generator_input'])
        }

        samples = self._tf_session.run(
            fetches=self._tf_graph.get_tensor_by_name(
                'generator/conv2_output/output:0',
            ),
            feed_dict=feed_dict,
        )

        return samples
        

def batch_normalization(
    input_tensor,
    momentum=0.99,
    epsilon=0.001,
    training=False,
    use_layers_api=False,
):
    x = input_tensor
    x_shape = x.shape.as_list()[1:]

    if use_layers_api:
        y = tf.layers.batch_normalization(
            x,
            momentum=momentum,
            epsilon=epsilon,
            training=training,
            name='layers_batch_normalization',
        )
    else:
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
                        [inf_mean_op, inf_variance_op]
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
