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
                self._G_var_list = None
                self._build_generator_network()
            with tf.variable_scope('discriminator'):
                self._D_var_list = None
                self._build_discriminator_network()

            if self.training:
                with tf.variable_scope('training'):
                    self._build_train_ops()

                with tf.variable_scope('summary'):
                    self._build_summary_ops()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            **self._config['variable_initializer']
        )

    def _build_generator_network(self):
        minibatch_size = self._config['minibatch_size']
#        minibatch_size = None

        cfg_G = self._config['generator']
        num_layers = len(cfg_G)

        prev_layer = None

        for i, (layer_name, layer_conf) in enumerate(cfg_G):
            first_layer = False
            last_layer = False
            if i == 0:
                first_layer = True
            elif i == num_layers - 1:
                last_layer = True 

#            with tf.variable_scope(layer_name):
#                filter_size = layer_conf['filter_size']
#                stride = layer_conf['stride']
#                out_chs = layer_conf['out_chs']
#
#                if first_layer:
#                    in_chs = self._config['generator_input']['size']
#                    prev_layer = tf.placeholder(
#                        dtype=tf.float32,
#                        shape=(minibatch_size, 1, 1, in_chs),
#                        name='Zs',
#                    )
#                    out_size = filter_size
#                else:
#                    _, _, in_size, in_chs = prev_layer.shape.as_list()
#                    out_size =  in_size * stride
#
#                W = tf.get_variable(
#                    name='W',
#                    shape=(filter_size,
#                           filter_size,
#                           out_chs,
#                           in_chs),
#                    initializer=self._get_variable_initializer(),
#                )
#
#                pre_activation = tf.nn.conv2d_transpose(
#                    prev_layer,
#                    W,
#                    output_shape=(minibatch_size,
#                                  out_size,
#                                  out_size,
#                                  out_chs),
#                    strides=(1, stride, stride, 1),
#                    padding='SAME',
#                    name='pre_activation',
#                )
#
#                if not last_layer:
#                    new_layer = tf.nn.relu(
#                        batch_normalization(
#                            input_tensor=pre_activation,
#                            training=self.training,
#                        ),
#                        name='activation',
#                    )
#                elif last_layer:
#                    new_layer = tf.sigmoid(pre_activation, name='output')
#                else:
#                    raise RuntimeError

            with tf.variable_scope(layer_name):
                
                if first_layer:
                    in_chs = self._config['generator_input']['size']
                    prev_layer = tf.placeholder(
                        dtype=tf.float32,
                        shape=(minibatch_size, in_chs),
                        name='Zs',
                    )
                    out_size = layer_conf['out_size']
                    out_chs = layer_conf['out_chs']

                    W = tf.get_variable(
                        name='W',
                        shape=(in_chs, out_size * out_size * out_chs),
                        initializer=self._get_variable_initializer(),
                    )

                    pre_activation = tf.reshape(
                        tf.matmul(prev_layer, W),
                        shape=(minibatch_size, out_size, out_size, out_chs),
                        name='pre_activation',
                    )
                else:
                    filter_size = layer_conf['filter_size']
                    stride = layer_conf['stride']
                    out_chs = layer_conf['out_chs']

                    _, _, in_size, in_chs = prev_layer.shape.as_list()
                    out_size =  in_size * stride

                    W = tf.get_variable(
                        name='W',
                        shape=(filter_size,
                               filter_size,
                               out_chs,
                               in_chs),
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

                if not last_layer:
                    new_layer = tf.nn.relu(
                        batch_normalization(
                            input_tensor=pre_activation,
                            training=self.training,
                        ),
                        name='activation',
                    )
                elif last_layer:
                    new_layer = tf.sigmoid(pre_activation, name='output')
                else:
                    raise RuntimeError

            # End of layer_name variable scope.

            prev_layer = new_layer

        # End of conf for loop.

    def _get_G_input_tensor(self):
        G_input_layer_name, _ = self._config['generator'][0]
        return self._tf_graph.get_tensor_by_name(
            'generator/{}/Zs:0'.format(G_input_layer_name),
        )

    def _get_G_output_tensor(self):
        G_output_layer_name, _ = self._config['generator'][-1]
        return self._tf_graph.get_tensor_by_name(
            'generator/{}/output:0'.format(G_output_layer_name),
        )

    def _build_discriminator_network(self):
        minibatch_size = self._config['minibatch_size']
#        minibatch_size = None

        cfg_D = self._config['discriminator']
        num_layers = len(cfg_D)

        prev_layer = self._get_G_output_tensor()

        for i, (layer_name, layer_conf) in enumerate(cfg_D):
            first_layer = False
            last_layer = False
            if i == 0:
                first_layer = True
            elif i == num_layers - 1:
                last_layer = True 

            with tf.variable_scope(layer_name):
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

                if not last_layer:
                    padding = 'SAME'
                else:
                    padding = 'VALID'

                pre_activation = tf.nn.conv2d(
                    prev_layer,
                    W,
                    strides=(1, stride, stride, 1),
                    padding=padding,
                    name='pre_activation',
                )

                if first_layer:
                    new_layer = leaky_relu(pre_activation)
                elif not last_layer:
                    new_layer = leaky_relu(
                        batch_normalization(
                            input_tensor=pre_activation,
                            training=self.training,
                        ),
                    )
                elif last_layer:
                    # XXX Just logits or plus sigmoid?
                    new_layer = tf.reshape(
                        pre_activation,
#                        (minibatch_size, 1),
                        shape=(-1, 1),
                        name='logits',
                    )
                else:
                    raise RuntimeError

            # End of layer_name variable scope.

            prev_layer = new_layer

    def _get_D_logits_tensor(self):
        D_output_layer_name, _ = self._config['discriminator'][-1]
        return self._tf_graph.get_tensor_by_name(
            'discriminator/{}/logits:0'.format(D_output_layer_name),
        )

    def _build_train_ops(self):
        D_logits = self._get_D_logits_tensor()

        logit_labels = tf.placeholder(
            dtype=tf.float32,
            shape=D_logits.shape,
            name='logit_labels',
        )

        D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=logit_labels,
                logits=D_logits,
            ),
            name='D_loss',
        )

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(D_logits),
                logits=(-D_logits),
            ),
            name='G_loss',
        )

        # TODO: check hyperparameters.
        adam = tf.train.AdamOptimizer(**self._config['adam'])

        train_D_op = adam.minimize(
            loss=D_loss,
            name='minimize_D_loss',
            var_list=self._tf_graph.get_collection(
                name='trainable_variables',
                scope='discriminator',
            )
        )

        train_G_op = adam.minimize(
            loss=G_loss,
            name='minimize_G_loss',
            var_list=self._tf_graph.get_collection(
                name='trainable_variables',
                scope='generator',
            )
        )

    def _build_summary_ops(self):
        pass

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
            samples = samples[:,:,:,np.newaxis]
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

        return samples

    def _sample_Zs(self, minibatch_size=1):
        cfg = self._config['generator_input']
        Zs = np.reshape(
            np.random.uniform(
                low=cfg['low'],
                high=cfg['high'],
                size=(minibatch_size * cfg['size']),
            ),
            (minibatch_size, cfg['size']),
        )
        return Zs

    def generate_samples(self):
        minibatch_size = self._config['minibatch_size']

        feed_dict = {
            self._get_G_input_tensor(): self._sample_Zs(minibatch_size)
        }

        samples = self._tf_session.run(
            fetches=self._get_G_output_tensor(),
            feed_dict=feed_dict,
        )

        return samples
        
    def train(self):
        if not self.training:
            raise RuntimeError

        minibatch_size = self._config['minibatch_size']

        for i in range(self._config['num_training_iterations']):
#            real_inputs = self.get_samples_from_data(minibatch_size)
#            fake_inputs = self.generate_samples(minibatch_size)
#            D_inputs = np.concatenate(
#                (real_inputs, fake_inputs),
#            )
#            logit_labels = np.concatenate(
#                (np.ones(minibatch_size), np.zeros(minibatch_size)),
#            )
#
#            fetches = [
#                self._tf_graph.get_tensor_by_name(
#                    'training/D_loss:0',
#                ),
#                self._tf_graph.get_operation_by_name(
#                    'train/minimize_D_loss',
#                ),
#            ]
#
#            D_input_tesnor = self._get_G_output_tensor() 
#            feed_dict = {
#                D_input_tensor: D_inputs,
#                self._tf_graph.get_tensor_by_name(
#                    'training/logit_labels:0',
#                ): logit_labels,
#            }
#
#            D_loss, _ = self._tf_session.run(
#                fetches=fetches,
#                feed_dict=feed_dict,
#            )

            # Train D.
            fetches = [
                self._tf_graph.get_tensor_by_name(
                    'training/D_loss:0'
                ),
                self._tf_graph.get_operation_by_name(
                    'training/minimize_D_loss'
                ),
            ]

            D_input_tensor = self._get_G_output_tensor() 
            D_logit_labels_tensor = self._tf_graph.get_tensor_by_name(
                'training/logit_labels:0'
            )
            D_logit_labels_shape = D_logit_labels_tensor.shape.as_list()

            for inputs, logit_labels in (
                (self.get_samples_from_data(minibatch_size),
                 np.ones(D_logit_labels_shape)),
                (self.generate_samples(minibatch_size),
                 np.zeros(D_logit_labels_shape)),
            ):
                feed_dict = {
                    D_input_tensor: inputs,
                    D_logit_labels_tensor: logit_labels,
                }

                D_loss, _ = self._tf_session.run(
                    fetches=fetches,
                    feed_dict=feed_dict,
                )

            # Train G.
            fetches = [
                self._tf_graph.get_tensor_by_name(
                    'training/G_loss:0'
                ),
                self._tf_graph.get_operation_by_name(
                    'training/minimize_G_loss'
                ),
            ]

            feed_dict = {
                self._get_G_input_tensor(): self._sample_Zs(minibatch_size),
            }

            G_loss, _ = self._tf_session.run(
                fetches=fetches,
                feed_dict=feed_dict,
            )


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

