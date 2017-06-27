# TODO: Use TF queue & build a single train op
# to put everything on GPU and compare performances.

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
            if self.training:
                with tf.variable_scope('generator'):
                    self._build_generator_network(
                        training=self.training,
                    )
                with tf.variable_scope('discriminator'):
                    self._build_discriminator_network(
                        inputs='real',
                    )
                    self._build_discriminator_network(
                        inputs='fake',
                    )
                with tf.variable_scope('training'):
                    self._build_train_ops()

                with tf.variable_scope('summary'):
                    self._build_summary_ops()
            else:
                with tf.variable_scope('generator'):
                    self._build_generator_network()
                with tf.variable_scope('discriminator'):
                    self._build_discriminator_network()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            **self._config['variable_initializer']
        )

    def _build_generator_network(self, training=False):
        minibatch_size = self._config['minibatch_size']

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

            with tf.variable_scope(layer_name):
                
                if first_layer:
                    if training:
                        cfg_G_input = self._config['generator_input']
                        low = cfg_G_input['low']
                        high = cfg_G_input['high']
                        in_chs = cfg_G_input['size']
                        prev_layer = tf.random_uniform(
                            shape=(minibatch_size, in_chs),
                            minval=low,
                            maxval=high,
                            dtype=tf.float32,
                            name='Zs',
                        )
                    else:
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

    def _build_discriminator_network(self, inputs=None):
        minibatch_size = self._config['minibatch_size']

        cfg_D = self._config['discriminator']
        num_layers = len(cfg_D)

        if inputs == 'real':
            image_batch = tf.train.shuffle_batch(
                tensors=[self._data],
                batch_size=minibatch_size,
                capacity=len(self._data),
                min_after_dequeue=minibatch_size,
            )
            prev_layer = image_batch
            reuse = False
        elif inputs == 'fake':
            prev_layer = self._get_G_output_tensor()
            reuse = True
        else:
            cfg_D_input = cfg_D['input']
            input_size = cfg_D_input['size']
            in_chs = cfg_D_input['in_chs']
            prev_layer = tf.placeholder(
                dtype=tf.float32,
                shape=(minibatch_size, input_size, input_size, in_chs),
                name='input',
            )
            reuse = False

        for i, (layer_name, layer_conf) in enumerate(cfg_D):
            first_layer = False
            last_layer = False
            if i == 0:
                first_layer = True
            elif i == num_layers - 1:
                last_layer = True 

            with tf.variable_scope(layer_name, reuse=reuse):
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
#                elif last_layer:
#                    # XXX Just logits or plus sigmoid?
#                    new_layer = tf.reshape(
#                        pre_activation,
#                        shape=(minibatch_size, 1),
#                        #shape=(-1, 1),
#                        name='logits',
#                    )
                else:
                    raise RuntimeError

            # End of layer_name variable scope.

            prev_layer = new_layer

        # End of conf for loop.

        # XXX Just logits or plus sigmoid?
        if inputs is not None:
            output_logits_name = 'logits_{}'.format(inputs)
        else:
            output_logits_name = 'logits'
        new_layer = tf.reshape(
            pre_activation,
            shape=(minibatch_size, 1),
            name=output_logits_name,
        )

#    def _get_D_logits_tensor(self):
#        D_output_layer_name, _ = self._config['discriminator'][-1]
#        return self._tf_graph.get_tensor_by_name(
#            'discriminator/{}/logits:0'.format(D_output_layer_name),
#        )

    def _build_train_ops(self):
        # TODO: check hyperparameters.
        adam = tf.train.AdamOptimizer(**self._config['adam'])

        D_logits_real = self._tf_graph.get_tensor_by_name(
            'discriminator/logits_real:0'
        )

        D_logit_labels_real = tf.ones_like(D_logits_real)

        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=D_logit_labels_real,
                logits=D_logits_real,
            ),
            name='D_loss_real',
        )

        D_logits_fake = self._tf_graph.get_tensor_by_name(
            'discriminator/logits_fake:0'
        )

        D_logit_labels_fake = tf.ones_like(D_logits_fake)

        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=D_logit_labels_fake,
                logits=D_logits_fake,
            ),
            name='D_loss_fake',
        )

        D_loss = D_loss_real + D_loss_fake

        train_D_op = adam.minimize(
            loss=D_loss,
            name='minimize_D_loss',
            var_list=self._tf_graph.get_collection(
                name='trainable_variables',
                scope='discriminator',
            )
        )

        with tf.control_dependencies([train_D_op]):
            # TODO: Check if this refreshes D logits.
            D_logits = self._tf_graph.get_tensor_by_name(
                'discriminator/logits_fake:0'
            )

            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(D_logits),
                    logits=(-D_logits),
                ),
                name='G_loss',
            )

            train_G_op = adam.minimize(
                loss=G_loss,
                name='minimize_G_loss',
                var_list=self._tf_graph.get_collection(
                    name='trainable_variables',
                    scope='generator',
                )
            )

        train_op = tf.group(
            train_D_op,
            train_G_op,
            name='train_op',
        )

    def _build_summary_ops(self):
        with tf.variable_scope('discriminator'):
            tf.summary.scalar(
                name='real_loss',
                tensor=tf.placeholder(
                    dtype=tf.float32,
                    shape=(1),
                    name='t_real_loss',
                )
            )
            tf.summary.scalar(
                name='real_mean_logit',
                tensor=tf.placeholder(
                    dtype=tf.float32,
                    shape=(1),
                    name='t_real_mean_logit',
                )
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
            self._data = np.reshape(x_train, (-1, 28, 28, 1))
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
        
#    def train(self):
#        if not self.training:
#            raise RuntimeError
#
#        minibatch_size = self._config['minibatch_size']
#
#        for i in range(self._config['num_training_iterations']):
##            real_inputs = self.get_samples_from_data(minibatch_size)
##            fake_inputs = self.generate_samples(minibatch_size)
##            D_inputs = np.concatenate(
##                (real_inputs, fake_inputs),
##            )
##            logit_labels = np.concatenate(
##                (np.ones(minibatch_size), np.zeros(minibatch_size)),
##            )
##
##            fetches = [
##                self._tf_graph.get_tensor_by_name(
##                    'training/D_loss:0',
##                ),
##                self._tf_graph.get_operation_by_name(
##                    'train/minimize_D_loss',
##                ),
##            ]
##
##            D_input_tesnor = self._get_G_output_tensor() 
##            feed_dict = {
##                D_input_tensor: D_inputs,
##                self._tf_graph.get_tensor_by_name(
##                    'training/logit_labels:0',
##                ): logit_labels,
##            }
##
##            D_loss, _ = self._tf_session.run(
##                fetches=fetches,
##                feed_dict=feed_dict,
##            )
#
#            # Train D.
#            fetches = [
#                self._tf_graph.get_tensor_by_name(
#                    'training/D_loss:0'
#                ),
#                self._get_D_logits_tensor(),
#                self._tf_graph.get_operation_by_name(
#                    'training/minimize_D_loss'
#                ),
#            ]
#
#            # TODO: Study paper if optimizing
#            # D_loss_real and D_loss_fake separately
#            # is equivalent to optimizing their sum.
#
#            D_input_tensor = self._get_G_output_tensor() 
#            D_logit_labels_tensor = self._tf_graph.get_tensor_by_name(
#                'training/logit_labels:0'
#            )
#            D_logit_labels_shape = D_logit_labels_tensor.shape.as_list()
#
#            # Real samples.
#            feed_dict = {
#                D_input_tensor: self.get_samples_from_data(minibatch_size),
#                D_logit_labels_tensor: np.ones(D_logit_labels_shape),
#            }
#            D_loss_real, D_logits_real, _ = self._tf_session.run(
#                fetches=fetches,
#                feed_dict=feed_dict,
#            )
#            tf.assign(
#                self._tf_graph.get_tensor_by_name(
#                    'summary/discriminator/real_loss:0'
#                ),
#                D_loss_real,
#            )
#            tf.assign(
#                self._tf_graph.get_tensor_by_name(
#                    'summary/discriminator/real_mean_logit:0'
#                ),
#                np.mean(D_logits_real),
#            )
#
#            for inputs, logit_labels in (
#                (,
#                 ),
#                (self.generate_samples(),
#                 np.zeros(D_logit_labels_shape)),
#            ):
#                feed_dict = {
#                    D_input_tensor: inputs,
#                    D_logit_labels_tensor: logit_labels,
#                }
#
##            for inputs, logit_labels in (
##                (self.get_samples_from_data(minibatch_size),
##                 np.ones(D_logit_labels_shape)),
##                (self.generate_samples(),
##                 np.zeros(D_logit_labels_shape)),
##            ):
##                feed_dict = {
##                    D_input_tensor: inputs,
##                    D_logit_labels_tensor: logit_labels,
##                }
##
##                D_loss, _ = self._tf_session.run(
##                    fetches=fetches,
##                    feed_dict=feed_dict,
##                )
#            # Train G.
#            fetches = [
#                self._tf_graph.get_tensor_by_name(
#                    'training/G_loss:0'
#                ),
#                self._tf_graph.get_operation_by_name(
#                    'training/minimize_G_loss'
#                ),
#            ]
#
#            feed_dict = {
#                self._get_G_input_tensor(): self._sample_Zs(minibatch_size),
#            }
#
#            G_loss, _ = self._tf_session.run(
#                fetches=fetches,
#                feed_dict=feed_dict,
#            )

    def train(self):
        if not self.training:
            raise RuntimeError

        minibatch_size = self._config['minibatch_size']

        for i in range(self._config['num_training_iterations']):
            fetches = [
                self._tf_graph.get_operation_by_name(
                   'training/train_op' 
                )
            ]
            _ = self._tf_session.run(
                fetches=fetches,
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

