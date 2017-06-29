# TODO: Use TF queue & build a single train op
# to put everything on GPU and compare performances.

import os
import time
import json

import numpy as np
import tensorflow as tf

class DCGAN:
    def __init__(
        self,
        config=None,
        training=None,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        run_dir='runs',
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
        saver_var_list=None,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        self._config = config
        self._data = None

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

            self._tf_saver = tf.train.Saver(var_list=saver_var_list)

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
#                        # XXX
#                        prev_layer = tf.ones(
#                            shape=(minibatch_size, in_chs),
#                            name='Zs',
#                        )
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
            new_layer = self._get_data_batch_tensor() 
#            # XXX
#            checksum = tf.reduce_mean(
#                new_layer,
#                name='inputs_real_checksum',
#            )
            reuse = False
        elif inputs == 'fake':
            new_layer = self._get_G_output_tensor()
            reuse = True
        else:
            input_size = self._config['input_data']['size']
            in_chs = self._config['input_data']['num_chs']
            new_layer = tf.placeholder(
                dtype=tf.float32,
                shape=(minibatch_size, input_size, input_size, in_chs),
                name='inputs',
            )
            reuse = False

        for i, (layer_name, layer_conf) in enumerate(cfg_D):
            prev_layer = new_layer

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
#                else:
#                   raise RuntimeError

            # End of layer_name variable scope.


        # End of layer_conf for loop.

        if inputs is not None:
            logits_name = 'logits_{}'.format(inputs)
        else:
            logits_name = 'logits'
        new_layer = tf.reshape(
            pre_activation,
            shape=(minibatch_size, 1),
            name=logits_name,
        )

    def _get_D_logits_tensor(self, inputs=None):
#        D_output_layer_name, _ = self._config['discriminator'][-1]
        if inputs is not None:
            logits_name = 'logits_{}'.format(inputs)
        else:
            logits_name = 'logits'
        return self._tf_graph.get_tensor_by_name(
#            'discriminator/{}/{}:0'.format(
#                D_output_layer_name, logits_name,
            'discriminator/{}:0'.format(logits_name),
        )

    def _build_train_ops(self):
        # TODO: check hyperparameters.
        adam = tf.train.AdamOptimizer(**self._config['adam'])

        D_logits_real = self._get_D_logits_tensor(inputs='real')

        D_logit_labels_real = tf.ones_like(D_logits_real)

        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=D_logit_labels_real,
                logits=D_logits_real,
            ),
            name='D_loss_real',
        )

        D_logits_fake = self._get_D_logits_tensor(inputs='fake')

        D_logit_labels_fake = tf.zeros_like(D_logits_fake)

        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=D_logit_labels_fake,
                logits=D_logits_fake,
            ),
            name='D_loss_fake',
        )

        D_loss = tf.add(
            D_loss_real,
            D_loss_fake,
            name='D_loss',
        )

        train_D_op = adam.minimize(
            loss=D_loss,
            name='minimize_D_loss',
            var_list=self._tf_graph.get_collection(
                name='trainable_variables',
                scope='discriminator',
            )
        )

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(D_logits_fake),
                logits=(-D_logits_fake),
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

    def _build_summary_ops(self):
        D_summaries = []
        with tf.variable_scope('discriminator'):
            for input_type in ('real', 'fake'):
                with tf.variable_scope('inputs_{}'.format(input_type)):
                    if input_type == 'real':
                        image_tensor =  self._tf_graph.get_tensor_by_name(
                            'discriminator/inputs_real:0' 
                        )
                    else:
                        image_tensor = self._get_G_output_tensor() 
                    summary_image = tf.summary.image(
                        name='input_images',
                        tensor=image_tensor,
                    )
                    D_summaries.append(summary_image)

                    summary_mean_output = tf.summary.scalar(
                        name='mean_output',
                        tensor=tf.reduce_mean(
                            tf.sigmoid(
                                self._get_D_logits_tensor(
                                    inputs=input_type
                                )
                            )
                        )
                    )
                    D_summaries.append(summary_mean_output)

                    summary_loss = tf.summary.scalar(
                        name='loss',
                        tensor=self._tf_graph.get_tensor_by_name(
                            'training/D_loss_{}:0'.format(input_type)
                        )
                    )
                    D_summaries.append(summary_loss)

            summary_total_loss = tf.summary.scalar(
                name='loss',
                tensor=self._tf_graph.get_tensor_by_name(
                    'training/D_loss:0'
                )
            )
            D_summaries.append(summary_total_loss)

        tf.summary.merge(
            D_summaries,
            name='discriminator_summaries',
        )

        with tf.variable_scope('generator'):
            tf.summary.scalar(
                name='loss',
                tensor=self._tf_graph.get_tensor_by_name(
                    'training/G_loss:0'
                )
            )

    def _get_data_batch_tensor(self):
        dataset_name=self._config['dataset_name']
        minibatch_size = self._config['minibatch_size']
        data_batch_name = 'inputs_real'

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
#            data = np.concatenate((x_train, x_test))
            # XXX
            data = x_train[:200]
            data = np.array(
                (data / np.iinfo(np.uint8).max),
                dtype=np.float32,
            )
            self._data = np.reshape(data, (-1, 28, 28, 1))
            data_batch = tf.train.shuffle_batch(
                tensors=[self._data],
                batch_size=minibatch_size,
#                capacity=len(data),
                capacity=(100 * minibatch_size),
                min_after_dequeue=minibatch_size,
                #min_after_dequeue=0,
                enqueue_many=True,
                name=data_batch_name,
            )
#            for qr in self._tf_graph.get_collection('queue_runners'):
#                if qr.name == (
#                    'discriminator/{}/random_shuffle_queue'
#                    .format(data_batch_name)
#                ):
#                    break
#            self._data_queue = qr.queue 
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

        return data_batch

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
        
    def train(
        self,
        save_path=None,
        run_name=None,
    ):
        if not self.training:
            raise RuntimeError

        if save_path is not None:
            self._tf_saver.restore(self._tf_session, save_path)
            step = get_step_from_checkpoint(save_path)

        if run_name is None:
            run_name = (
                '{:02}{:02}_{:02}{:02}{:02}'.format(*time.localtime()[1:6])
            )

        summary_writer = tf.summary.FileWriter(
            logdir='logs/{}'.format(run_name),
            graph=self._tf_graph,
        )

        with open('runs/{}'.format(run_name), 'w') as fp:
            json.dump(self._config, fp)

        minibatch_size = self._config['minibatch_size']

        print('Starting data input queue...')
        coord = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(
            sess=self._tf_session,
            coord=coord,
        )
        epoch = 0
        step = 1
        max_num_steps = self._config['num_training_iterations']
#        max_num_epochs = self._config['num_training_epochs']
        num_steps_display = max_num_steps // 100
        num_steps_save = max_num_steps // 10

        try:
            while step <= max_num_steps and not coord.should_stop():
                # Train D.
                fetches = [
#                    self._tf_graph.get_tensor_by_name(
#                        'discriminator/inputs_real_checksum:0'
#                    ),
                    self._tf_graph.get_tensor_by_name(
                        'training/D_loss_real:0'
                    ),
                    self._tf_graph.get_tensor_by_name(
                        'training/D_loss_fake:0'
                    ),
                    self._tf_graph.get_tensor_by_name(
                        'summary/discriminator_summaries/'
                        'discriminator_summaries:0'
                    ),
                    self._tf_graph.get_operation_by_name(
                        'training/minimize_D_loss' 
                    ),
                ]
                (
#                    inputs_real_checksum,
                    D_loss_real,
                    D_loss_fake,
                    D_summaries,
                    _
                ) = self._tf_session.run(
                    fetches=fetches,
                )
                summary_writer.add_summary(D_summaries, step)

                # Train G.
                fetches = [
                    self._tf_graph.get_tensor_by_name(
                        'training/G_loss:0'
                    ),
                    self._tf_graph.get_tensor_by_name(
                        'summary/generator/loss:0'
                    ),
                    self._tf_graph.get_operation_by_name(
                        'training/minimize_G_loss' 
                    ),
                ]
                G_loss, G_summary, _ = self._tf_session.run(
                    fetches=fetches,
                )
                summary_writer.add_summary(G_summary, step)
                
                if step % num_steps_display == 0:
                    print(
                        'step {}: '
                        'D_loss_real = {:g}, '
                        'D_loss_fake = {:g}, '
                        'G_loss = {:g}.'
                        .format(step, D_loss_real, D_loss_fake, G_loss)
                    )

                if step % num_steps_save == 0:
                    save_path = self._tf_saver.save(
                        self._tf_session,
                        'checkpoints/{}'.format(run_name),
                        step,
                    )
                    print('checkpoint saved at {}'.format(save_path))

                step += 1

        except tf.errors.OutOfRangeError:
            raise RuntimeError

        finally:
            coord.request_stop()

        coord.join(queue_threads)

        summary_writer.close()


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

def get_step_from_checkpoint(save_path):
    return int(save_path.split('-')[-1])
