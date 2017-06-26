dcgan_mnist = {
    'dataset_name': 'MNIST',
    'generator_input': {'low': -1.0, 'high': 1.0, 'size': 100},
    'generator': [
#        ('conv0', {'filter_size': 7, 'stride': 1, 'out_chs': 128}),
        ('fc', {'out_size': 7, 'out_chs': 128}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv2', {'filter_size': 5, 'stride': 2, 'out_chs': 1}),
    ],
    'discriminator_input': {'low': 0.0, 'high': 1.0, 'size': 28, 'in_chs': 1},
    'discriminator': [
        ('conv0', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 128}),
        ('conv2', {'filter_size': 7, 'stride': 1, 'out_chs': 1}),
    ],
    'leaky_relu_alpha': 0.2,
    'input_size': 28,
    'num_input_chs': 1,
    'minibatch_size': 128,
    'adam': {
        'learning_rate': 0.0002,
        'beta1': 0.5,
    },
    'num_training_iterations': 100,
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
}
