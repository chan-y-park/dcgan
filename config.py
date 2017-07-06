mnist = {
    'dataset_name': 'MNIST',
    'generator_input': {'low': -1.0, 'high': 1.0, 'size': 100},
    'generator': [
        ('fc', {'out_size': 7, 'out_chs': 128}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv2', {'filter_size': 5, 'stride': 2, 'out_chs': 1}),
    ],
    'discriminator': [
        ('conv0', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 128}),
        ('conv2', {'filter_size': 7, 'stride': 1, 'out_chs': 1}),
    ],
    'leaky_relu_alpha': 0.2,
    'minibatch_size': 128,
    'adam': {
        'learning_rate': 0.0002,
        'beta1': 0.5,
    },
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
    'input_data': {
        'size': 28,
        'num_chs': 1,
        'train_size': 60000,
        'test_size': 10000,
    },
    'training': {
        'max_num_steps': 10 ** 5,
        'num_steps_display': 10 ** 3,
        'num_steps_save': 10 ** 4,
    }
}

svhn = {
    'dataset_name': 'SVHN',
    'generator_input': {'low': -1.0, 'high': 1.0, 'size': 100},
    'generator': [
        ('fc', {'out_size': 4, 'out_chs': 512}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 256}),
        ('conv2', {'filter_size': 5, 'stride': 2, 'out_chs': 128}),
        ('conv3', {'filter_size': 5, 'stride': 2, 'out_chs': 3}),
    ],
    'discriminator': [
        ('conv0', {'filter_size': 5, 'stride': 2, 'out_chs': 128}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 256}),
        ('conv2', {'filter_size': 5, 'stride': 2, 'out_chs': 512}),
        ('conv3', {'filter_size': 4, 'stride': 1, 'out_chs': 1}),
    ],
    'leaky_relu_alpha': 0.2,
    'minibatch_size': 128,
    'adam': {
        'learning_rate': 0.0002,
        'beta1': 0.5,
    },
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
    'input_data': {
        'size': 32,
        'num_chs': 3,
        'train_size': 73257,
        'test_size': 26032,
    },
    'training': {
        'max_num_steps': 10 ** 6,
        'num_steps_display': 10 ** 3,
        'num_steps_save': 10 ** 4,
    }
}
