dcgan_mnist = {
    'generator': [
#        ('input', {'dim': 100}),
#        ('fc', {'out_size': 7, 'out_chs': 128}),
#        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
#        ('conv2_output', {'filter_size': 5, 'stride': 2, 'out_chs': 1}),
        ('conv0_input',
            {'input_dim': 100, 'filter_size': 7, 'stride': 1, 'out_chs': 128}
        ),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv2_output', {'filter_size': 5, 'stride': 2, 'out_chs': 1}),
    ],
    'discriminator': [
#        ('input', {'size': 28, 'num_chs': 1}),
        ('conv0_input',
            {'input_size', 28, 'in_chs': 1,
             'filter_size': 5, 'stride': 2, 'out_chs': 64}
        ),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 128}),
        ('conv2_output', {'filter_size': 7, 'stride': 1, 'out_chs': 1}),
    ],
}
