dcgan_mnist = {
    'generator': [
        ('input', {'dim': 100}),
        ('fc', {'out_size': 7, 'out_chs': 128}),
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv2_output', {'filter_size': 5, 'stride': 2, 'out_chs': 1}),
    ],
    'discriminator': [
        ('conv1', {'filter_size': 5, 'stride': 2, 'out_chs': 64}),
        ('conv2', {'filter_size': 5, 'stride': 2, 'out_chs': 128}),
        ('fc', {}),
    ],
}
