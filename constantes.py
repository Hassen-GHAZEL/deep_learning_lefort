tab_batch_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
tab_learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 1, 2]
tab_hidden_size = [
    [128, 64],          # 2 couches cachées, avec peu de neurones
    [256, 128],         # 2 couches cachées (par défaut)
    [512, 256],         # 2 couches avec plus de neurones
    [1024, 512],        # 2 couches plus larges
    [256, 128, 64],     # 3 couches plus profondes
    [512, 256, 128, 64] # 4 couches encore plus profondes
]
tab_weight_init_range = [(-0.0001, 0.0001), (-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1), (-2, 2), (-5, 5), (-10, 10)]


# tab_batch_size = [2, 3, 4, 5] # 4
# tab_learning_rate = [0.001, 0.0025, 0.005, 0.0075, 0.01] # 5
# tab_hidden_size = [128, 256, 512, 1024, 2048, 4096] # 6
# tab_weight_init_range = [(-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1)] # 4


"""
tab_batch_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
tab_learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 1, 2]
tab_hidden_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
tab_weight_init_range = [(-0.0001, 0.0001), (-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1), (-2, 2), (-5, 5), (-10, 10)]
"""

