tab_batch_size = [2, 4, 8, 16, 32, 64, 128]
tab_learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 1, 2]


liste_value = [256, 512]

tab_hidden_size = [(a, b, c, d) for a in liste_value for b in liste_value for c in liste_value for d in liste_value]

liste_value = [512, 1024]

tab_hidden_size.extend([(a, b, c, d, e) for a in liste_value for b in liste_value for c in liste_value for d in liste_value for e in liste_value])

# tab_hidden_size = [(x, y, z) for x in values_for_triplets for y in values_for_triplets for z in values_for_triplets]


tab_weight_init_range = [(-0.0001, 0.0001), (-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1), (-2, 2), (-5, 5), (-10, 10)]


# tab_batch_size = [2, 3, 4, 5] # 4
# tab_learning_rate = [0.001, 0.0025, 0.005, 0.0075, 0.01] # 5
# tab_hidden_size = [128, 256, 512, 1024, 2048, 4096] # 6
# tab_weight_init_range = [(-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1)] # 4


"""
tab_batch_size = [2, 4, 8, 16, 32, 64, 128]
tab_learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 1, 2]
tab_hidden_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
tab_weight_init_range = [(-0.0001, 0.0001), (-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1), (-2, 2), (-5, 5), (-10, 10)]
"""

