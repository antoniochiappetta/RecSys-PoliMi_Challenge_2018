from numpy import *
import time

threshold = 0.7

big_matrix = random.random((10000, 10000))

big_matrix_copy = big_matrix.copy()

start_time = time.time()

big_matrix_copy[big_matrix_copy < threshold] = 0.0

# NEVER use loops for computational tasks that can be easily vectorized

print("Vectorization takes {:.2f} seconds".format(time.time() - start_time))
