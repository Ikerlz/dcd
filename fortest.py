import numpy as np
import time

laplace_matrix = np.random.rand(20000, 20000)
start = time.time()
np.linalg.svd(laplace_matrix)
end = time.time()

print(end - start)
