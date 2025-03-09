from fai import fnn
import numpy as np

def main():
    print("Hello World!")

    # m = 10
    # n = 5
    # matrix = np.zeros((m,n), dtype=np.float64, order="F") * np.random.rand() * 25
    # matrix.flags["F_CONTIGUOUS"]
    # vector = np.ones(n, dtype=np.float64, order="F") * np.random.rand() * 5
    # vector.flags["F_CONTIGUOUS"]
    # res = np.zeros(m, dtype=np.float64, order="F")
    # error = fnn.matrix_vector_product(matrix, vector, res)
    # print(res)
    # print(np.dot(matrix,vector))
    
    error = fnn.net_init(5, 1, 1)
    error = fnn.net_add_layer(10, "sigmoid")
    #error = fnn.net_compile("MSE", 10, 1, "GD", 0.1)
    error = fnn.net_print()

if __name__ == "__main__":
    main()
