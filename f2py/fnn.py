from fai import fnn

def main():
    print("Hello World!")
    error = fnn.net_init(5)
    error = fnn.net_add_layer(10, "sigmoid")
    error = fnn.net_add_layer(5, "ReLU")
    error = fnn.net_compile("MSE", 10, 1, "GD", 0.1)

if __name__ == "__main__":
    main()
