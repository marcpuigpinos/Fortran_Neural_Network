import fnn

def main():

    print("Defining the network")
    # Define the network
    number_inputs = 3
    number_layers = 2
    number_outputs = 1
    epochs = 1000
    learning_rate = 0.1
    epsilon = 0.1
    print("Creating the network")
    error = fortran.fnn.network(number_inputs, number_layers)
    print("Adding layer")
    error = fortran.fnn.add_layer(2, activation="sigmoid")
    print("Adding layer")
    error = fortran.fnn.add_layer(number_outputs, activation="sigmoid")
    print("Print the network")
    fortran.fnn.print_network()

    #! Create the samples input and samples output arrays
    #number_samples = 4
    #allocate (samples_input(number_inputs, number_samples))
    #allocate (samples_output(number_outputs, number_samples))
    #samples_input(1, 1) = 1.0; samples_input(2, 1) = 0.0; samples_input(3, 1) = 0.0; samples_output(1, 1) = 1.0
    #samples_input(1, 2) = 1.0; samples_input(2, 2) = 1.0; samples_input(3, 2) = 0.0; samples_output(1, 2) = 0.0
    #samples_input(1, 3) = 1.0; samples_input(2, 3) = 0.0; samples_input(3, 3) = 1.0; samples_output(1, 3) = 0.0
    #samples_input(1, 4) = 1.0; samples_input(2, 4) = 1.0; samples_input(3, 4) = 1.0; samples_output(1, 4) = 1.0


if __name__ == "__main__":
    main()
