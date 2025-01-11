program main

    use iso_fortran_env, only: int32, real64
    use FortranNeuralNetwork

    implicit none

    type(fnn_layer), pointer :: layer
    integer(kind=int32) error, number_inputs, number_neurons
    real(kind=real64), pointer :: inputs(:), predictions(:)
    procedure(fnn_activation_function), pointer :: activation
    procedure(fnn_derivative_activation_function), pointer :: derivative_activation        

    activation => fnn_ReLU
    derivative_activation => fnn_derivative_ReLU
    number_inputs = 3
    nullify(inputs)
    allocate(inputs(number_inputs))
    inputs(1) = 2.0
    inputs(2) = 1.0
    inputs(3) = 45.0
    number_neurons = 2
    nullify(predictions)
    allocate(predictions(number_neurons))
    nullify(layer)
    error = allocate_layer(layer)
    print *, "Error allocate: ", error
    error = initialize_layer(layer, number_inputs, number_neurons, activation, derivative_activation)
    print *, "Error initialize: ", error
    call print_layer(layer, 4)
    error = prediction_layer(layer, predictions, number_inputs, inputs)
    print *, "Error prediction: ", error
    print *, "Inputs: ", inputs
    print *, "Predictions: ", predictions

    deallocate(inputs)
    deallocate(predictions)
    nullify(inputs,predictions)

end program main
