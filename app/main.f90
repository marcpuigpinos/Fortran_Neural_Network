program main

    use iso_fortran_env, only: int32, real64
    use FortranNeuralNetwork

    implicit none

    integer error
    procedure(fnn_activation_function), pointer :: activation
    procedure(fnn_activation_function), pointer :: derivative_activation
    real(kind=8), pointer :: inputs(:), prediction(:)
    integer prediction_size

    activation => fnn_sigmoid
    derivative_activation => fnn_derivative_sigmoid

    ! error = fnn_net(number_inputs, number_layers)
    ! error = fnn_add(number_neurons, activation, derivative_activation)
    ! error = fnn_predict(prediction_size, prediction, n_inputs, inputs)
    ! call fnn_print()

    error = fnn_net(3, 1)
    allocate (inputs(3))
    inputs(1) = 5.0; inputs(2) = 2.1; inputs(3) = 6.7
    error = fnn_add(1, activation, derivative_activation)
    nullify (prediction)
    error = fnn_predict(prediction_size, prediction, 3, inputs)
    call fnn_print()

    error = fnn_net(4, 2)
    deallocate (inputs)
    nullify (inputs)
    allocate (inputs(4))
    inputs(1) = 5.0; inputs(2) = 2.1; inputs(3) = 6.7; inputs(4) = -5.0
    error = fnn_add(3, activation, derivative_activation)
    error = fnn_add(2, activation, derivative_activation)
    deallocate (prediction)
    nullify (prediction)
    error = fnn_predict(prediction_size, prediction, 4, inputs)
    call fnn_print()

end program main
