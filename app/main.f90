program main

   use iso_fortran_env, only: int32, real64
   use FortranNeuralNetwork

   implicit none

   type(fnn_network), pointer :: network
   integer(kind=int32) error, number_inputs, number_layers
   real(kind=real64), pointer :: inputs(:), predictions(:)
   procedure(fnn_activation_function), pointer :: activation
   procedure(fnn_derivative_activation_function), pointer :: derivative_activation

   activation => fnn_ReLU
   derivative_activation => fnn_derivative_ReLU
   number_inputs = 3
   number_layers = 3
   nullify (inputs)
   allocate (inputs(number_inputs))
   inputs(1) = 1.0
   inputs(2) = 1.0
   inputs(3) = 1.0

   ! Allocate network
   error = allocate_network(network)

   ! Initialize
   error = initialize_network(network, number_inputs, number_layers)

   ! Add first layer
   activation => fnn_ReLU
   derivative_activation => fnn_derivative_ReLU
   error = add_layer_to_network(network, 1, 4, activation, derivative_activation)
   
   ! Add second layer
   activation => fnn_ReLU
   derivative_activation => fnn_derivative_ReLU
   error = add_layer_to_network(network, 2, 2, activation, derivative_activation)
   
   ! Add third layer
   activation => fnn_sigmoid
   derivative_activation => fnn_derivative_sigmoid
   error = add_layer_to_network(network, 3, 1, activation, derivative_activation)

   ! activate network
   nullify(predictions)
   allocate(predictions(1))
   error = activate_network(network, predictions, number_inputs, inputs)
   
   ! Print network
   call print_network(network)

   ! Print predictions
   print *, "Predictions: ", predictions
   
   ! Deallocate network
   error = deallocate_network(network)

   deallocate (inputs)
   deallocate (predictions)
   nullify (inputs, predictions)

end program main
