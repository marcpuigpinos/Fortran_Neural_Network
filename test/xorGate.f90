program xorGate

      use iso_fortran_env, only: int32, real64
      use FortranNeuralNetwork
  
      implicit none
  
      integer error
      procedure(fnn_activation_function), pointer :: activation
      procedure(fnn_derivative_activation_function), pointer :: derivative_activation
      procedure(fnn_cost_function), pointer :: cost_function
      integer number_inputs, number_layers, number_outputs, number_samples, epochs, i
      real(kind=real64), pointer :: samples_input(:, :), samples_output(:, :), prediction(:), inputs(:)
      real(kind=real64) learning_rate, epsilon
  
      ! Nullify local pointers
      nullify (samples_input, samples_output, prediction, inputs)
  
      ! Define activation functions
      activation => fnn_sigmoid
      derivative_activation => fnn_derivative_sigmoid
  
      ! Define network
      number_inputs = 3
      number_layers = 3
      number_outputs = 1
      epochs = 100000
      learning_rate = 0.1
      epsilon = 0.1
      error = fnn_net(number_inputs, number_layers)
      error = fnn_add(4, activation, derivative_activation)
      error = fnn_add(2, activation, derivative_activation)
      error = fnn_add(number_outputs, activation, derivative_activation)
  
      call fnn_print()
  
      ! Create the samples input and samples output arrays
      number_samples = 4
      allocate (samples_input(number_inputs, number_samples))
      allocate (samples_output(number_outputs, number_samples))
      samples_input(1, 1) = 1.0; samples_input(2, 1) = 0.0; samples_input(3, 1) = 0.0; samples_output(1, 1) = 1.0
      samples_input(1, 2) = 1.0; samples_input(2, 2) = 1.0; samples_input(3, 2) = 0.0; samples_output(1, 2) = 0.0
      samples_input(1, 3) = 1.0; samples_input(2, 3) = 0.0; samples_input(3, 3) = 1.0; samples_output(1, 3) = 0.0
      samples_input(1, 4) = 1.0; samples_input(2, 4) = 1.0; samples_input(3, 4) = 1.0; samples_output(1, 4) = 1.0
  
      ! Train
      cost_function => fnn_cost_cross_entropy
      error = fnn_train(number_inputs, number_outputs, number_samples, epochs, &
                        samples_input, samples_output, learning_rate, epsilon, cost_function)
      if (error /= 0) then
          print *, "Training failed"
      end if
  
      ! Print the network
      call fnn_print()
  
      ! Prediction
      allocate (prediction(number_outputs))
      allocate (inputs(number_inputs))
      inputs(1) = 1.0; inputs(2) = 0.0; inputs(3) = 0.0; 
      error = fnn_predict(number_outputs, prediction, number_inputs, inputs)
      do i = 1, number_outputs
          if (prediction(i) >= 0.5d0 ) then
              prediction(i) = 1d0
          else
              prediction(i) = 0d0
          endif
      enddo
  
      ! Print result
      write (*, *) "Inputs:"
      write (*, *) inputs
      write (*, *) "Prediction:"
      write (*, *) prediction
  
      ! Prediction
      inputs(1) = 1.0; inputs(2) = 1.0; inputs(3) = 0.0; 
      error = fnn_predict(number_outputs, prediction, number_inputs, inputs)
      do i = 1, number_outputs
          if (prediction(i) >= 0.5d0 ) then
              prediction(i) = 1d0
          else
              prediction(i) = 0d0
          endif
      enddo
  
      ! Print result
      write (*, *) "Inputs:"
      write (*, *) inputs
      write (*, *) "Prediction:"
      write (*, *) prediction
  
      ! Prediction
      inputs(1) = 1.0; inputs(2) = 0.0; inputs(3) = 1.0; 
      error = fnn_predict(number_outputs, prediction, number_inputs, inputs)
      do i = 1, number_outputs
          if (prediction(i) >= 0.5d0 ) then
              prediction(i) = 1d0
          else
              prediction(i) = 0d0
          endif
      enddo
  
      ! Print result
      write (*, *) "Inputs:"
      write (*, *) inputs
      write (*, *) "Prediction:"
      write (*, *) prediction
  
      ! Prediction
      inputs(1) = 1.0; inputs(2) = 1.0; inputs(3) = 1.0; 
      error = fnn_predict(number_outputs, prediction, number_inputs, inputs)
      do i = 1, number_outputs
          if (prediction(i) >= 0.5d0 ) then
              prediction(i) = 1d0
          else
              prediction(i) = 0d0
          endif
      enddo
  
      ! Print result
      write (*, *) "Inputs:"
      write (*, *) inputs
      write (*, *) "Prediction:"
      write (*, *) prediction
  
      ! Free memory and Nullify local pointers
      if (associated(samples_input)) deallocate (samples_input)
      if (associated(samples_output)) deallocate (samples_output)
      if (associated(prediction)) deallocate (prediction)
      if (associated(inputs)) deallocate (inputs)
      nullify (samples_input, samples_output, prediction, inputs)
  
end program xorGate
