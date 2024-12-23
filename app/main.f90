program main

  use iso_fortran_env, only: int32, real64
  use FortranNeuralNetwork
  
  implicit none

  type(fnn_neuron), pointer :: neuron
  real(kind=real64), pointer :: inputs(:), samples(:,:)
  integer(kind=int32) error, number_inputs, number_samples
  real(kind=real64) prediction, cost
  ! Declare a procedure pointer for the activation function
  procedure(fnn_activation_function), pointer :: activation

  !----- f(x, y) = x^2 + y^2 + K ----
  ! Input values to predict
  number_inputs = 2
  nullify(inputs)
  allocate(inputs(number_inputs))
  inputs(1) = 5d0
  inputs(2) = 5d0
  ! result must be (for K = 0) 50

  ! Sample for prediction
  nullify(samples)
  number_samples = 4
  allocate(samples(number_inputs + 1, number_samples))
  samples(1,1) = 0; samples(2,1) = 0; samples(3,1) = 0
  samples(1,2) = 3; samples(2,2) = 3; samples(3,2) = 18
  samples(1,3) = 6; samples(2,3) = 6; samples(3,3) = 72
  samples(1,4) = 5; samples(2,4) = 0; samples(3,4) = 25
  

  ! Allocate the neuron
  error = allocate_neuron(neuron)

  ! Point activation function
  !activation => fnn_sigmoid  ! Point to the sigmoid function
  activation => fnn_ReLU ! Pint to ReLU function

  ! Initialize the neuron
  error = initialize_neuron(neuron, number_inputs, activation)
  call print_neuron(neuron, 0)

  ! Train the neuron
  
  ! Compute the cost
  error = cost_function_neuron(neuron, cost, number_inputs, number_samples, samples)
  write(*,*) "Cost = ", cost
  
  ! Make prediction
  error = prediction_neuron(neuron, prediction, number_inputs, inputs)
  write(*,*) "Prediction = ", prediction
  error = deallocate_neuron(neuron)

  if ( associated(inputs) ) deallocate(inputs)
  nullify(inputs)

  if ( associated(samples) ) deallocate(samples)
  nullify(samples)
  
end program main
