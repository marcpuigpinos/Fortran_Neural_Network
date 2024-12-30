program main

  use iso_fortran_env, only: int32, real64
  use FortranNeuralNetwork
  
  implicit none

  type(fnn_neuron), pointer :: neuron
  real(kind=real64), pointer :: inputs(:), samples(:,:), dcost(:)
  integer(kind=int32) error, number_inputs, number_samples, i, epoch, number_of_epochs
  real(kind=real64) prediction, cost, pcost
  ! Declare a procedure pointer for the activation function
  procedure(fnn_activation_function), pointer :: activation
  ! Declare a procedure pointer for the derivative of the activation function
  procedure(fnn_derivative_activation_function), pointer :: dactivation
  
  !!!!!---------  f(x) = x --------
  !!!!number_inputs = 1
!!!!
  !!!!! Sample for prediction
  !!!!nullify(samples)
  !!!!number_samples = 2
  !!!!allocate(samples(number_inputs + 1, number_samples))
  !!!!samples(1,1) = 0d0; samples(2,1) = 5d0
  !!!!samples(1,2) = 4d0; samples(2,2) = 9d0
  
  !----- f(x, y) = x + y + K ----
  ! Input values to predict
  number_inputs = 2

  ! Sample for prediction
  nullify(samples)
  number_samples = 4
  allocate(samples(number_inputs + 1, number_samples))
  samples(1,1) = 0; samples(2,1) = 0; samples(3,1) = 0
  samples(1,2) = 3; samples(2,2) = 3; samples(3,2) = 6
  samples(1,3) = 6; samples(2,3) = 6; samples(3,3) = 12
  samples(1,4) = 5; samples(2,4) = 0; samples(3,4) = 5
  
  ! Allocate the neuron
  error = allocate_neuron(neuron)

  ! Point activation function
  !activation => fnn_sigmoid  ! Point to the sigmoid function
  activation => fnn_ReLU ! Pint to ReLU function

  ! Point derivative activation function
  !dactivation => fnn_derivative_sigmoid
  dactivation => fnn_derivative_ReLU

  ! Initialize the neuron
  error = initialize_neuron(neuron, number_inputs, activation, dactivation)
  call print_neuron(neuron, 0)

  ! Train the neuron
  ! allocate dcost
  nullify(dcost)
  allocate(dcost(number_inputs + 1))
  dcost = 0d0
  pcost = huge(0d0)
  ! Loop over epocs
  number_of_epochs = 10000000
  do epoch = 1, number_of_epochs
  
     ! Compute the cost
     error = cost_function_neuron(neuron, cost, number_inputs, number_samples, samples)

     ! check exit
     if ( cost < 1e-6 ) exit

     ! Compute derivative of the cost

     error = derivative_cost_function_neuron(neuron, dcost, number_inputs, number_samples, samples)

     error = update_neuron(neuron, dcost, number_inputs, 0.01d0)
     
  enddo
  
  ! Write cost
  write(*,*) "Cost: "
  write(*,*) epoch, cost

  ! Write neuron
  call print_neuron(neuron, 0)

  ! Make predictions
  nullify(inputs)
  allocate(inputs(number_inputs))
  inputs(1) = 5d0
  inputs(2) = 0d0
  error = prediction_neuron(neuron, prediction, number_inputs, inputs)
  write(*,*) "Prediction = ", inputs(1), inputs(2), prediction
  !write(*,*) "Prediction = ", inputs(1), prediction

  inputs(1) = 2d0
  inputs(2) = 2d0
  error = prediction_neuron(neuron, prediction, number_inputs, inputs)
  write(*,*) "Prediction = ", inputs(1), inputs(2), prediction
  !write(*,*) "Prediction = ", inputs(1), prediction

  inputs(1) = 3d0
  inputs(2) = 1d0
  error = prediction_neuron(neuron, prediction, number_inputs, inputs)
  write(*,*) "Prediction = ", inputs(1), inputs(2), prediction

  inputs(1) = 14d0
  inputs(2) = 10d0
  error = prediction_neuron(neuron, prediction, number_inputs, inputs)
  write(*,*) "Prediction = ", inputs(1), inputs(2), prediction

  ! Deallocate neuron
  error = deallocate_neuron(neuron)

  if ( associated(inputs) ) deallocate(inputs)
  nullify(inputs)

  if ( associated(samples) ) deallocate(samples)
  nullify(samples)

  if ( associated(dcost) ) deallocate(dcost)
  nullify(dcost)
  
end program main
