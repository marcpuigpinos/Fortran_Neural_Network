program main

  use iso_fortran_env, only: int32, real64
  use FortranNeuralNetwork
  
  implicit none

  type(fnn_neuron), pointer :: neuron
  real(kind=real64), pointer :: inputs(:), samples(:,:), dcost(:)
  integer(kind=int32) error, number_inputs, number_samples, i, epoch, number_of_epochs
  real(kind=real64) prediction, cost
  ! Declare a procedure pointer for the activation function
  procedure(fnn_activation_function), pointer :: activation
  ! Declare a procedure pointer for the derivative of the activation function
  procedure(fnn_derivative_activation_function), pointer :: dactivation

  !---------  f(x) = x^2 --------
  number_inputs = 1
  nullify(inputs)
  allocate(inputs(number_inputs))
  inputs(1) = 5d0
  ! result must be 25

  ! Sample for prediction
  nullify(samples)
  number_samples = 4
  allocate(samples(number_inputs + 1, number_samples))
  samples(1,1) = 0d0; samples(2,1) = 0d0
  samples(1,2) = 2d0; samples(2,2) = 4d0
  samples(1,3) = 4d0; samples(2,3) = 16d0
  samples(1,4) = 6d0; samples(2,4) = 36d0
  
  !!!!!!----- f(x, y) = x^2 + y^2 + K ----
  !!!!!! Input values to predict
  !!!!!number_inputs = 2
  !!!!!nullify(inputs)
  !!!!!allocate(inputs(number_inputs))
  !!!!!inputs(1) = 5d0
  !!!!!inputs(2) = 5d0
  !!!!!! result must be (for K = 0) 50
!!!!!
  !!!!!! Sample for prediction
  !!!!!nullify(samples)
  !!!!!number_samples = 4
  !!!!!allocate(samples(number_inputs + 1, number_samples))
  !!!!!samples(1,1) = 0; samples(2,1) = 0; samples(3,1) = 0
  !!!!!samples(1,2) = 3; samples(2,2) = 3; samples(3,2) = 18
  !!!!!samples(1,3) = 6; samples(2,3) = 6; samples(3,3) = 72
  !!!!!samples(1,4) = 5; samples(2,4) = 0; samples(3,4) = 25
  

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

  ! Loop over epocs
  number_of_epochs = 100000
  write(*,*) "Cost: "
  do epoch = 1, number_of_epochs
  
     ! Compute the cost
     error = cost_function_neuron(neuron, cost, number_inputs, number_samples, samples)
     write(*,*) epoch, cost
     
     ! Compute derivative of the cost

     error = derivative_cost_function_neuron(neuron, dcost, number_inputs, number_samples, samples)

     error = update_neuron(neuron, dcost, number_inputs, 0.001d0)
     
  enddo
  
  ! Make prediction
  error = prediction_neuron(neuron, prediction, number_inputs, inputs)
  write(*,*) "Prediction = ", prediction
  error = deallocate_neuron(neuron)

  if ( associated(inputs) ) deallocate(inputs)
  nullify(inputs)

  if ( associated(samples) ) deallocate(samples)
  nullify(samples)

  if ( associated(dcost) ) deallocate(dcost)
  nullify(dcost)
  
end program main
