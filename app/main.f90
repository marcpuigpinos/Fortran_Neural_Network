program main

  use iso_fortran_env, only: int32, real64
  use FortranNeuralNetwork
  
  implicit none

  type(fnn_neuron), pointer :: neuron
  real(kind=real64), allocatable :: inputs(:)
  integer(kind=int32) error, number_inputs
  real(kind=real64) prediction
  ! Declare a procedure pointer for the activation function
  procedure(activation_function), pointer :: activation

  number_inputs = 2
  if ( allocated(inputs) ) deallocate(inputs)
  allocate(inputs(number_inputs))
  inputs(1) = 10d0
  inputs(2) = 5d0
  error = allocate_neuron(neuron)
  error = initialize_neuron(neuron, number_inputs, inputs)
  call print_neuron(neuron, 0)
  !activation => sigmoid  ! Point to the sigmoid function
  activation => ReLU
  error = prediction_neuron(neuron, prediction, activation)
  write(*,*) "Prediction: ", prediction
  error = deallocate_neuron(neuron)

contains

  real(kind=real64) function sigmoid(x) result(y)
    real(kind=real64), intent(in) :: x
    y = 1d0 / ( 1d0 + exp(-x) )
  end function sigmoid

  real(kind=real64) function ReLU(x) result(y)
    real(kind=real64), intent(in) :: x
    y = max(0d0, x)
  end function ReLU
  
end program main
