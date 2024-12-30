module FortranNeuralNetwork

  use iso_fortran_env, only: int32, real64
  
  implicit none

  private

  ! Activation functions public interfaces
  public :: fnn_sigmoid, fnn_ReLU, fnn_derivative_sigmoid, fnn_derivative_ReLU, fnn_activation_function, fnn_derivative_activation_function

  ! Neuron public interfaces
  public :: fnn_neuron, allocate_neuron, deallocate_neuron, initialize_neuron, prediction_neuron, cost_function_neuron, derivative_cost_function_neuron, update_neuron, print_neuron

  !------ Activation function interface ------
  interface
     function fnn_activation_function(x) result(y)
       use iso_fortran_env, only: real64
       real(kind=real64), intent(in) :: x
       real(kind=real64) :: y
    end function fnn_activation_function
  end interface
  !------ End Activation function interface ------

  !------ Derivative Activation function interface ------
  interface
     function fnn_derivative_activation_function(x) result(y)
       use iso_fortran_env, only: real64
       real(kind=real64), intent(in) :: x
       real(kind=real64) :: y
     end function fnn_derivative_activation_function
  end interface
  !------ End Derivative Activation function interface ------
  
  !------ Neuron type definition ------
  type fnn_neuron
     logical :: allocated = .false.
     logical :: initialized = .false.
     integer(kind=int32) :: number_inputs
     real(kind=real64), allocatable :: weights(:)
     real(kind=real64) :: bias
     procedure(fnn_activation_function), nopass, pointer :: activation  => null()
     procedure(fnn_derivative_activation_function), nopass, pointer :: derivative_activation => null()
  end type fnn_neuron
  !------ End Neuron type definition ------
  
contains

  !------ Activation functions ------
  real(kind=real64) function fnn_sigmoid(x) result(y)
    real(kind=real64), intent(in) :: x
    y = 1d0 / ( 1d0 + exp(-x) )
  end function fnn_sigmoid

  real(kind=real64) function fnn_ReLU(x) result(y)
    real(kind=real64), intent(in) :: x
    y = max(0d0, x)
  end function fnn_ReLU
  !------ End Activation functions ------

  !------ Derivative activation functions ------
  real(kind=real64) function fnn_derivative_sigmoid(x) result(y)
    real(kind=real64), intent(in) :: x
    real(kind=real64) sigmoid
    sigmoid = fnn_sigmoid(x)
    y = sigmoid * ( 1 - sigmoid )
  end function fnn_derivative_sigmoid

  real(kind=real64) function fnn_derivative_ReLU(x) result(y)
    real(kind=real64), intent(in) :: x
    y = 0d0
    if ( x > 1.0e-9 ) y = 1d0
  end function fnn_derivative_ReLU  
  !------ End Derivative activation functions ------
  
  !------ Neuron procedures ------
  integer(kind=int32) function allocate_neuron(neuron) result(error)
    type(fnn_neuron), pointer :: neuron
    integer(kind=int32) :: status

    error = 0
    nullify(neuron)
    allocate(neuron, stat=status)
    if ( status /= 0 ) error = status
    neuron%number_inputs = 0
    if ( allocated(neuron%weights) ) deallocate(neuron%weights, stat=error)
    if ( status /= 0 ) error = status
    neuron%bias = 0d0
    nullify(neuron%activation)
    nullify(neuron%derivative_activation)
    neuron%allocated = .true.
  end function allocate_neuron
  
  integer(kind=int32) function deallocate_neuron(neuron) result(error)
    type(fnn_neuron), pointer :: neuron

    integer status
    
    error = 0
    if ( allocated(neuron%weights) ) deallocate(neuron%weights, stat=status)
    if ( status /= 0 ) error = status
    neuron%allocated = .false.
    neuron%initialized = .false.
    neuron%number_inputs = 0
    neuron%bias = 0d0
    nullify(neuron%activation)
    nullify(neuron%derivative_activation)
    nullify(neuron)
  end function deallocate_neuron
  
  integer(kind=int32) function initialize_neuron(neuron, number_inputs, activation, derivative_activation) result(error)
    type(fnn_neuron), pointer :: neuron
    integer(kind=int32), intent(in) :: number_inputs
    procedure(fnn_activation_function), pointer :: activation
    procedure(fnn_derivative_activation_function), pointer :: derivative_activation

    ! Initialize error
    error = 0

    ! Check if neuron is allocated
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    neuron%number_inputs = number_inputs
    allocate(neuron%weights(number_inputs), stat=error)
    if ( error /= 0 ) return
    call random_seed()
    call random_number(neuron%weights)
    call random_number(neuron%bias)
    neuron%activation => activation
    neuron%derivative_activation => derivative_activation
    neuron%initialized = .true.
    
  end function initialize_neuron

  integer(kind=int32) function prediction_neuron(neuron, prediction, n_inputs, inputs) result(error)
    type(fnn_neuron), pointer :: neuron
    real(kind=real64), intent(out) :: prediction
    integer(kind=int32), intent(in) :: n_inputs
    real(kind=real64), pointer :: inputs(:)
    integer i

    ! Initialize vars
    error = 0
    prediction = 0d0

    ! If neuron is not allocated, return with error.
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    ! If neuron is not initialized, return with error.
    if ( .not. neuron%initialized ) then
       error = 1
       return
    endif

    ! If number of inputs is not the same that the neuron, error
    if ( n_inputs /= neuron%number_inputs ) then
       error = 1
       return
    endif

    ! If not associated inputs, error
    if ( .not. associated(inputs) ) then
       error = 1
       return
    endif

    do i=1, neuron%number_inputs
       prediction = prediction + neuron%weights(i) * inputs(i)
    enddo

    prediction = prediction + neuron%bias

    ! Apply activation function
     prediction = neuron%activation(prediction)
    
  end function prediction_neuron

  integer(kind=int32) function dprediction_neuron(neuron, dprediction, n_inputs, inputs) result(error)
    type(fnn_neuron), pointer :: neuron
    real(kind=real64), intent(out) :: dprediction
    integer(kind=int32), intent(in) :: n_inputs
    real(kind=real64), pointer :: inputs(:)
    integer i

    ! Initialize vars
    error = 0
    dprediction = 0d0

    ! If neuron is not allocated, return with error.
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    ! If neuron is not initialized, return with error.
    if ( .not. neuron%initialized ) then
       error = 1
       return
    endif

    ! If number of inputs is not the same that the neuron, error
    if ( n_inputs /= neuron%number_inputs ) then
       error = 1
       return
    endif

    ! If not associated inputs, error
    if ( .not. associated(inputs) ) then
       error = 1
       return
    endif

    do i=1, neuron%number_inputs
       dprediction = dprediction + neuron%weights(i) * inputs(i)
    enddo

    dprediction = dprediction + neuron%bias

    ! Apply activation function
    dprediction = neuron%derivative_activation(dprediction) 
    
  end function dprediction_neuron
  
  
  integer(kind=int32) function cost_function_neuron(neuron, cost, n_inputs, n_samples, samples) result(error)
    type(fnn_neuron), pointer :: neuron
    real(kind=real64), intent(out) :: cost
    integer(kind=int32), intent(in) :: n_inputs, n_samples
    real(kind=real64), intent(in), pointer :: samples(:,:)

    integer sample_output, i, err_stat
    real(kind=real64) prediction, rval
    real(kind=real64), pointer :: inputs(:)
    
    error = 0
    cost = 0d0

    ! Nullify inputs pointer
    nullify(inputs)

    ! If neuron is not allocated, return with error.
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    ! If neuron is not initialized, return with error.
    if ( .not. neuron%initialized ) then
       error = 1
       return
    endif

    ! If the number of inputs of the samples array does not correspond to the number of inputs of the neuron, return with an error.
    if ( n_inputs /= neuron%number_inputs ) then
       error = 1
       return
    endif
    
    !samples(n_inputs + 1, n_samples)
    sample_output = n_inputs + 1
    do i = 1, n_samples
       inputs => samples(:, i)
       err_stat = prediction_neuron(neuron, prediction, n_inputs, inputs)
       error = error + err_stat
       rval = prediction - samples(sample_output, i)
       cost = cost + rval * rval
    enddo

    ! If there is an error, retur
    if ( error /= 0 ) return
    
    cost = cost / ( 2.0 * n_samples )

    nullify(inputs)
    
  end function cost_function_neuron

  integer(kind=int32) function derivative_cost_function_neuron(neuron, dcost, n_inputs, n_samples, samples) result(error)
    type(fnn_neuron), pointer :: neuron
    real(kind=real64), pointer :: dcost(:) ! dcost: 1:n_inputs -> weights; n_inputs + 1: bias
    integer(kind=int32), intent(in) :: n_inputs, n_samples
    real(kind=real64), intent(in), pointer :: samples(:,:)

    integer sample_output, i, err_stat
    real(kind=real64) prediction, dprediction, rval, dcost_l
    real(kind=real64), pointer :: inputs(:)
    
    error = 0
    dcost_l = 0d0
    dcost = 0d0
    nullify(inputs)    
    ! If neuron is not allocated, return with error.
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    ! If neuron is not initialized, return with error.
    if ( .not. neuron%initialized ) then
       error = 1
       return
    endif

    ! If the number of inputs of the samples array does not correspond to the number of inputs of the neuron, return with an error.
    if ( n_inputs /= neuron%number_inputs ) then
       error = 1
       return
    endif

    ! Check that dcost is correctly allocated and has the correct size
    if ( .not. associated(dcost) ) then
       error = 1
       return
    endif
    
    if ( size(dcost) /= n_inputs + 1 ) then
       error = 1
       return
    endif

    !samples(n_inputs + 1, n_samples)
    sample_output = n_inputs + 1
    do i = 1, n_samples
       inputs => samples(:, i)
       err_stat = prediction_neuron(neuron, prediction, n_inputs, inputs)
       error = error + err_stat
       rval = prediction - samples(sample_output, i)
       err_stat = dprediction_neuron(neuron, dprediction, n_inputs, inputs)
       rval = rval * dprediction
       dcost(1:n_inputs) = dcost(1:n_inputs) + rval * inputs(1:n_inputs)
       dcost(n_inputs + 1) = dcost(n_inputs + 1) + rval
       !dcost_l = dcost_l + rval
    enddo 
    
    ! If there is an error, retur
    if ( error /= 0 ) return
    
    !dcost_l = dcost_l / n_samples
    dcost = dcost / n_samples

    nullify(inputs)
  end function derivative_cost_function_neuron

  integer(kind=int32) function update_neuron(neuron, dcost, n_inputs, learning_rate) result(error)
    type(fnn_neuron), pointer :: neuron
    real(kind=real64), pointer :: dcost(:)
    integer(kind=int32), intent(in) :: n_inputs
    real(kind=real64), intent(in) :: learning_rate
    integer i

    error = 0
    
    ! If neuron is not allocated, return with error.
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    ! If neuron is not initialized, return with error.
    if ( .not. neuron%initialized ) then
       error = 1
       return
    endif

    ! If the number of inputs of the samples array does not correspond to the number of inputs of the neuron, return with an error.
    if ( n_inputs /= neuron%number_inputs ) then
       error = 1
       return
    endif

    ! Check that dcost is correctly allocated and has the correct size
    if ( .not. associated(dcost) ) then
       error = 1
       return
    endif
    
    if ( size(dcost) /= n_inputs + 1 ) then
       error = 1
       return
    endif

    ! Update the weights:
    do i = 1, neuron%number_inputs
       neuron%weights(i) = neuron%weights(i) - learning_rate * dcost(i)
    enddo
    
    ! Update the bias:
    neuron%bias = neuron%bias - learning_rate * dcost(n_inputs + 1)
    
  end function update_neuron
  
  subroutine print_neuron(neuron, padding)
    type(fnn_neuron), pointer :: neuron
    integer(kind=int32), intent(in) :: padding
    integer :: i
  
    ! Print the opening bracket for the neuron
    write(*, '(A)') repeat(' ', padding) // "["
    
    ! Print the weights
    write(*, '(A)') repeat(' ', padding + 5) // "weights = ["
    do i = 1, neuron%number_inputs
        write(*, '(A, F12.5)') repeat(' ', padding + 10),  neuron%weights(i)
    end do
    write(*, '(A)') repeat(' ', padding + 5) // "]"
    
    ! Print the bias
    write(*, '(A, F12.5)') repeat(' ', padding + 5) // "bias = ", neuron%bias
    
    ! Print the closing bracket for the neuron
    write(*, '(A)') repeat(' ', padding) // "]"
  end subroutine print_neuron
  !------ End Neuron procedures ------
  
end module FortranNeuralNetwork
