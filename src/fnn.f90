module fnn

  use iso_fortran_env, only: int32, real64
  
  implicit none

  private

  public :: fnn_neuron, activation_function, allocate_neuron, deallocate_neuron, initialize_neuron, prediction_neuron, print_neuron

  ! Neuron type definition
  type fnn_neuron
     logical :: allocated = .false.
     logical :: initialized = .false.
     integer(kind=int32) :: number_inputs
     real(kind=real64), allocatable :: inputs(:)
     real(kind=real64), allocatable :: weights(:)
     real(kind=real64) :: bias
  end type fnn_neuron

  ! Activation function interface
  abstract interface
     function activation_function(x) result(y)
       use iso_fortran_env, only: real64
       real(kind=real64), intent(in) :: x
       real(kind=real64) :: y
    end function activation_function
  end interface
  
contains

  ! Neuron procedures 
  integer(kind=int32) function allocate_neuron(neuron) result(error)
    type(fnn_neuron), pointer :: neuron
    integer(kind=int32) :: status

    error = 0
    nullify(neuron)
    allocate(neuron, stat=status)
    if ( status /= 0 ) error = status
    neuron%number_inputs = 0
    if ( allocated(neuron%inputs) ) deallocate(neuron%inputs, stat=error)
    if ( status /= 0 ) error = status    
    if ( allocated(neuron%weights) ) deallocate(neuron%inputs, stat=error)
    if ( status /= 0 ) error = status
    neuron%bias = 0d0
    neuron%allocated = .true.
  end function allocate_neuron
  
  integer(kind=int32) function deallocate_neuron(neuron) result(error)
    type(fnn_neuron), pointer :: neuron

    integer status
    
    error = 0
    if ( allocated(neuron%inputs) ) deallocate(neuron%inputs, stat=status)
    if ( status /= 0 ) error = status    
    if ( allocated(neuron%weights) ) deallocate(neuron%inputs, stat=status)
    if ( status /= 0 ) error = status
    neuron%allocated = .false.
    neuron%initialized = .false.
    neuron%number_inputs = 0
    neuron%bias = 0d0
    nullify(neuron)
  end function deallocate_neuron
  
  integer(kind=int32) function initialize_neuron(neuron, number_inputs, inputs) result(error)
    type(fnn_neuron), pointer :: neuron
    integer(kind=int32), intent(in) :: number_inputs
    real(kind=real64), allocatable, intent(in) :: inputs(:)

    error = 0
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    if ( number_inputs /= size(inputs) ) then
       error = 1
       return
    endif

    neuron%number_inputs = number_inputs
    allocate(neuron%inputs(number_inputs), stat=error)
    if ( error /= 0 ) return
    neuron%inputs = inputs
    allocate(neuron%weights(number_inputs), stat=error)
    if ( error /= 0 ) return
    call random_seed()
    call random_number(neuron%weights)
    call random_number(neuron%bias)
    neuron%initialized = .true.
    
  end function initialize_neuron

  real(kind=int32) function prediction_neuron(neuron, prediction, activation) result(error)
    type(fnn_neuron), pointer :: neuron
    real(kind=real64), intent(out) :: prediction
    procedure(activation_function), pointer :: activation
    integer i

    error = 0
    prediction = 0d0
    
    if ( .not. neuron%allocated ) then
       error = 1
       return
    endif

    if ( .not. neuron%initialized ) then
       error = 1
       return
    endif

    do i=1, neuron%number_inputs
       prediction = prediction + neuron%weights(i) * neuron%inputs(i)
    enddo

    prediction = prediction + neuron%bias

    ! Apply activation function
     prediction = activation(prediction)
    
  end function prediction_neuron
  
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
  
end module fnn
