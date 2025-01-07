module FortranNeuralNetwork

   use iso_fortran_env, only: int32, real64

   implicit none

   private

   ! Activation functions public interfaces
  public :: fnn_sigmoid, fnn_ReLU, fnn_derivative_sigmoid, fnn_derivative_ReLU, fnn_activation_function, fnn_derivative_activation_function

   !  public interfaces
   public :: fnn_net, fnn_add, fnn_compile
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
      procedure(fnn_activation_function), nopass, pointer :: activation => null()
      procedure(fnn_derivative_activation_function), nopass, pointer :: derivative_activation => null()
   end type fnn_neuron
   !------ End Neuron type definition ------

   !------ Layer type definition -------
   type fnn_layer
      logical :: allocated = .false.
      logical :: initialized = .false.
      integer(kind=int32) :: number_inputs, number_neurons
      real(kind=real64), allocatable :: inputs(:)
      type(fnn_neuron), allocatable :: neurons(:)
      procedure(fnn_activation_function), nopass, pointer :: activation => null()
      procedure(fnn_derivative_activation_function), nopass, pointer :: derivative_activation => null()
   end type fnn_layer
   !------ End Layer type definition ------

   !------ Network type definition -------
   type fnn_network
      logical :: allocated = .false.
      logical :: initialized = .false.
   end type fnn_network
   !------ End Network type definition -------

contains

   !------ Activation functions ------
   real(kind=real64) function fnn_sigmoid(x) result(y)
      real(kind=real64), intent(in) :: x
      y = 1d0/(1d0 + exp(-x))
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
      y = sigmoid*(1 - sigmoid)
   end function fnn_derivative_sigmoid

   real(kind=real64) function fnn_derivative_ReLU(x) result(y)
      real(kind=real64), intent(in) :: x
      y = 0d0
      if (x > 1.0e-9) y = 1d0
   end function fnn_derivative_ReLU
   !------ End Derivative activation functions ------

   !------ Neuron procedures ------
   integer(kind=int32) function allocate_neuron(neuron) result(error)
      type(fnn_neuron), pointer :: neuron
      integer(kind=int32) :: status

      error = 0
      nullify (neuron)
      allocate (neuron, stat=status)
      if (status /= 0) then
         error = status
         return
      end if
      neuron%number_inputs = 0
      if (allocated(neuron%weights)) deallocate (neuron%weights, stat=status)
      if (status /= 0) then
         error = status
         return
      end if
      neuron%bias = 0d0
      nullify (neuron%activation)
      nullify (neuron%derivative_activation)
      neuron%allocated = .true.
      neuron%initialized = .false.
   end function allocate_neuron

   integer(kind=int32) function deallocate_neuron(neuron) result(error)
      type(fnn_neuron), pointer :: neuron

      integer status

      error = 0
      if (allocated(neuron%weights)) deallocate (neuron%weights, stat=status)
      if (status /= 0) error = status
      neuron%allocated = .false.
      neuron%initialized = .false.
      neuron%number_inputs = 0
      neuron%bias = 0d0
      nullify (neuron%activation)
      nullify (neuron%derivative_activation)
      nullify (neuron)
   end function deallocate_neuron

   integer(kind=int32) function initialize_neuron(neuron, number_inputs, activation, derivative_activation) result(error)
      type(fnn_neuron), pointer :: neuron
      integer(kind=int32), intent(in) :: number_inputs
      procedure(fnn_activation_function), pointer :: activation
      procedure(fnn_derivative_activation_function), pointer :: derivative_activation

      ! Initialize error
      error = 0

      ! Check if neuron is allocated
      if (.not. neuron%allocated) then
         error = 1
         return
      end if

      neuron%number_inputs = number_inputs
      allocate (neuron%weights(number_inputs), stat=error)
      if (error /= 0) return
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
      if (.not. neuron%allocated) then
         error = 1
         return
      end if

      ! If neuron is not initialized, return with error.
      if (.not. neuron%initialized) then
         error = 1
         return
      end if

      ! If number of inputs is not the same that the neuron, error
      if (n_inputs /= neuron%number_inputs) then
         error = 1
         return
      end if

      ! If not associated inputs, error
      if (.not. associated(inputs)) then
         error = 1
         return
      end if

      do i = 1, neuron%number_inputs
         prediction = prediction + neuron%weights(i)*inputs(i)
      end do

      prediction = prediction + neuron%bias

      ! Apply activation function
      prediction = neuron%activation(prediction)

   end function prediction_neuron

   integer(kind=int32) function update_neuron(neuron, grad_cost, n_inputs, learning_rate) result(error)
      type(fnn_neuron), pointer :: neuron
      real(kind=real64), pointer :: grad_cost(:)
      integer(kind=int32), intent(in) :: n_inputs
      real(kind=real64), intent(in) :: learning_rate
      integer i

      error = 0

      ! If neuron is not allocated, return with error.
      if (.not. neuron%allocated) then
         error = 1
         return
      end if

      ! If neuron is not initialized, return with error.
      if (.not. neuron%initialized) then
         error = 1
         return
      end if

      ! If the number of inputs of the samples array does not correspond to the number of inputs of the neuron, return with an error.
      if (n_inputs /= neuron%number_inputs) then
         error = 1
         return
      end if

      ! Check that grad_cost is correctly allocated and has the correct size
      if (.not. associated(grad_cost)) then
         error = 1
         return
      end if

      if (size(grad_cost) /= n_inputs + 1) then
         error = 1
         return
      end if

      ! Update the weights:
      do i = 1, neuron%number_inputs
         neuron%weights(i) = neuron%weights(i) - learning_rate*grad_cost(i)
      end do

      ! Update the bias:
      neuron%bias = neuron%bias - learning_rate*grad_cost(n_inputs + 1)

   end function update_neuron

   subroutine print_neuron(neuron, padding)
      type(fnn_neuron), pointer :: neuron
      integer(kind=int32), intent(in) :: padding
      integer :: i

      ! Print the opening bracket for the neuron
      write (*, '(A)') repeat(' ', padding)//"["

      ! Print the weights
      write (*, '(A)') repeat(' ', padding + 5)//"weights = ["
      do i = 1, neuron%number_inputs
         write (*, '(A, F12.5)') repeat(' ', padding + 10), neuron%weights(i)
      end do
      write (*, '(A)') repeat(' ', padding + 5)//"]"

      ! Print the bias
      write (*, '(A, F12.5)') repeat(' ', padding + 5)//"bias = ", neuron%bias

      ! Print the closing bracket for the neuron
      write (*, '(A)') repeat(' ', padding)//"]"
   end subroutine print_neuron
   !------ End Neuron procedures ------

   !------ Layer procedures ------
   integer(kind=int32) function allocate_layer(layer) result(error)
      type(fnn_layer), pointer :: layer
      integer(kind=int32) :: status

      error = 0 ! Error initialization

      ! Allcoate layer
      nullify (layer)
      allocate (layer, stat=status)
      if (status /= 0) then
         error = status
         return
      end if

      ! Allocate layer inputs
      layer%number_inputs = 0
      if (allocated(layer%inputs)) deallocate (layer%inputs, stat=status)
      if (status /= 0) then
         error = status
         return
      end if

      ! Allocate layer neurons
      layer%number_neurons = 0
      if (allocated(layer%neurons)) deallocate (layer%neurons, stat=status)
      if (status /= 0) then
         error = status
         return
      end if

      ! Nullify procedure pointers
      nullify (layer%activation)
      nullify (layer%derivative_activation)
      layer%allocated = .true.
      layer%initialized = .false.

   end function allocate_layer

   integer(kind=int32) function deallocate_layer(layer) result(error)
      type(fnn_layer), pointer :: layer
      error = 0
   end function deallocate_layer

   integer(kind=int32) function initialize_layer(layer) result(error)
      type(fnn_layer), pointer :: layer
      error = 0
   end function initialize_layer

   integer(kind=int32) function prediction_layer(layer) result(error)
      type(fnn_layer), pointer :: layer
      error = 0
   end function prediction_layer

   integer(kind=int32) function update_layer(layer) result(error)
      type(fnn_layer), pointer :: layer
      error = 0
   end function update_layer
   !------ End Layer procedures ------

   !------ Network procedures -------
   !------ End Network procedures ------

   !------ Public procedures of the neural network ------
   integer(kind=int32) function fnn_net(network) result(error)
      type(fnn_network), pointer :: network
      error = 0
   end function fnn_net

   integer(kind=int32) function fnn_add(network) result(error)
      type(fnn_network), pointer :: network
      error = 0
   end function fnn_add

   integer(kind=int32) function fnn_compile(network) result(error)
      type(fnn_network), pointer :: network
      error = 0
   end function fnn_compile

end module FortranNeuralNetwork
