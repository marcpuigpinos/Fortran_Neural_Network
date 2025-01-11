module FortranNeuralNetwork

    use iso_fortran_env, only: ik => int32, rk => real64

    implicit none

    private

    ! Activation functions public interfaces
    public :: fnn_activation_function, fnn_derivative_activation_function, fnn_sigmoid, fnn_ReLU, fnn_derivative_sigmoid, fnn_derivative_ReLU

    !  public interfaces
    public :: fnn_net, fnn_add, fnn_compile

    ! temporal public layer
    public :: fnn_layer, allocate_layer, deallocate_layer, initialize_layer, prediction_layer, print_layer

    !------ Activation function interface ------
    interface
        function fnn_activation_function(x) result(y)
            use iso_fortran_env, only: rk => real64
            real(kind=rk), intent(in) :: x
            real(kind=rk) :: y
        end function fnn_activation_function
    end interface
    !------ End Activation function interface ------

    !------ Derivative Activation function interface ------
    interface
        function fnn_derivative_activation_function(x) result(y)
            use iso_fortran_env, only: rk => real64
            real(kind=rk), intent(in) :: x
            real(kind=rk) :: y
        end function fnn_derivative_activation_function
    end interface
    !------ End Derivative Activation function interface ------

    !------ Neuron type definition ------
    type fnn_neuron
        logical :: allocated = .false.
        logical :: initialized = .false.
        integer(kind=ik) :: number_inputs
        real(kind=rk), allocatable :: weights(:)
        real(kind=rk) :: bias
        procedure(fnn_activation_function), nopass, pointer :: activation => null()
        procedure(fnn_derivative_activation_function), nopass, pointer :: derivative_activation => null()
    end type fnn_neuron
    !------ End Neuron type definition ------

    !------ Layer type definition -------
    type fnn_layer
        logical :: allocated = .false.
        logical :: initialized = .false.
        integer(kind=ik) :: number_inputs, number_neurons
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
    real(kind=rk) function fnn_sigmoid(x) result(y)
        real(kind=rk), intent(in) :: x
        y = 1d0/(1d0 + exp(-x))
    end function fnn_sigmoid

    real(kind=rk) function fnn_ReLU(x) result(y)
        real(kind=rk), intent(in) :: x
        y = max(0d0, x)
    end function fnn_ReLU
    !------ End Activation functions ------

    !------ Derivative activation functions ------
    real(kind=rk) function fnn_derivative_sigmoid(x) result(y)
        real(kind=rk), intent(in) :: x
        real(kind=rk) sigmoid
        sigmoid = fnn_sigmoid(x)
        y = sigmoid*(1 - sigmoid)
    end function fnn_derivative_sigmoid

    real(kind=rk) function fnn_derivative_ReLU(x) result(y)
        real(kind=rk), intent(in) :: x
        y = 0d0
        if (x > 1.0e-9) y = 1d0
    end function fnn_derivative_ReLU
    !------ End Derivative activation functions ------

    !------ Neuron procedures ------
    integer(kind=ik) function allocate_neuron(neuron) result(error)
        type(fnn_neuron), pointer :: neuron
        integer(kind=ik) :: status

        ! Initialize error
        error = 0

        ! Nullify neuron
        nullify (neuron)

        ! Allocate neuron
        allocate (neuron, stat=status)
        if (status /= 0) then
            error = status
            return
        end if

        ! Parameters
        neuron%number_inputs = 0
        if (allocated(neuron%weights)) deallocate (neuron%weights, stat=status)
        if (status /= 0) then
            error = status
            return
        end if
        neuron%bias = 0d0

        ! Activation function
        nullify (neuron%activation)
        nullify (neuron%derivative_activation)

        ! Neuron state vars
        neuron%allocated = .true.
        neuron%initialized = .false.
    end function allocate_neuron

    integer(kind=ik) function deallocate_neuron(neuron) result(error)
        type(fnn_neuron), pointer :: neuron

        ! Local vars
        integer status

        ! Initialize error
        error = 0

        ! If neuron is not allocated, return
        if (.not. neuron%allocated) return

        ! Allocate weights
        if (allocated(neuron%weights)) deallocate (neuron%weights, stat=status)
        if (status /= 0) error = status

        ! Set neuron state vars
        neuron%allocated = .false.
        neuron%initialized = .false.

        ! Set vars to zero
        neuron%number_inputs = 0
        neuron%bias = 0d0

        ! Nullify pointers
        nullify (neuron%activation)
        nullify (neuron%derivative_activation)
        nullify (neuron)
    end function deallocate_neuron

    integer(kind=ik) function initialize_neuron(neuron, number_inputs, activation, derivative_activation) result(error)
        type(fnn_neuron), pointer :: neuron
        integer(kind=ik), intent(in) :: number_inputs
        procedure(fnn_activation_function), pointer :: activation
        procedure(fnn_derivative_activation_function), pointer :: derivative_activation

        ! Initialize error
        error = 0

        ! Check if neuron is allocated
        if (.not. neuron%allocated) then
            error = 1
            return
        end if

        ! Initialize neuron parameters
        neuron%number_inputs = number_inputs
        allocate (neuron%weights(number_inputs), stat=error)
        if (error /= 0) return
        call random_seed()
        call random_number(neuron%weights)
        call random_number(neuron%bias)

        ! Point activation and derivative_activaton function
        neuron%activation => activation
        neuron%derivative_activation => derivative_activation

        ! Set initialized neuron state variable to true
        neuron%initialized = .true.

    end function initialize_neuron

    integer(kind=ik) function prediction_neuron(neuron, prediction, n_inputs, inputs) result(error)
        type(fnn_neuron), pointer :: neuron
        real(kind=rk), intent(out) :: prediction
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)
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

    integer(kind=ik) function update_neuron(neuron, grad_cost, n_inputs, learning_rate) result(error)
        type(fnn_neuron), pointer :: neuron
        real(kind=rk), pointer :: grad_cost(:)
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), intent(in) :: learning_rate
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
        integer(kind=ik), intent(in) :: padding
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
    integer(kind=ik) function allocate_layer(layer) result(error)
        type(fnn_layer), pointer :: layer
        integer(kind=ik) :: status

        error = 0 ! Error initialization

        ! Allocate layer
        nullify (layer)
        allocate (layer, stat=status)
        if (status /= 0) then
            error = status
            return
        end if

        ! Inputs
        layer%number_inputs = 0

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

    integer(kind=ik) function deallocate_layer(layer) result(error)
        type(fnn_layer), pointer :: layer

        ! Local vars
        integer(kind=ik) i
        type(fnn_neuron), pointer :: neuron

        ! Nullify local pointers
        nullify(neuron)

        ! Initialize error
        error = 0

        ! If layer is not allocated simply return
        if ( .not. layer%allocated ) return

        ! Deallocate arrays
        if (allocated(layer%neurons)) then
            do i=1, layer%number_neurons
               neuron => layer%neurons(i)
               error = deallocate_neuron(neuron)
               if (error /= 0) return
            enddo
            deallocate(layer%neurons, stat=error)
            if (error /= 0) return
        endif

        ! Initialize integers
        layer%number_inputs = 0
        layer%number_neurons = 0

        ! Initialize logicals
        layer%allocated = .false.
        layer%initialized = .false.       

        ! nullify procedure pointers
        nullify(layer%activation, layer%derivative_activation)

        ! Nullify local pointers
        nullify(neuron)

    end function deallocate_layer

    integer(kind=ik) function initialize_layer(layer, number_inputs, number_neurons, activation, derivative_activation) result(error)
        type(fnn_layer), pointer :: layer
        integer, intent(in) :: number_inputs, number_neurons
        procedure(fnn_activation_function), pointer :: activation
        procedure(fnn_derivative_activation_function), pointer :: derivative_activation        

        ! Local vars
        integer(kind=ik) i
        type(fnn_neuron), pointer :: neuron

        ! Nullify local pointers
        nullify(neuron)

        ! Initialize error
        error = 0

        ! If layer is not allocated, return
        if (.not. layer%allocated) then
            error = 1
            return
        endif

        ! Point to the activation functions of the layer
        layer%activation => activation
        layer%derivative_activation => derivative_activation

        ! Inputs
        layer%number_inputs = number_inputs

        ! Number neurons
        layer%number_neurons = number_neurons
        if (allocated(layer%neurons)) deallocate(layer%neurons, stat=error)
        if (error /= 0) return
        allocate(layer%neurons(layer%number_neurons), stat=error)
        if (error /= 0) return
        do i=1, layer%number_neurons
           neuron => layer%neurons(i)
           error = allocate_neuron(neuron)
           if (error /= 0) return
           error = initialize_neuron(neuron, number_inputs, activation, derivative_activation)
           if (error /= 0) return
        enddo

        ! Nullify local pointers
        nullify(neuron)        

        ! Once every task is done, set the initialized variable to true.
        layer%initialized = .true.

    end function initialize_layer

    integer(kind=ik) function prediction_layer(layer, prediction, n_inputs, inputs) result(error)
        type(fnn_layer), pointer :: layer
        real(kind=rk), pointer :: prediction(:) !layer%number_neurons length
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)

        ! Local vars
        integer(kind=ik) ineuron
        type(fnn_neuron), pointer :: neuron

        ! Nullify
        nullify(neuron)

        ! Initialize error
        error = 0

        ! Check if layer is allocated
        if (.not. layer%allocated) then
            error = 1
            return
        endif

        ! Check if layer is initialized
        if (.not. layer%initialized) then
            error = 1
            return
        endif

        ! Check if n_inputs is the same that layer number_inputs
        if (n_inputs /= layer%number_inputs) then
            error = 1
            return
        endif
        
        ! Loop over neurons to compute prediction
        do ineuron = 1, layer%number_neurons
           neuron => layer%neurons(ineuron)
           error = prediction_neuron(neuron, prediction(ineuron), layer%number_inputs, inputs)
        enddo

        ! Nullify
        nullify(neuron)        

    end function prediction_layer

    subroutine print_layer(layer, padding)
        type(fnn_layer), pointer :: layer
        integer(kind=ik), intent(in) :: padding

        ! Local var
        integer(kind=ik) ineuron
        type(fnn_neuron), pointer :: neuron

        ! Nullify local pointers
        nullify(neuron)
        
        ! Print the opening bracket for the neuron
        write (*, '(A)') repeat(' ', padding)//"Layer: ["
        do ineuron = 1, layer%number_neurons
           neuron => layer%neurons(ineuron)
           call print_neuron(neuron, padding + 4)
        enddo
        write(*, '(A)') repeat(' ', padding)//"]"

        ! Nullify local pointers
        nullify(neuron)
        
    end subroutine print_layer
    
    integer(kind=ik) function update_layer(layer) result(error)
        type(fnn_layer), pointer :: layer
        error = 0
    end function update_layer
    !------ End Layer procedures ------

    !------ Network procedures -------
    !------ End Network procedures ------

    !------ Public procedures of the neural network ------
    integer(kind=ik) function fnn_net(network) result(error)
        type(fnn_network), pointer :: network
        error = 0
    end function fnn_net

    integer(kind=ik) function fnn_add(network) result(error)
        type(fnn_network), pointer :: network
        error = 0
    end function fnn_add

    integer(kind=ik) function fnn_compile(network) result(error)
        type(fnn_network), pointer :: network
        error = 0
    end function fnn_compile

end module FortranNeuralNetwork
