module FortranNeuralNetwork

    use iso_fortran_env, only: ik => int32, rk => real64

    implicit none

    private

    ! Activation functions public interfaces and procedures
    public :: fnn_activation_function, fnn_derivative_activation_function, fnn_sigmoid, &
              fnn_ReLU, fnn_derivative_sigmoid, fnn_derivative_ReLU

    ! Cost functions public interfaces and procedures
    public :: fnn_cost_function, fnn_cost_MSE, fnn_cost_cross_entropy

    !  public interfaces
    public :: fnn_net, fnn_add, fnn_predict, fnn_train, fnn_print

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

    !------ Cost function interface -------
    interface
        real(kind=rk) function fnn_cost_function(yp, y, n_predictions, n_samples) result(c)
            use iso_fortran_env, only: ik => int32, rk => real64
            real(kind=rk), pointer :: yp(:, :), y(:, :) ! Arrays n_samples x n_predictions
            integer(kind=ik), intent(in) :: n_predictions, n_samples
        end function fnn_cost_function
    end interface
    !------ End cost function interface -------

    !------ Neuron type definition ------
    type fnn_neuron
        logical :: allocated = .false.
        logical :: initialized = .false.
        integer(kind=ik) :: number_inputs
        real(kind=rk), allocatable :: weights(:)
        real(kind=rk), allocatable :: temp_weights(:)
        real(kind=rk) :: z
        real(kind=rk) :: a
        procedure(fnn_activation_function), nopass, pointer :: activation => null()
        procedure(fnn_derivative_activation_function), nopass, pointer :: derivative_activation => null()
    end type fnn_neuron

    type fnn_neuron_pointer
        type(fnn_neuron), pointer :: neuron
    end type fnn_neuron_pointer

    !------ End Neuron type definition ------

    !------ Layer type definition -------
    type fnn_layer
        logical :: allocated = .false.
        logical :: initialized = .false.
        integer(kind=ik) :: number_inputs, number_neurons
        type(fnn_neuron_pointer), allocatable :: neurons(:)
        procedure(fnn_activation_function), nopass, pointer :: activation => null()
        procedure(fnn_derivative_activation_function), nopass, pointer :: derivative_activation => null()
    end type fnn_layer

    type fnn_layer_pointer
        type(fnn_layer), pointer :: layer
    end type fnn_layer_pointer
    !------ End Layer type definition ------

    !------ Network type definition -------
    type fnn_network
        logical :: allocated = .false.
        logical :: initialized = .false.
        integer(kind=ik) :: number_inputs, number_layers, max_number_neurons
        type(fnn_layer_pointer), allocatable :: layers(:)
    end type fnn_network
    !------ End Network type definition -------

    !------ Network control vars  ------
    type(fnn_network), pointer :: net
    logical :: alloc = .false.
    logical :: init = .false.
    integer :: layer_id = 0

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

    !------ Cost functions ------
    real(kind=rk) function fnn_cost_MSE(yp, y, n_predictions, n_samples) result(c)
        real(kind=rk), pointer :: yp(:, :), y(:, :) ! Arrays n_predictions x n_samples
        integer(kind=ik), intent(in) :: n_predictions, n_samples

        ! Local vars
        real(kind=rk), allocatable :: diff(:)
        integer(kind=ik) i_sample, i_prediction

        ! Initialie cost
        c = 0d0

        ! Reserve memory
        allocate (diff(n_predictions))

        ! Loop over samples
        do i_sample = 1, n_samples
            do i_prediction = 1, n_predictions
                diff(i_prediction) = yp(i_prediction, i_sample) - y(i_prediction, i_sample)
            end do
            c = c + dot_product(diff, diff)
        end do

        ! Average cost
        c = 0.5d0*c/n_samples

        !Free memory
        deallocate (diff)

    end function fnn_cost_MSE

    ! Cross entropy procedure: https://en.wikipedia.org/wiki/Cross-entropy
    real(kind=rk) function fnn_cost_cross_entropy(yp, y, n_predictions, n_samples) result(c)
        real(kind=rk), pointer :: yp(:, :), y(:, :) ! Arrays n_samples x n_predictions
        integer(kind=ik), intent(in) :: n_predictions, n_samples
        integer(kind=ik) isample, ipred
        c = 0d0
        do isample = 1, n_samples
            do ipred = 1, n_predictions
                c = c + (y(ipred, isample)*log(yp(ipred, isample)) + (1d0-y(ipred, isample))*log(1d0 -yp(ipred, isample))) 
            enddo
        enddo
        c = -c / real(n_samples, kind=rk)
    end function fnn_cost_cross_entropy
    !------ End cost functions ------

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
        if (allocated(neuron%temp_weights)) deallocate (neuron%temp_weights, stat=status)
        if (status /= 0) then
            error = status
            return
        end if
        neuron%z = 0d0
        neuron%a = 0d0

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

        ! deAllocate weights
        if (allocated(neuron%weights)) deallocate (neuron%weights, stat=status)
        if (status /= 0) error = status
        if (allocated(neuron%temp_weights)) deallocate (neuron%temp_weights, stat=status)
        if (status /= 0) error = status

        ! Set neuron state vars
        neuron%allocated = .false.
        neuron%initialized = .false.

        ! Set vars to zero
        neuron%number_inputs = 0
        neuron%z = 0d0
        neuron%a = 0d0

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

        allocate (neuron%temp_weights(number_inputs), stat=error)
        if (error /= 0) return
        neuron%temp_weights = neuron%weights

        ! Point activation and derivative_activaton function
        neuron%activation => activation
        neuron%derivative_activation => derivative_activation

        ! Set initialized neuron state variable to true
        neuron%initialized = .true.

    end function initialize_neuron

    integer(kind=ik) function activation_neuron(neuron, n_inputs, inputs) result(error)
        type(fnn_neuron), pointer :: neuron
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)
        integer i

        ! Initialize vars
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

        ! Compute z
        neuron%z = dot_product(neuron%weights(1:neuron%number_inputs), inputs(1:neuron%number_inputs))

        ! Apply activation function
        neuron%a = neuron%activation(neuron%z)

    end function activation_neuron

    integer(kind=ik) function update_neuron(neuron, grad_cost, n_inputs, learning_rate) result(error)
        type(fnn_neuron), pointer :: neuron
        real(kind=rk), pointer :: grad_cost(:)
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), intent(in) :: learning_rate
        integer i

        ! Initialize error
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

        if (size(grad_cost) /= n_inputs) then
            error = 1
            return
        end if

        ! Update the weights:
        neuron%weights(:) = neuron%weights(:) - learning_rate*grad_cost(:)
        neuron%temp_weights(:) = neuron%weights(:)

    end function update_neuron

    subroutine print_neuron(neuron, padding)
        type(fnn_neuron), pointer :: neuron
        integer(kind=ik), intent(in) :: padding
        integer :: i

        ! Print the opening bracket for the neuron
        write (*, '(A)') repeat(' ', padding)//"Neuron: ["

        ! Print the weights
        write (*, '(A)') repeat(' ', padding + 5)//"w = ["
        do i = 1, neuron%number_inputs
            write (*, '(A, F12.5)') repeat(' ', padding + 10), neuron%weights(i)
        end do
        write (*, '(A)') repeat(' ', padding + 5)//"]"

        write (*, '(A, F12.5)') repeat(' ', padding + 5)//"z = ", neuron%z
        write (*, '(A, F12.5)') repeat(' ', padding + 5)//"a = ", neuron%a

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

        ! Initialize error
        error = 0

        ! If layer is not allocated simply return
        if (.not. layer%allocated) return

        ! Deallocate arrays
        if (allocated(layer%neurons)) then
            do i = 1, layer%number_neurons
                error = deallocate_neuron(layer%neurons(i)%neuron)
                if (error /= 0) return
            end do
            deallocate (layer%neurons, stat=error)
            if (error /= 0) return
        end if

        ! Initialize integers
        layer%number_inputs = 0
        layer%number_neurons = 0

        ! Initialize logicals
        layer%allocated = .false.
        layer%initialized = .false.

        ! nullify procedure pointers
        nullify (layer%activation, layer%derivative_activation)

    end function deallocate_layer

   integer(kind=ik) function initialize_layer(layer, number_inputs, number_neurons, activation, derivative_activation) result(error)
        type(fnn_layer), pointer :: layer
        integer, intent(in) :: number_inputs, number_neurons
        procedure(fnn_activation_function), pointer :: activation
        procedure(fnn_derivative_activation_function), pointer :: derivative_activation

        ! Local vars
        integer(kind=ik) i

        ! Initialize error
        error = 0

        ! If layer is not allocated, return
        if (.not. layer%allocated) then
            error = 1
            return
        end if

        ! Point to the activation functions of the layer
        layer%activation => activation
        layer%derivative_activation => derivative_activation

        ! Inputs
        layer%number_inputs = number_inputs

        ! Number neurons
        layer%number_neurons = number_neurons
        if (allocated(layer%neurons)) deallocate (layer%neurons, stat=error)
        if (error /= 0) return
        allocate (layer%neurons(layer%number_neurons), stat=error)
        if (error /= 0) return
        do i = 1, layer%number_neurons
            error = allocate_neuron(layer%neurons(i)%neuron)
            if (error /= 0) return
            error = initialize_neuron(layer%neurons(i)%neuron, number_inputs, activation, derivative_activation)
            if (error /= 0) return
        end do

        ! Once every task is done, set the initialized variable to true.
        layer%initialized = .true.

    end function initialize_layer

    integer(kind=ik) function activations_layer(layer, activations, n_inputs, inputs) result(error)
        type(fnn_layer), pointer :: layer
        real(kind=rk), pointer :: activations(:) !layer%number_neurons length
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)

        ! Local vars
        integer(kind=ik) ineuron
        type(fnn_neuron), pointer :: neuron

        ! Nullify
        nullify (neuron)

        ! Initialize error
        error = 0

        ! Check if layer is allocated
        if (.not. layer%allocated) then
            error = 1
            return
        end if

        ! Check if layer is initialized
        if (.not. layer%initialized) then
            error = 1
            return
        end if

        ! Check if n_inputs is the same that layer number_inputs
        if (n_inputs /= layer%number_inputs) then
            error = 1
            return
        end if

        ! Loop over neurons to compute prediction
        do ineuron = 1, layer%number_neurons
            neuron => layer%neurons(ineuron)%neuron
            error = activation_neuron(neuron, layer%number_inputs, inputs)
            if (error /= 0) return
            activations(ineuron) = neuron%a
        end do

        ! Nullify
        nullify (neuron)

    end function activations_layer

    integer(kind=ik) function update_layer(layer, grad_cost, n_inputs, learning_rate) result(error)
        type(fnn_layer), pointer :: layer
        real(kind=rk), pointer :: grad_cost(:)
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), intent(in) :: learning_rate

        ! Local var
        integer(kind=ik) :: ineuron

        ! Initialize error
        error = 0

        ! Check if layer is allocated
        if (.not. layer%allocated) then
            error = 1
            return
        end if

        ! Check if layer is initialized
        if (.not. layer%initialized) then
            error = 1
            return
        end if

        ! Check if n_inputs is the same that layer number_inputs
        if (n_inputs /= layer%number_inputs) then
            error = 1
            return
        end if

        ! Loop over each neuron and update
        do ineuron = 1, layer%number_neurons
            error = update_neuron(layer%neurons(ineuron)%neuron, grad_cost, n_inputs, learning_rate)
        end do

    end function update_layer

    subroutine print_layer(layer, padding)
        type(fnn_layer), pointer :: layer
        integer(kind=ik), intent(in) :: padding

        ! Local var
        integer(kind=ik) ineuron
        type(fnn_neuron), pointer :: neuron

        ! Nullify local pointers
        nullify (neuron)

        ! Print the opening bracket for the neuron
        write (*, '(A)') repeat(' ', padding)//"Layer: ["
        do ineuron = 1, layer%number_neurons
            neuron => layer%neurons(ineuron)%neuron
            call print_neuron(neuron, padding + 4)
        end do
        write (*, '(A)') repeat(' ', padding)//"]"

        ! Nullify local pointers
        nullify (neuron)

    end subroutine print_layer
    !------ End Layer procedures ------

    !------ Network procedures -------
    integer(kind=ik) function allocate_network(network) result(error)
        type(fnn_network), pointer :: network

        ! Initialize error
        error = 0

        ! Nullify and allocate network
        nullify (network)
        allocate (network, stat=error)
        if (error /= 0) return

        ! Initialize scalars
        network%number_inputs = 0
        network%number_layers = 0
        network%max_number_neurons = 0

        ! Free Network memory
        if (allocated(network%layers)) deallocate (network%layers, stat=error)
        if (error /= 0) return

        ! Set state vars
        network%allocated = .true.
        network%initialized = .false.

    end function allocate_network

    integer(kind=ik) function deallocate_network(network) result(error)
        type(fnn_network), pointer :: network

        ! Initialize error
        error = 0

        ! Free network memory:
        if (allocated(network%layers)) deallocate (network%layers, stat=error)
        if (error /= 0) return

        ! Set scalars to zero
        network%number_inputs = 0
        network%number_layers = 0
        network%max_number_neurons = 0

        ! Set status variables to false
        network%allocated = .false.
        network%initialized = .false.

        ! Free memory
        if (associated(network)) deallocate (network, stat=error)
        if (error /= 0) return

        ! Nullify network pointer
        nullify (network)

    end function deallocate_network

    integer(kind=ik) function initialize_network(network, number_inputs, number_layers) result(error)
        type(fnn_network), pointer :: network
        integer(kind=ik), intent(in) :: number_inputs, number_layers

        ! Local vars
        integer(kind=ik) ilayer

        ! Initialize error
        error = 0

        ! Check if network is allcoated
        if (.not. associated(network)) then
            error = 1
            return
        end if

        ! Save scalars
        network%number_inputs = number_inputs
        network%number_layers = number_layers
        network%max_number_neurons = 0

        ! Reserve memory
        allocate (network%layers(number_layers), stat=error)
        if (error /= 0) return

        ! Nullify pointers of layers
        do ilayer = 1, network%number_layers
            nullify (network%layers(ilayer)%layer)
        end do

        ! Set initialize state to true
        network%initialized = .true.

    end function initialize_network

  integer(kind=ik) function add_layer_to_network(network, layer_id, number_neurons, activation, derivative_activation) result(error)
        type(fnn_network), pointer :: network
        integer(kind=ik), intent(in) :: layer_id, number_neurons
        procedure(fnn_activation_function), pointer :: activation
        procedure(fnn_derivative_activation_function), pointer :: derivative_activation

        ! Initialize error
        error = 0

        ! If network is not initialized return with error
        if (.not. network%initialized) then
            error = 1
            return
        end if

        ! If layer_id is not > 0 return with error
        if (layer_id < 1) then
            error = 2
            return
        end if

        ! Add the layer
       !! If layer is already associated, then return with error
        if (associated(network%layers(layer_id)%layer)) then
            error = 3
            return
        end if

        ! Check max number of neurons
        if (number_neurons > network%max_number_neurons) network%max_number_neurons = number_neurons

        error = allocate_layer(network%layers(layer_id)%layer)
        if (error /= 0) return

        ! If it is first layer, initialize with number of inputs
        ! else it is not the first layer, initialize with number of inputs
        ! equal to number of neurons of layer_id - 1
        if (layer_id == 1) then
  error = initialize_layer(network%layers(layer_id)%layer, network%number_inputs, number_neurons, activation, derivative_activation)
            if (error /= 0) return
        else
            !The input of this layer is the output + 1(bias) of the last layer
            error = initialize_layer(network%layers(layer_id)%layer, network%layers(layer_id - 1)%layer%number_neurons + 1, number_neurons, activation, derivative_activation)
            if (error /= 0) return
        end if

    end function add_layer_to_network

    integer(kind=ik) function activate_network(network, predictions, n_inputs, inputs) result(error)
        type(fnn_network), pointer :: network
        real(kind=rk), pointer :: predictions(:)
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)

        ! Local vars
        real(kind=rk), pointer :: activations(:)
        real(kind=rk), pointer :: local_inputs(:)
        integer(kind=ik) ilayer, local_number_neurons, local_number_inputs

        ! Nullify local pointers
        nullify (activations, local_inputs)

        ! Initialize error
        error = 0

        ! Check if network is allocated and initialized
        if ((.not. network%allocated) .or. (.not. network%initialized)) then
            error = 1
            return
        end if

        ! If n_inputs is different that first layer n_inputs exit with error
        if (n_inputs /= network%layers(1)%layer%number_inputs) then
            error = 2
            return
        end if

        ! Check if predictions is associated
        if (.not. associated(predictions)) then
            error = 3
            return
        end if

        ! Check if predictions array has the same size af the last layer number of layers
        if (size(predictions) /= network%layers(network%number_layers)%layer%number_neurons) then
            error = 4
            return
        end if

        ! Check that the number of inputs is the same that the network first layer
        if (n_inputs /= network%layers(1)%layer%number_inputs) then
            error = 5
            return
        end if

        ! Allocate activations array with the maximum number of neurons
        allocate (activations(network%max_number_neurons), stat=error)
        if (error /= 0) return

        !Allocate local inputs with the maximum between n_inputs and max_number_neurons + 1 (bias)
        allocate (local_inputs(max(n_inputs, network%max_number_neurons + 1)), stat=error)
        if (error /= 0) return

        ! Initialize local_inputs
        local_inputs(1:n_inputs) = inputs(1:n_inputs)

        ! Compute the activations
        do ilayer = 1, network%number_layers
            local_number_inputs = network%layers(ilayer)%layer%number_inputs
            local_number_neurons = network%layers(ilayer)%layer%number_neurons
            error = activations_layer(network%layers(ilayer)%layer, activations, local_number_inputs, local_inputs)
            local_inputs(1) = 1.0d0 ! Bias
            local_inputs(2:local_number_neurons+1) = activations(1:local_number_neurons) ! Weights
        end do

        ! Output
        predictions(1:network%layers(network%number_layers)%layer%number_neurons) = &
            activations(1:network%layers(network%number_layers)%layer%number_neurons)

        ! Free memmory
        if (associated(local_inputs)) deallocate (local_inputs, stat=error)
        if (error /= 0) return
        if (associated(activations)) deallocate (activations, stat=error)
        if (error /= 0) return
        nullify (local_inputs, activations)

    end function activate_network

    integer(kind=ik) function cost_network(network, number_inputs, number_predictions, number_samples, samples, expected, &
                                           cost_function, cost) result(error)
        type(fnn_network), pointer :: network
        integer(kind=ik), intent(in) :: number_inputs, number_predictions, number_samples
        real(kind=rk), pointer :: samples(:, :), expected(:, :)
        procedure(fnn_cost_function), pointer :: cost_function
        real(kind=rk), intent(out) :: cost

        ! Local var
        real(kind=8), pointer :: yp(:, :), predictions(:), inputs(:)
        integer isample

        ! Nullify local pointers
        nullify (yp, predictions, inputs)

        ! Initialize error
        error = 0

        ! Initialize cost_function
        cost = 0

        ! Check if number of predictions is the same value that number of neurons last layer
        if (network%layers(network%number_layers)%layer%number_neurons /= number_predictions) then
            error = 1
            return
        end if

        ! Check that samples and expected are associated pointers
        if (.not. associated(samples) .or. .not. associated(expected)) then
            error = 2
            return
        end if

        ! Check that samples and expected pointers have the correct size
        if (size(samples, 2) /= number_samples .or. size(samples, 1) /= number_inputs) then
            error = 3
            return
        end if

        if (size(expected, 2) /= number_samples .or. size(expected, 1) /= number_predictions) then
            error = 4
            return
        end if

        ! Compute prediction for each sample
        allocate (yp(number_predictions, number_samples), stat=error)
        if (error /= 0) return

        do isample = 1, number_samples
            predictions => yp(:, isample)
            inputs => samples(:, isample)
            error = activate_network(network, predictions, number_inputs, inputs)
            nullify (predictions, inputs)
            if (error /= 0) return
        end do

        ! Compute the cost
        cost = cost_function(yp, expected, number_predictions, number_samples)

        ! Free memory
        if (associated(yp)) deallocate (yp)

        ! Nullify local pointers
        nullify (yp, predictions, inputs)

    end function cost_network

    subroutine print_network(network)
        type(fnn_network), pointer :: network

        ! Local vars
        integer(kind=ik) ilayer

        write (*, '(A)') "Network: ["
        do ilayer = 1, network%number_layers
            call print_layer(network%layers(ilayer)%layer, 4)
        end do
        write (*, '(A)') "]"
    end subroutine print_network
    !------ End Network procedures ------

    !----- Network cost function -------
    !----- End Network cost function ---------

    !------ Public procedures of the neural network ------
    integer(kind=ik) function fnn_net(number_inputs, number_layers) result(error)
        integer(kind=ik), intent(in) :: number_inputs, number_layers

        ! Initialize error
        error = 0

        ! Allocate memory
        if (alloc) then
            if (associated(net)) error = deallocate_network(net)
            alloc = .false.
            init = .false.
            if (error /= 0) then
                nullify (net)
                return
            end if
        end if
        nullify (net)
        error = allocate_network(net)
        if (error /= 0) then
            nullify (net)
            return
        end if
        alloc = .true.

        ! Initialize network
        error = initialize_network(net, number_inputs, number_layers)
        if (error /= 0) return
        init = .true.

        ! Init layer id
        layer_id = 0

    end function fnn_net

    integer(kind=ik) function fnn_add(number_neurons, activation, derivative_activation) result(error)
        integer(kind=ik), intent(in) :: number_neurons
        procedure(fnn_activation_function), pointer :: activation
        procedure(fnn_derivative_activation_function), pointer :: derivative_activation

        ! Initialize
        error = 0

        ! If not alloc return with error 1
        if (.not. alloc) then
            error = 1
            return
        end if

        ! If not init return with error 2
        if (.not. init) then
            error = 2
            return
        end if

        ! Add the layer
        layer_id = layer_id + 1
        error = add_layer_to_network(net, layer_id, number_neurons, activation, derivative_activation)

    end function fnn_add

    integer(kind=ik) function fnn_predict(prediction_size, prediction, n_inputs, inputs) result(error)
        integer(kind=ik), intent(out) :: prediction_size
        real(kind=rk), pointer :: prediction(:)
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)

        ! Initialize
        error = 0

        ! If not alloc return with error 1
        if (.not. alloc) then
            error = 1
            return
        end if

        ! If not init return with error 2
        if (.not. init) then
            error = 2
            return
        end if

        ! Check that inputs is associated
        if (.not. associated(inputs)) then
            error = 3
            return
        end if

        ! Check that inputs size is equal to n_inputs
        if (size(inputs) /= n_inputs) then
            error = 4
            return
        end if

        ! Check that the n_inputs is the same net number of inputs
        if (n_inputs /= net%number_inputs) then
            error = 5
            return
        end if

        ! Allocate prediction and initialize
        prediction_size = net%layers(net%number_layers)%layer%number_neurons
        nullify (prediction)
        allocate (prediction(prediction_size), stat=error)
        prediction = 0d0

        ! Activate network
        error = activate_network(net, prediction, n_inputs, inputs)

    end function fnn_predict

    integer(kind=ik) function fnn_train(number_inputs, number_predictions, number_samples, epochs, samples, expected, &
                                        learning_rate, epsilon, cost_function) result(error)
        integer(kind=ik), intent(in) :: number_inputs, number_predictions, number_samples, epochs
        real(kind=rk), pointer :: samples(:, :), expected(:, :)
        real(kind=rk), intent(in) :: learning_rate, epsilon
        procedure(fnn_cost_function), pointer :: cost_function

        ! Local vars
        integer(kind=ik) :: ilayer, ineuron, iweight, iepochs
        real(kind=rk) cost, cost_incr, grad

        ! Initialize error
        error = 0
        cost = huge(0d0)
        iepochs = 0
        do while (cost > 1e-4 .and. iepochs < epochs)

            ! Compute the cost
            error = cost_network(net, number_inputs, number_predictions, number_samples, samples, expected, &
                                 cost_function, cost)

            ! Print epochs
            if (mod(iepochs, max(int(epochs / 10, kind=ik),1)) == 0) write (*, *) "Training epoch: ", iepochs, " Cost: ", cost

            ! Loop over layers
            do ilayer = 1, net%number_layers

                ! Loop over each neuron
                do ineuron = 1, net%layers(ilayer)%layer%number_neurons

                    ! Loop over each weight
                    do iweight = 1, net%layers(ilayer)%layer%neurons(ineuron)%neuron%number_inputs

                        ! Update the value of the weight with small epsilon value to compute derivative
                        net%layers(ilayer)%layer%neurons(ineuron)%neuron%weights(iweight) = &
                            net%layers(ilayer)%layer%neurons(ineuron)%neuron%weights(iweight) + epsilon

                        ! Compute the cost with the increment value
                        error = cost_network(net, number_inputs, number_predictions, number_samples, samples, expected, &
                                             cost_function, cost_incr)

                        ! Estimate the gradient with finite differences
                        grad = (cost_incr - cost)/epsilon

                        ! Update the weight with gradient descent
                        net%layers(ilayer)%layer%neurons(ineuron)%neuron%weights(iweight) = &
                            net%layers(ilayer)%layer%neurons(ineuron)%neuron%temp_weights(iweight) - learning_rate*grad

                        ! Update temp_weight
                        net%layers(ilayer)%layer%neurons(ineuron)%neuron%temp_weights(iweight) = &
                            net%layers(ilayer)%layer%neurons(ineuron)%neuron%weights(iweight)
                    end do
                end do
            end do
            iepochs = iepochs + 1
        end do

    end function fnn_train

    subroutine fnn_print()
        if ((.not. alloc) .or. (.not. init)) return
        call print_network(net)
    end subroutine fnn_print

end module FortranNeuralNetwork
