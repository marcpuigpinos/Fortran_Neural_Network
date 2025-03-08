module fnn
    !! fnn module: contains all the Fortran Neural Network procedures.

    use iso_fortran_env, only: error_unit
    
    implicit none

    private

    !  public interfaces
    public :: net_init, net_add_layer, net_compile, net_activate, net_train

    ! Interface procedures
    interface
        real function activation_function(z, z_sum) result(a)
            real, intent(in) :: z
            real, intent(in) :: z_sum
        end function activation_function
    end interface
    ! End interface procedures
    
    ! type layer
    type net_layer
        integer :: n_neurons = 0
        integer :: n_inputs = 0
        real, allocatable, dimension(:,:) :: mat_coeff ! Matrix of coefficients. Size n_inputs x n_neurons.
        real, allocatable, dimension(:) :: z           ! z values. Size n_neurons
        real, allocatable, dimension(:) :: activations ! Activations. Size n_neurons.
        procedure(activation_function), nopass, pointer :: activation_function ! Activation function procedure pointer.
    end type net_layer
    
    ! Net
    integer :: net_n_layers = 0 ! Total number of layers
    integer :: net_n_inputs = 0 ! Total number of inputs of the net (input layer) 
    integer :: net_n_outputs = 0 ! Total number of neurons of the output layer
    type(net_layer), allocatable, target, dimension(:) :: net_layers ! Array with the layers of the network

contains

    !------ Private procedures activation functions ------
    real function relu(z, z_sum) result(a)
        real, intent(in) :: z, z_sum
        a = max(0.0,z)
    end function relu

    real function sigmoid(z, z_sum) result(a)
        real, intent(in) :: z, z_sum
        a = 1.0/(1.0 + exp(-z))
    end function sigmoid
    !------ End Private procedures activation functions
    

    !------ End private procedures activation functions ------

    !------ Private procedures of the layer ------
    integer function init_layer(layer, n_neurons, n_inputs, f_activation_name) result(error)
        type(net_layer), pointer :: layer
        integer, intent(in) :: n_neurons, n_inputs
        character(len=*), intent(in) :: f_activation_name

        ! Initialize constants
        layer%n_neurons = n_neurons
        layer%n_inputs = n_inputs

        ! Initialize matrix of coefficients
        if (allocated(layer%mat_coeff)) allocate(layer%mat_coeff(n_inputs, n_neurons), stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: init_layer: initialization of the layer failed when allocating mat_coeff."
            return
        endif
        call random_number(layer%mat_coeff)

        ! Initialize z array
        if (allocated(layer%z)) allocate(layer%z(n_neurons), stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: init_layer: initialization of the layer failed when allocating z."
            return
        endif
        layer%z = 0d0

        ! Initialize activations
        if (allocated(layer%activations)) allocate(layer%activations(n_neurons), stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: init_layer: initialization of the layer failed when allocating activations."
            return
        endif
        layer%activations = 0d0

        ! Set activation function.
        nullify(layer%activation_function)
        select case( trim(adjustl(f_activation_name)) )
        case("ReLU")
            layer%activation_function => relu
        case("sigmoid")
            layer%activation_function => sigmoid
        end select
        
    end function init_layer

    integer function activate_layer(layer) result(error)
        type(net_layer), pointer :: layer
        
    end function activate_layer
    !------ End Private procedures of the layer ------

    !------ Private procedures of the network ------

    !------ End Private procedures of the network ------

    !------ Public procedures of the neural network ------
    integer function net_init(number_inputs) result(error)
    !! Initializer of the network
        integer, intent(in) :: number_inputs 
        !! Total number inputs of the net (input layer).

        ! Initialize error
        error = 0
        
        ! Initialize network varaibles
        net_n_layers = 0
        net_n_inputs = 0
        net_n_outputs = 0

        ! Print
        write(*,'(A)') "Network has been initialized: "
        write(*, '(4x,A,i10)') "- net_n_layers: ", net_n_layers
        write(*, '(4x,A,i10)') "- net_n_inputs: ", net_n_inputs
        write(*, '(4x,A,i10)') "- net_n_outputs: ", net_n_outputs
        
    end function net_init

    integer function net_add_layer(number_neurons, activation) result(error)
    !! Adds a layer to the network.
        integer, intent(in) :: number_neurons 
        !! Number of neurons of the added layer.
        character(len=*), intent(in) :: activation
        !! Activation function name: ReLU, sigmoid, softmax

        ! Initialize error
        error = 0

        ! Update number of layers
        net_n_layers = net_n_layers + 1
        net_n_outputs = number_neurons

        ! Print
        write(*,'(A)') "Layer has been added: "
        write(*,'(4x,A,i10)') "- net_n_layers: ", net_n_layers
        write(*,'(4x,A,i10)') "- net_n_outputs: ", net_n_outputs
        write(*,'(4x,A,i10)') "- number_neurons: ", number_neurons
        write(*,'(4x,A)') "- activation: "//trim(adjustl(activation))

    end function net_add_layer

    integer function net_compile(loss, epochs, batches, optimizer, learning_rate) result(error)
    !! Compile the network architecture.
        character(len=*), intent(in) :: loss
        !! Name of the cost function.
        integer, intent(in) ::  epochs
        !! Number of epochs the training will be executed.
        integer, intent(in), optional :: batches
        !! Number of batches used during the training stage.
        character(len=*), intent(in), optional :: optimizer
        !! Optimizer used. By default GD.       
        real, intent(in), optional :: learning_rate
        !! Learning rate used for some algorithms.

        ! Initialize error
        error = 0

        ! Print
        write(*,'(A)') "Network compiled ..."
        write(*,'(4x,A)') "- loss: "//trim(adjustl(loss))
        write(*,'(4x,A,i10)') "- epochs: ", epochs
        if (present(batches)) write(*,'(4x,A,i10)') "- batches: ", batches
        if (present(optimizer)) write(*,'(4x,A)') "- optimizer: "//trim(adjustl(optimizer))
        if (present(learning_rate)) write(*,'(4x,A,E15.7)') "- learning_rate: ", learning_rate

    end function net_compile

    integer function net_activate(activations, inputs, n_outputs, n_inputs) result(error)
    !! Given an input array, activates the network.
        real, dimension(n_outputs), intent(out) :: activations
        !! Output array containing the values predicted by the network for each output layer neuron.
        real, dimension(n_inputs), intent(in) :: inputs
        !! Input array containing the values of the input layer.
        integer, intent(in) :: n_outputs
        !! Size of the activations array. Must be the same of net_n_outputs (output layer)
        integer, intent(in) :: n_inputs
        !! Size of the inputs array. Must be the same of the net_n_inputs (input layer)

        ! Initialize error
        error = 0

        ! Print
        print *, "net_activate"
        
    end function net_activate

    integer function net_train(x_train, y_train, n_samples, n_inputs, n_outputs) result(error)
        !! Computes the training of the network given a training dataset
        !! First dimension size of the array y_train. Must be the same of the net_n_outputs (output layer).       
        real, dimension(n_inputs,n_samples), intent(in) :: x_train
        !! Array with the input values of the dataset
        real, dimension(n_outputs,n_samples), intent(in) :: y_train
        !! Array with the expected values of the dataset
        integer, intent(in) :: n_samples
        !! Second dimension size of the arrays x_train and y_train. Correspond to the number of samples of the training set.
        integer, intent(in) :: n_inputs
        !! First dimension size of the array x_train. Must be the same of the net_n_inputs (input layer).
        integer, intent(in) :: n_outputs

        ! Initialize error
        error = 0
        
        ! Print
        print *, "net_train"

    end function net_train

end module fnn
