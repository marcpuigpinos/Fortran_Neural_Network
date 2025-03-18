module fnn
    !! fnn module: contains all the Fortran Neural Network procedures.

    use iso_fortran_env, only: error_unit
    
    implicit none

    private

    !  public interfaces
    public :: net_init, net_add_layer, net_activate, net_train, net_print

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
        !! Type that stores the information of a layer.
        integer :: n_neurons = 0
        !! Number of neurons of a layer.
        integer :: n_inputs = 0
        !! Number of inputs of the layer. All inputs perform on all layer neurons.
        real, allocatable, dimension(:,:) :: mat_coeff
        !! Matrix storing weights and bias for each neuron. Size n_inputs x n_neurons
        real, allocatable, dimension(:) :: z
        !! Array storing the dot product beteween weights and inputs: $$z = \mathbf{w}^T \cdot \mathbf{x}$$. Size n_neurons.
        real, allocatable, dimension(:) :: activations
        !! Array storing the activations of the neurons. Size n_neurons.
        procedure(activation_function), nopass, pointer :: activation_function
        !! Activation function of the neurons for this specific layer.
    end type net_layer
    
    interface assignment(=)
        module procedure layer_assignment
    end interface assignment(=)
    
    ! Net
    integer :: net_n_layers = 0 ! Total number of layers
    integer :: net_n_inputs = 0 ! Total number of inputs of the net (input layer) 
    integer :: net_n_outputs = 0 ! Total number of neurons of the output layer
    integer :: net_layer_id = 0 ! Layer id counter for add layers procedure.
    type(net_layer), allocatable, target, dimension(:) :: net_layers ! Array with the layers of the network

contains

    !------ Private Generic procedures -------
    integer function matrix_vector_product(matrix, vector_inp, vector_out) result(error)
        !! Matrix vector product
        real, dimension(:,:), intent(in) :: matrix
        !! Input matrix. Size m x n.
        real, dimension(:), intent(in) :: vector_inp
        !! Input vector. Size m.
        real, dimension(:), intent(inout) :: vector_out
        !! Output vector. Size n.
        integer i, j, n_mat_1, n_mat_2, n_inp, n_out

        ! Initialize error
        error = 0

        n_mat_1 = size(matrix,1)
        n_mat_2 = size(matrix,2)
        n_inp = size(vector_inp)
        n_out = size(vector_out)

        ! Check sizes
        if ( n_mat_1 /= n_inp ) error = 1
        if ( n_mat_2 /= n_out ) error = 2

        ! Check error
        if ( error /= 0 ) then
            write(error_unit,*) "Error: matrix_vector_product, arrays shape does not conform."
            return
        endif

        ! compute product
        do i = 1, n_out
           vector_out(i) = 0d0
           do j = 1, n_inp
              vector_out(i) = vector_out(i) + matrix(j,i) * vector_inp(j)
           enddo
        enddo
        
    end function matrix_vector_product
    
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
    subroutine layer_assignment(layer_out, layer_in)
        ! Assign layer procedure
        type(net_layer), intent(out) :: layer_out
        type(net_layer), intent(in) :: layer_in

        ! Layer assignment constants
        layer_out%n_neurons = layer_in%n_neurons
        layer_out%n_inputs = layer_in%n_inputs

        ! Layer assignment Arrays
        ! mat_coeff
        if (allocated(layer_in%mat_coeff)) then
            if (allocated(layer_out%mat_coeff)) deallocate(layer_out%mat_coeff)
            allocate(layer_out%mat_coeff(layer_in%n_inputs, layer_in%n_neurons))
            layer_out%mat_coeff = layer_in%mat_coeff
        endif

        ! z
        if (allocated(layer_in%z)) then
            if (allocated(layer_out%z)) deallocate(layer_out%z)
            allocate(layer_out%z(layer_in%n_neurons))
            layer_out%z = layer_in%z
        endif

        ! activations
        if (allocated(layer_in%activations)) then
            if (allocated(layer_out%activations)) deallocate(layer_out%activations)
            allocate(layer_out%activations(layer_in%n_neurons))
            layer_out%activations = layer_in%activations
        endif        

        ! pointers assignment
        layer_out%activation_function => layer_in%activation_function
        
    end subroutine layer_assignment
    
    integer function init_layer(layer, n_neurons, n_inputs, f_activation_name) result(error)
        ! Initialize layer procedure
        type(net_layer), intent(inout) :: layer
        integer, intent(in) :: n_neurons, n_inputs
        character(len=*), intent(in) :: f_activation_name

        ! Initialize error
        error = 0

        ! Initialize constants
        layer%n_neurons = n_neurons
        layer%n_inputs = n_inputs

        ! Initialize matrix of coefficients
        if (.not. allocated(layer%mat_coeff)) allocate(layer%mat_coeff(n_inputs, n_neurons), stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: init_layer: initialization of the layer failed when allocating mat_coeff."
            return
        endif
        call random_number(layer%mat_coeff)

        ! Initialize z array
        if (.not. allocated(layer%z)) allocate(layer%z(n_neurons), stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: init_layer: initialization of the layer failed when allocating z."
            return
        endif
        layer%z = 0d0

        ! Initialize activations
        if (.not. allocated(layer%activations)) allocate(layer%activations(n_neurons), stat=error)
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
        case default
            error = 1
            write(error_unit,*) "Error: init_layer: initialization of the layer failed when selecting activation function"
            return
        end select
        
    end function init_layer

    integer function free_layer(layer) result(error)
        ! Free layer memory
        type(net_layer), intent(inout) :: layer

        ! Initialize error
        error = 0

        ! Initialize constants
        layer%n_neurons = 0
        layer%n_inputs = 0

        ! free matrix of coefficients
        if (allocated(layer%mat_coeff)) deallocate(layer%mat_coeff, stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: free_layer: free memory of the layer failed when deallocating mat_coeff."
            return
        endif

        ! free matrix z array
        if (allocated(layer%z)) deallocate(layer%z, stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: free_layer: free memory of the layer failed when deallocating z."
            return
        endif
        
        if (allocated(layer%activations)) deallocate(layer%activations, stat=error)
        if (error /= 0) then
            write(error_unit, *) "Error: free_layer: free memory of the layer failed when deallocating activations."
            return
        endif

        ! Nullify cost function
        nullify(layer%activation_function)
        
    end function free_layer

    integer function activate_layer(layer, inputs) result(error)
        ! Compute the activations of the layer
        type(net_layer), intent(inout) :: layer
        real, intent(in), dimension(:) :: inputs
        integer i

        ! Initialize error
        error = 0

        ! Check array size
        if ( size(inputs) /= layer%n_inputs ) then
            error = 1
            write(error_unit,*) "Error: activate_layer: inputs array size does not corresponds to the inputs array of the layer"
            return
        endif

        ! compute z
        error = matrix_vector_product(layer%mat_coeff, inputs, layer%z)

        ! compute activations
        do i = 1, layer%n_neurons
           layer%activations(i) = layer%activation_function(layer%z(i), sum(layer%z)) 
        enddo
        
    end function activate_layer
    !------ End Private procedures of the layer ------

    !------ Private procedures of the network ------
 
    !------ End Private procedures of the network ------

    !------ Public procedures of the neural network ------
    integer function net_init(number_inputs, number_outputs, number_layers) result(error)
    !! Initializer of the network
        integer, intent(in) :: number_inputs 
        !! Total number inputs of the net (input layer).
        integer, intent(in) :: number_outputs
        !! Total number of outputs of the net (output layer).
        integer, intent(in) :: number_layers
        !! Total number of layers of the net.

        ! Initialize error
        error = 0
        
        ! Initialize network varaibles
        net_n_inputs = number_inputs
        net_n_outputs = number_outputs
        net_n_layers = number_layers
        net_layer_id = 0

        ! Allocate array net_layers
        if ( .not. allocated(net_layers) ) allocate(net_layers(number_layers))

    end function net_init

    integer function net_add_layer(number_neurons, f_activation_name) result(error)
    !! Adds a layer to the network.
        integer, intent(in) :: number_neurons 
        !! Number of neurons of the added layer.
        character(len=*), intent(in) :: f_activation_name
        !! Activation function name: ReLU, sigmoid, softmax
        type(net_layer) :: layer

        ! Initialize error
        error = 0

        ! Update net_layer_id
        net_layer_id = net_layer_id + 1

        ! Initialize layer
        if (net_layer_id == 1 ) then
            error = init_layer(layer, number_neurons, net_n_inputs, f_activation_name)
        else
            error = init_layer(layer, number_neurons, net_layers(net_layer_id-1)%n_neurons, f_activation_name)
        endif
        if ( error /= 0 ) then
            write(error_unit, *) "Error: net_add_layer: error in layer initialization."
            return
        endif
            
        ! Add layer to the array of net_layers.
        net_layers(net_layer_id) = layer

        ! Free layer memory
        error = free_layer(layer)
        if ( error /= 0 ) then
            write(error_unit, *) "Error: net_add_layer: error when free memory layer."
            return
        endif        

    end function net_add_layer

    integer function net_activate(activations, inputs, n_outputs, n_inputs) result(error)
    !! Given an input array, activates the network.
        real, dimension(n_outputs), intent(out) :: activations
        !! Output array containing the values predicted by the network for each output layer neuron.
        real, dimension(n_inputs), target, intent(in) :: inputs
        !! Input array containing the values of the input layer.
        integer, intent(in) :: n_outputs
        !! Size of the activations array. Must be the same of net_n_outputs (output layer)
        integer, intent(in) :: n_inputs
        !! Size of the inputs array. Must be the same of the net_n_inputs (input layer)

        ! Local vars
        integer ilayer
        
        ! Initialize error
        error = 0

        ! Loop over layers to activate the network.
        error = activate_layer(net_layers(1), inputs)
        do ilayer = 2, net_n_layers
           error = activate_layer(net_layers(ilayer), net_layers(ilayer-1)%activations)
        enddo
        
    end function net_activate

    integer function net_train(x_train, y_train, n_samples, n_inputs, n_outputs, optimizer, loss, epochs, batches, learning_rate) result(error)
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
        !! First dimension of the array y_train.
        character(len=*), intent(in) :: optimizer
        !! Method used to optimize the loss function: gradient descent. 
        character(len=*), intent(in) :: loss
        !! Name of the loss/cost function: mean squared error.
        integer, intent(in) :: epochs
        !! Number of epochs of the training.
        integer, intent(in) :: batches
        !! Number of batches we want to divide the samples for training.
        real, intent(in) :: learning_rate
        !! Value of the learning rate.

        ! Initialize error
        error = 0

        ! Select the optimizer
        select case(trim(adjustl(optimizer)))
        case("gradient_descent")
            ! Select the loss
            select case(trim(adjustl(loss)))
            case("mean_squared_error")
                ! Call the train_gradient_descent_mean_squared_error procedure
            case default
                !Print error
            end select
        case default
            ! Print error
        end select
        
        ! Print
        print *, "net_train"

    end function net_train

    integer function net_print() result(error)
        integer ilayer
        error = 0

        do ilayer=1, net_n_layers
           write(*,'(A,i5)') "Layer ", ilayer
           write(*,'(4x,A,i5)') "n_neurons: ", net_layers(ilayer)%n_neurons
           write(*,'(4x,A,i5)') "n_inputs: ", net_layers(ilayer)%n_inputs
        enddo
    end function net_print
    
    
end module fnn
