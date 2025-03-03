module fnn

    use FortranNeuralNetwork

    public
    
contains

    integer function network(number_inputs, number_layers) result(error)
        integer, intent(in) :: number_inputs, number_layers
        error = fnn_net(number_inputs, number_layers)
    end function network

    integer function add_layer(number_neurons, activation) result(error)
        integer, intent(in) :: number_neurons
        character(len=*), intent(in) :: activation
        procedure(fnn_activation_function), pointer :: f_activation
        procedure(fnn_derivative_activation_function), pointer :: fd_activation

        nullify(f_activation, fd_activation)
        
        select case(trim(adjustl(activation)))
        case("ReLU")
            f_activation => fnn_ReLU
            fd_activation => fnn_derivative_ReLU
        case("sigmoid")
            f_activation => fnn_sigmoid
            fd_activation => fnn_derivative_sigmoid
        case default
            error = 1
            return
        end select
        
        error = fnn_add(number_neurons, f_activation, fd_activation)

        nullify(f_activation, fd_activation)
        
    end function add_layer

    integer function predict_network(number_outputs, prediction, number_inputs, inputs) result(error)
        integer, intent(inout) :: number_outputs
        integer, intent(in) :: number_inputs
        real(kind=8), intent(in), dimension(number_inputs) :: inputs
        real(kind=8), intent(inout), target, dimension(number_outputs) :: prediction
        real(kind=8), pointer :: p_prediction(:), p_inputs(:)

        nullify(p_prediction, p_inputs)

        p_prediction => prediction
        error = fnn_predict(number_outputs, p_prediction, number_inputs, p_inputs)

        nullify(p_prediction, p_inputs)
        
    end function predict_network

    integer function train_network(number_inputs, number_outputs, number_samples, epochs, samples_input, samples_output, learning_rate, epsilon, cost_function) result(error)
        integer, intent(in) :: number_inputs, number_outputs, number_samples, epochs
        real(kind=8), intent(in), dimension(number_inputs, number_samples) :: samples_input
        real(kind=8), intent(in), dimension(number_outputs, number_samples) :: samples_output
        real(kind=8), intent(in) :: learning_rate, epsilon
        character(len=*), intent(in) :: cost_function

        real(kind=8), pointer :: p_samples_input(:,:)
        real(kind=8), pointer :: p_samples_output(:,:)
        procedure(fnn_cost_function), pointer :: f_cost_function
        
        nullify(f_cost_function, p_samples_input, p_samples_output)
        
        select case(trim(adjustl(cost_function)))
        case("MSE")
            f_cost_function => fnn_cost_MSE
        case("CrossEntropy")
            f_cost_function => fnn_cost_cross_entropy
        case default
            error = 1
            return
        end select

        error = fnn_train(number_inputs, number_outputs, number_samples, epochs, p_samples_input, p_samples_output, learning_rate, epsilon, f_cost_function)
        
        nullify(f_cost_function, p_samples_input, p_samples_output)
        
    end function train_network
    
!    subroutine print_network()
!        call fnn_print()
!    end subroutine print_network
    
end module fnn
