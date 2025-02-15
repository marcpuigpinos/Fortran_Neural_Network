module NeuralNetwork

    use iso_c_binding
    use FortranNeuralNetwork, only: fnn_net, fnn_add, fnn_train, fnn_predict, fnn_print, &
                                    fnn_activation_function, fnn_derivative_activation_function, &
                                    fnn_ReLU, fnn_sigmoid, fnn_derivative_ReLU, fnn_derivative_sigmoid, &
                                    fnn_cost_function, fnn_cost_MSE

    implicit none

    private

    public :: create_net, add_layer_to_net, train_network, print_network

contains

    subroutine c_to_fortran_string(c_string, fortran_string)
        character(kind=c_char), dimension(*) :: c_string  ! C-style string
        character(:), allocatable :: fortran_string  ! Allocatable Fortran string
        integer :: i
        integer :: len_c_string

        ! Determine the length of the C string (up to the null terminator)
        len_c_string = 0
        do while (c_string(len_c_string + 1) /= c_null_char)
            len_c_string = len_c_string + 1
        end do

        ! Allocate the Fortran string with the correct length
        allocate (character(len=len_c_string) :: fortran_string)

        ! Copy the characters from the C string to the Fortran string
        do i = 1, len_c_string
            fortran_string(i) = c_string(i:i)  ! Assign individual characters
        end do

    end subroutine c_to_fortran_string

    integer(kind=c_int) function create_net(n_inputs, n_layers) result(error) bind(c, name="create_net")
        integer(kind=c_int), intent(in) :: n_inputs, n_layers
        error = fnn_net(n_inputs, n_layers)
    end function

    integer(kind=c_int) function add_layer_to_net(n_neurons, activation_name) result(error) bind(c, name="add_layer_to_net")
        integer(kind=c_int), intent(in) :: n_neurons
        character(kind=c_char), dimension(*) :: activation_name

        ! Local vars
        procedure(fnn_activation_function), pointer :: activation
        procedure(fnn_derivative_activation_function), pointer :: derivative_activation
        character(len=:), allocatable :: fortran_activation_name

        ! Nullify local pointers
        nullify (activation, derivative_activation)

        ! Convert c string to fortran string
        call c_to_fortran_string(activation_name, fortran_activation_name)

        ! Select activation
        select case (trim(adjustl(fortran_activation_name)))
        case ("ReLU")
            activation => fnn_ReLU
            derivative_activation => fnn_derivative_ReLU
        case ("sigmoid")
            activation => fnn_sigmoid
            derivative_activation => fnn_derivative_sigmoid
        end select

        ! Call the add layer
        error = fnn_add(n_neurons, activation, derivative_activation)

        ! Nullify local pointers
        nullify (activation, derivative_activation)

    end function

    integer(kind=c_int) function train_network(n_inputs, n_outputs, n_samples, &
                                               sample_inputs, sample_outputs, learning_rate, &
                                               epsilon, cost_type_name) result(error) bind(c, name="train_network")

        integer(kind=c_int), intent(in) :: n_inputs, n_outputs, n_samples
        real(kind=c_double), pointer :: sample_inputs(:, :), sample_outputs(:, :)
        real(kind=c_double), intent(in) :: learning_rate, epsilon
        character(kind=c_char), dimension(*) :: cost_type_name

        ! Local vars
        procedure(fnn_cost_function), pointer :: cost_function
        character(len=:), allocatable :: fortran_cost_type_name

        ! Nullify local pointers
        nullify (cost_function)

        ! Convert c_string to fortran
        call c_to_fortran_string(cost_type_name, fortran_cost_type_name)

        ! Select cost function type:
        select case (trim(adjustl(fortran_cost_type_name)))
        case ("MSE")
            cost_function => fnn_cost_MSE
        end select

        ! Train
        error = fnn_train(n_inputs, n_outputs, n_samples, sample_inputs, sample_outputs, &
                          learning_rate, epsilon, cost_function)

        ! Nullify local pointers
        nullify (cost_function)

    end function

    integer(kind=c_int) function print_network() result(error) bind(c, name="print_neetwork")
        error = 0
        call fnn_print()
    end function

end module
