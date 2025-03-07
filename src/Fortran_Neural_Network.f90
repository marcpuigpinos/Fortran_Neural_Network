module FortranNeuralNetwork

    implicit none

    private

    !  public interfaces
    public :: fnn_net, fnn_add, fnn_predict, fnn_train, fnn_print


    integer, parameter :: ik = 4
    integer, parameter :: rk = 8
    
contains

    !------ Public procedures of the neural network ------
    integer(kind=ik) function fnn_net(number_inputs, number_layers) result(error)
        integer(kind=ik), intent(in) :: number_inputs, number_layers

        ! Initialize error
        error = 0

        ! Print
        print *, number_inputs, number_layers
        
    end function fnn_net

    integer(kind=ik) function fnn_add(number_neurons, activation) result(error)
        integer(kind=ik), intent(in) :: number_neurons
        character(len=*), intent(in) :: activation

        ! Initialize
        error = 0

    end function fnn_add

    integer(kind=ik) function fnn_predict(n_prediction, prediction, n_inputs, inputs) result(error)
        integer(kind=ik), intent(out) :: n_prediction
        real(kind=rk), pointer :: prediction(:)
        integer(kind=ik), intent(in) :: n_inputs
        real(kind=rk), pointer :: inputs(:)

        ! Initialize
        error = 0
        
    end function fnn_predict

    integer(kind=ik) function fnn_train(number_inputs, number_predictions, number_samples, epochs, samples, expected, &
                                        learning_rate, epsilon, cost_function) result(error)
        integer(kind=ik), intent(in) :: number_inputs, number_predictions, number_samples, epochs
        real(kind=rk), pointer :: samples(:, :), expected(:, :)
        real(kind=rk), intent(in) :: learning_rate, epsilon
        character(len=*), intent(in) :: cost_function

        ! Initialize
        error = 0
        
    end function fnn_train

    subroutine fnn_print()
    end subroutine fnn_print

end module FortranNeuralNetwork
