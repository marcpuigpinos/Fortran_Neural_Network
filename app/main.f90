program main

    use fnn

    implicit none

    integer :: error, number_inputs, number_layers, number_outputs
    real, allocatable, dimension(:) :: predictions, inputs

    
    write(*,*) "Hello World!"

    number_inputs = 3
    number_outputs = 1
    number_layers = 1
    allocate(predictions(number_outputs))
    allocate(inputs(number_inputs))
    inputs(1) = 1.0; inputs(2) = 2.0; inputs(3) = 0.5
    
    error = net_init(number_inputs, number_outputs, number_layers)
    error = net_add_layer(1, "sigmoid")
    error = net_activate(predictions, inputs, number_outputs, number_inputs)
    error = net_print()
    
end program main
