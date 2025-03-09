program main

    use fnn

    implicit none

    integer :: error, number_inputs, number_layers, number_outputs
    
    write(*,*) "Hello World!"

    number_inputs = 3
    number_outputs = 1
    number_layers = 1
    
    error = net_init(number_inputs, number_outputs, number_layers)
    error = net_add_layer(1, "sigmoid")
    error = net_print()
    
end program main
