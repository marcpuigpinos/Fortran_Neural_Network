program main

   use fnn

   implicit none

   type(fnn_error) :: error

   real(kind=8), dimension(2) :: input, output
   
   error = fnn_initialize_network()
   if (error%code /= 0) then
       print *, "Error initializing network: ", error%msg
       stop
   end if

   error = fnn_add_layer(2, 2, FNN_RELU)
   if (error%code /= 0) then
       print *, "Error adding layer: ", error%msg
       stop
   end if

   input = (/ 1d0, 2d0 /)
   
   error = fnn_inference(input, output)
   if (error%code /= 0) then
       print *, "Error during inference: ", error%msg
       stop
   end if

   print *, "Inference output: ", output

   
end program main
