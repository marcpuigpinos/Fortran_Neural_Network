module fnn
    !! fnn module: contains all the Fortran Neural Network procedures.

   use iso_fortran_env, only: error_unit

   implicit none

   private

   !  public interfaces
   !public

   ! Constants
   integer, parameter :: FNN_MSG_BUFF = 256
   
   ! Definition of error type
   type fnn_error
       integer :: code ! Code of the error
       character(len=FNN_MSG_BUFF) :: msg ! Message of the error
   end type fnn_error
   

   ! Definition of the layer type
   type layer
       integer :: id, & ! Id of the layer
                  na, &  ! number of activations
                  nn     ! number of neurons
   end type layer

   ! Global network parameters
   integer nl, & ! Number of layers
           tnn   ! total number of neurons
   
contains
    
   
end module fnn
