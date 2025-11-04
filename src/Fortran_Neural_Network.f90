module fnn
    !! fnn module: contains all the Fortran Neural Network procedures.

   use iso_fortran_env, only: error_unit

   implicit none

   private

   !  public interfaces
   !public

   ! Constants
   integer, parameter :: FNN_MSG_BUFF = 256
   integer, parameter :: FNN_ARRAY_INC = 128
   
   ! Definition of error type
   type fnn_error
       integer :: code ! Code of the error
       character(len=FNN_MSG_BUFF) :: msg ! Message of the error
   end type fnn_error
   

   ! Definition of the layer type
   type fnn_layer
       integer :: id, & ! Id of the layer
           na, &  ! number of activations
           nn     ! number of neurons
   end type fnn_layer
   
   ! Global network parameters
   integer nl, & ! Number of layers
       ll, & ! Layers array length
       tnn   ! total number of neurons
   
   type(fnn_layer), allocatable :: layers(:) ! Array containing the layers of the network. 
   
contains

    ! Error procedures
    
    subroutine default_error(error)
        type(fnn_error), intent(inout) :: error
        error%code = 0
        error%msg = ""
    end subroutine default_error


    
    ! Layer procedures
    
    type(fnn_error) function layer_arr_inc() result(error)
        type(fnn_layer), allocatable :: aux(:)
        integer size_aux
        
        ! Initialize error
        call default_error(error)

        ! Move allocatable: layers array will be deallocted and aux will contains information of layers
        call move_alloc(layers, aux)
        !if (error%code /= 0) call exit_proc() ! If any error, return.
        
        ! Reserve memory
        size_aux = size(aux)
        ll = size_aux + FNN_ARRAY_INC ! Update layers length
        allocate(layers(ll), stat=error%code, errmsg=error%msg)
        if (error%code /= 0) call exit_proc() ! If any error, return.

        ! Save aux to layers array
        layers(1:size_aux) = aux

        ! Exit
        call exit_proc()
        
    contains

        subroutine exit_proc()
            if (allocated(aux)) deallocate(aux)
            return
        end subroutine exit_proc
        
    end function layer_arr_inc
    
    type(fnn_error) function initialize_layer(layer, id, na, nn) result(error)
        type(fnn_layer), pointer :: layer
        integer, intent(in) :: id, & ! Layer id
            na, & ! Number of activations of the layer
            nn ! Number of neurons of the layer

        ! Initialize error
        call default_error(error)

        ! Initialize values
        layer%id = id
        layer%na = na
        layer%nn = nn
        
    end function initialize_layer


    
    ! Definition of the public interfaces
    
    type(fnn_error) function fnn_initialize_network() result(error)

        ! Initialize error
        call default_error(error)

        ! Initialize network parameters
        nl = 0
        tnn = 0
        ll = FNN_ARRAY_INC
        if (allocated(layers)) deallocate(layers, stat=error%code, errmsg=error%msg)
        if (error%code /= 0) return ! If any error return
        allocate(layers(ll), stat=error%code, errmsg=error%msg)
        if (error%code /= 0) return ! If any error return
        
        
    end function fnn_initialize_network    
        
   
end module fnn
