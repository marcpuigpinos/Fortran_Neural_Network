module fnn
    !! fnn module: contains all the Fortran Neural Network procedures.

   use iso_fortran_env, only: error_unit

   implicit none

   private

   ! public types
   public :: fnn_error

   !  public interfaces
   public :: fnn_initialize_network, &
             fnn_add_layer

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
                  na, & ! number of activations
                  nn    ! number of neurons

        real(kind=8), allocatable, dimension(:) :: activations ! Vector of activations
        real(kind=8), allocatable, dimension(:) :: outputs ! Vector of outputs

   end type fnn_layer
   
   ! Global network parameters
   integer nl, & ! Number of layers
           ll, & ! Layers array length
           tnn   ! total number of neurons
   
   type(fnn_layer), allocatable, target :: layers(:) ! Array containing the layers of the network. 
   
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
    
    type(fnn_error) function initialize_layer(layer, id, na, nn, activations) result(error)
        type(fnn_layer), pointer :: layer
        integer, intent(in) :: id, & ! Layer id
                               na, & ! Number of activations of the layer
                               nn ! Number of neurons of the layer
        real(kind=8), intent(in) :: activations(na)

        ! Initialize error
        call default_error(error)

        ! Initialize values
        layer%id = id
        layer%na = na
        layer%nn = nn

        ! Initialize vector of activations
        if (allocated(layer%activations)) deallocate(layer%activations, stat=error%code, errmsg=error%msg)
        if (error%code /= 0) call exit_proc() ! Any error exit
        allocate(layer%activations(na+1), stat=error%code, errmsg=error%msg)
        if (error%code /= 0) call exit_proc() ! Any error exit
        
        layer%activations(1) = 1d0
        layer%activations(2:na+1) = activations

        ! Initialize vector of outputs
        if (allocated(layer%outputs)) deallocate(layer%outputs, stat=error%code, errmsg=error%msg)
        if (error%code /= 0) call exit_proc() ! Any error exit
        allocate(layer%outputs(nn), stat=error%code, errmsg=error%msg)
        if (error%code /= 0) call exit_proc() ! Any error exit
        layer%outputs = 0d0
        
    contains

        subroutine exit_proc()
            return
        end subroutine exit_proc

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
    
    type(fnn_error) function fnn_add_layer(nn, na, activations) result(error)
        integer, intent(in) :: nn ! Number of neurons of the layer
        integer, optional, intent(in) :: na ! Number of activatoins. Optional: only needed when is first layer
        real(kind=8), optional, intent(in) :: activations(:) ! Array of activations. Optional: only needed when is first layer.
        type(fnn_layer), pointer :: layer, prev_layer

        ! Initialize error
        call default_error(error)

        ! Nullify local pointers
        call nullify_local_pointers()

        ! Increase the number of layers by one
        nl = nl + 1

        ! If layer array length is equal to numbers of layers, increase the layers array.
        if (nl > ll) then
            error = layer_arr_inc()
            if (error%code /= 0) then
                call exit_proc() ! Exit the procedure
            endif
        endif

        ! Increase the total number of neurons, by adding the number of neurons of this layer
        tnn = tnn + nn

        ! Check if we are first layer, and we have number of activations na argument given
        ! Check if we are first layer, and we have activations array.
        if (nl == 1 .and. (.not. present(na))) then
            error%code = 1
            error%msg = "fnn_add_layer: number of activations not given on first layer."
            call exit_proc()

        else if (nl > 1 .and. present(na)) then
            error%code = 2
            error%msg = "fnn_add_layer: number of activations is given in an intermidate layer."
            call exit_proc()

        end if        

        ! Check if we are first layer, and we have activations array.
        if (nl == 1 .and. (.not. present(activations))) then
            error%code = 3
            error%msg = "fnn_add_layer: activations array missing on first layer."
            call exit_proc()

        else if (nl > 1 .and. present(activations)) then
            error%code = 4
            error%msg = "fnn_add_layer: activations array is given in an intermidate layer."
            call exit_proc()

        end if

        ! Check if activations array have the length.
        if (nl==1) then
            if (size(activations) /= na) then
                error%code = 5
                error%msg = "fnn_add_layer: activations array length is different that number of activations"
            endif
        endif

        ! Reserve the memory for the layer
        if (nl == 1) then
            layer => layers(nl)
            error = initialize_layer(layer, nl, na, nn, activations)
            nullify(layer, prev_layer)
        else
            layer => layers(nl)
            prev_layer => layers(nl-1)
            error = initialize_layer(layer, nl, prev_layer%nn, nn, prev_layer%outputs)
            nullify(layer, prev_layer)
        endif

        call exit_proc() ! Exit the procedure

    contains

        subroutine nullify_local_pointers()
            nullify(layer, prev_layer)
        end subroutine

        subroutine exit_proc()
            call nullify_local_pointers()
            return
        end subroutine

    end function fnn_add_layer
        
   
end module fnn
