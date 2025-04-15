program mnist

   use FortranNeuralNetwork

   implicit none

   integer, parameter :: n_train_samples = 60000
   integer, parameter :: n_pixels = 784 ! 28 x 28
   real(kind=8), allocatable, target :: samples(:, :)
   real(kind=8), pointer :: samples_input(:, :)
   real(kind=8), pointer :: samples_output(:, :)
   real number
   integer i, j, n, n_layers, n_inputs, n_outputs, epochs
   real(kind=8) learning_rate, epsilon
   procedure(fnn_activation_function), pointer :: activation
   procedure(fnn_derivative_activation_function), pointer :: dactivation
   procedure(fnn_cost_function), pointer :: cost_function
   character(len=2*(n_pixels + 1)) :: line
   character(len=1) :: chnum
   integer :: ios, error

   nullify (samples_input, samples_output, activation, dactivation, cost_function)

   open (unit=50, file="test/MNIST/mnist_train.csv")

   ! Define dynamic format
   allocate (samples(n_pixels + 1, n_train_samples))

   ! Read each line
   do i = 1, n_train_samples
      read (50, '(A)', iostat=ios) line
      if (ios /= 0) then
         print *, "Error reading line ", i
         stop
      end if

      ! Read each character of the line
      n = 0
      do j = 1, len(trim(adjustl(line)))
         chnum = line(j:j)
         if (chnum == ",") cycle
         if (chnum == "\n") exit
         if (chnum == " ") exit
         if (chnum == "") exit
         n = n + 1
         if (n > n_pixels + 1) exit
         call char_to_real(chnum, number)
         samples(n, i) = number
      end do
   end do

   close (50)

   ! Define neural network
   allocate (samples_output(10, 1000))
   samples_output = 0d0
   do i = 1, 1000
      samples_output(int(samples(1, i)) + 1, i) = 1d0
   end do
   allocate (samples_input(n_pixels + 1, 1000))
   samples_input(1, :) = 1d0
   samples_input(2:n_pixels + 1, :) = samples(2:n_pixels + 1, 1:1000)
   deallocate (samples)
   n_inputs = n_pixels + 1
   n_layers = 2
   n_outputs = 10
   epochs = 1000
   learning_rate = 0.5
   epsilon = 0.5

   print *, "Creating the network"

   error = fnn_net(n_inputs, n_layers)
   activation => fnn_ReLU
   dactivation => fnn_derivative_ReLU
   error = fnn_add(800, activation, dactivation)
   activation => fnn_sigmoid
   dactivation => fnn_derivative_sigmoid
   error = fnn_add(n_outputs, activation, dactivation)

   print *, "Training the network"

   cost_function => fnn_cost_MSE
   error = fnn_train(n_inputs, n_outputs, 1000, epochs, samples_input, samples_output, &
                     learning_rate, epsilon, cost_function)

   print *, "Finish the training with error: ", error

   deallocate (samples_input)
   deallocate (samples_output)
   nullify (samples_input, samples_output, activation, dactivation, cost_function)

contains

   subroutine char_to_real(character_num, real_num)
      !character to real number
      character(len=*), intent(in)::character_num
      real, intent(out) :: real_num
      integer :: istat

      read (character_num, *, iostat=istat) real_num
      if (istat /= 0) then
         real_num = 0
      end if
   end subroutine

end program mnist
