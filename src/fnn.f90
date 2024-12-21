module fnn
  implicit none
  private

  public :: say_hello
contains
  subroutine say_hello
    print *, "Hello, fnn!"
  end subroutine say_hello
end module fnn
