c========================================================================
      module Type2module
        save
        type Type2
          integer(kind=4):: cobj__ ! This must be added to all derived types
          integer(kind=4):: j
          real(kind=8):: b
        end type Type2
      contains
        ! This function must be added for each derived type, replacing 'Type2'
        ! with the name of the type.
        function NewType2() RESULT(newobj__)
          type(Type2),pointer:: newobj__
          integer:: error
          allocate(newobj__,STAT=error)
          if (error /= 0) then
            print*,"ERROR during allocation of Type2"
            stop
          endif
          call initType2py(-1,newobj__,newobj__%cobj__)
          return
        end function NewType2
      end module Type2module
c========================================================================
      module Module2
        use Type2module
        save
        type(Type2)::t1
        type(Type2),pointer::t2
      end module Module2
c========================================================================
      subroutine testsub2(ii,aa)
      use Module2
      integer(4):: ii
      real(8):: aa
      t1%j = ii
      t1%b = aa
      return
      end
c========================================================================
