c========================================================================
      module Type2module
        save
        type Type2
          integer(kind=4):: j
          real(kind=8):: b
        end type Type2
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
