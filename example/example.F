c========================================================================
      subroutine testsub1(ii,aa)
      integer(4):: ii
      real(8):: aa
      print*,"ii,aa = ",ii,aa
      return
      end
c========================================================================
      subroutine testsub2(ii,aa,dd)
      use Module1
      integer(4):: ii
      real(8):: aa,dd(3)
      print*,"i,a = ",i,a
      print*,"d = ",d
      i = ii
      a = aa
      d = dd
      return
      end
c========================================================================
      subroutine testsub3(ii,aa,nn)
      use Module2
      integer(4):: ii,nn
      real(8):: aa
      t2 => NewType1()
      t2%j = ii
      t2%b = aa
      t2%m = nn
      call Type1allot(t2)
      t2%y = 1.
      return
      end
c========================================================================
      subroutine testsub4()
      use Module1
      allocate(z(11))
      z = 4
      return
      end
c========================================================================
      subroutine testsub5()
      use Module1
      print*,"z = ",z
      return
      end
c========================================================================
      subroutine testsub6(t)
      use Type1module
      Type(Type1):: t
      print*,"t%j = ",t%j
      return
      end
c========================================================================
      subroutine testsub33()
      use Module2
      type(Type1),pointer:: xx
      xx => NewType1()
      call DelType1(xx)
      return
      end
c========================================================================
      subroutine a1()
      use Module2
      allocate(t2)
      return
      end
      subroutine a2()
      use Module2
      call ReleaseType1(t2)
      return
      end
      subroutine a3()
      use Module2
      print*,t2%j
      return
      end

