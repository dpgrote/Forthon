#ifndef ISZ
#define ISZ 4
#endif
c========================================================================
      subroutine testsub1(ii,aa)
      integer(ISZ):: ii
      real(8):: aa
      print*,"ii,aa = ",ii,aa
      return
      end
c========================================================================
      subroutine testsub2(ii,aa,dd)
      use Module1
      integer(ISZ):: ii
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
      integer(ISZ):: ii,nn
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
      t2 => NewType1()
      t2%next => NewType1()
      t2%next%prev => t2
      t2%next%next => NewType1()
      t2%next%next%prev => t2%next
      t2%next%next%next => NewType1()
      t2%next%next%next%prev => t2%next%next
      return
      end
      subroutine a2()
      use Module2
      call ReleaseType1(t2%next%next%next)
      NULLIFY(t2%next%next%next)
      call ReleaseType1(t2%next%next)
      NULLIFY(t2%next%next)
      call ReleaseType1(t2%next)
      NULLIFY(t2%next)
      call ReleaseType1(t2)
      NULLIFY(t2)
      return
      end
      subroutine a3()
      use Module2
      t2 => NewType1()
      return
      end
c========================================================================
      subroutine printchar8(cccc)
      character(8):: cccc
      print*,"|",cccc,"|"
      return
      end
c========================================================================
      subroutine examplesetactionaction1(i)
      integer(ISZ):: i
      print*,"action1 is being set to ",i
      return
      end
c========================================================================
      subroutine examplegetactionaction1()
      print*,"action1 is being get"
      return
      end
c========================================================================
      subroutine type2setactionxx(x)
      real(kind=8):: x
      print*,"Type2 xx being set to ",x
      return
      end
c========================================================================
      subroutine type2getactionxx()
      print*,"Type2 xx being get"
      return
      end
c========================================================================
