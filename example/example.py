from Forthon import *
from examplepy import *

print 'Testing basic call to fortran routine'
testsub1(5,6.)
print 'Should be'
print ' ii,aa =            5   6.00000000000000'
print ''

print 'Testing of setting variables in python and fortran'
print example.i,example.a,example.d
print 'Should be'
print '3 7.0 [ 10.  10.  10.]'
print ''
example.i = 7
example.a = 8.
example.d = [3.1415926,2.99792458e+8,2.718281828459]
print example.i,example.a,example.d
print 'Should be'
print '7 8.0 [  3.14159260e+00   2.99792458e+08   2.71828183e+00]'
print ''
testsub2(5,6.,[-1.,0.,1.])
print 'Should be'
print ' i,a =             7   8.00000000000000'
print ' d =    3.14159260000000        299792458.000000        2.71828182845900'
print ''
print example.i,example.a,example.d
print 'Should be'
print '5 6.0 [-1.  0.  1.]'
print ''

print 'Testing operating on dynamic derived type'
testsub3(5,6.,4)
print example.t2.j,example.t2.b,example.t2.y
print 'Should be'
print '5 6.0 [ 1.  1.  1.  1.  1.]'
print ''

print 'Testing shapeless arrays'
example.z = [1,2,3]
print example.z
print 'Should be'
print '[ 1.  2.  3.]'
print ''
testsub5()
print 'Should be'
print ' z =    1.00000000000000        2.00000000000000        3.00000000000000'

print 'Passing derived types into fortran subrotuines'
example.t2.j = 20
testsub6(example.t2)
print 'Should be'
print ' t%j =           20'
print ''


print 'Testing array assignment of the form pkg.x = x'
xx = fzeros((2,2),'d')
example.xxx = xx
xx[1,1] = 1
print 'The following two arrays should be identical, since only pointer'
print 'referencing is done if the RHS array has fortran ordering.'
print xx
print example.xxx
xx = zeros((2,2),'d')
example.xxx = xx
xx[1,1] = 1
print 'The following two arrays should be different, since a copy is done'
print 'if the RHS array has C ordering.'
print xx
print example.xxx


print 'Tests complete'
