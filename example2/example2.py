from example2py import *

print 'Testing basic call to fortran routine'
testsub2(5,6.)
print example2.t1.j,example2.t1.b
print 'Should be'
print '5 6.0'
print ''
print example2.t2.j,example2.t2.b
print 'Should be'
print '25 36.0'
print ''

testsub3()
print example2.t2
print 'Should be'
print 'unallocated'


print 'Tests complete'
