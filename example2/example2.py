from example2py import *

print 'Testing basic call to fortran routine'
testsub2(5,6.)
print example2.t1.j,example2.t1.b
print 'Should be'
print '5 6.0'
print ''



print 'Tests complete'
