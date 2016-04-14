import example2py
from example2py import *

print('Testing basic call to fortran routine')
testsub2(5,6.)
print(example2.t1.j,example2.t1.b)
print('Should be')
print('5 6.0')
print('')
print(example2.t2.j,example2.t2.b)
print('Should be')
print('25 36.0')
print('')

testsub3()
try:
  print(example2.t2)
except example2py.error:
  print("printing t2 produced the correct error since it is unassociated")

print('Tests complete')
