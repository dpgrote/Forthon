example2

%%%%% Type2:
j integer # Integer element of a derived type
b real # Real element of a derived type

***** Module2:
t1 Type2 # Test derived type
t2 _Type2 # Test derived type pointer. This is included to show the required
          # way to create and delete a derived type reference.

***** Subroutines:
testsub2(ii:integer,aa:real) subroutine
testsub3() subroutine

