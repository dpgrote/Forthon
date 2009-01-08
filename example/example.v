example

##############################################################################
# This example is also the documentation for the format of the interface
# description file.
#
# This file contains descriptions of the modules, derived types, and callable
# subroutines. Variables are accessed in python by prefixing their name
# with the package name, for example the variable xxx can be accessed
# using example.xxx. Note that this puts a constraint on the namespace
# the does not exist in fortran - two variables in a package cannot have the
# same name even though they may be in different modules.
# When calling subroutines, the package name prefix is not needed (but can
# be used).
#
# The first line of the interface file is the name of the package and must be
# present and consistent with the name used in the Forthon command line.
#
# The variables are grouped by fortran modules. The line defining a module
# has the format
#    **** Modulename [moduleattributes]:
# Any number of '*' can be used but they must be contiguous (no spaces between).
# The Modulename corresponds to the Fortran module name. Fortran routines
# using the module would have the following line
#    use Modulename
# There can be zero or any number of moduleattributes. All variables in the
# group will have that attribute (unless explicitly removed). The attributes
# are primarily used in python to select subsets of variables. For example the
# command example.varlist('test') will return a list of all variables with
# the attribute test. There are three special attributes.
#   dump: The function dump will by default write to the dump file all
#         variables with the attribute dump.
#   fassign: This only affects array pointers. For arrays with
#            this attribute, the wrapper generator will create extra coding
#            which allows a pointer assignment to be done in fortran and
#            have it be recognized in python.
#   hidden: Variables with this attribute will be hidden and not be accessible
#           from python.
#
# The definition of scalar variables has the following format
#   varname type /initvalue/ [units] +attr1 -attr2 # documentation
# Only the varname and type are required and must be first and in that order.
# The remaining fields can be in any order except that the documentation
# (preceded by the # sign) must be last. The documentation can be
# multiline, with each line begining with a # sign. Comments can be given
# preceded by a $ sign - these are not accessible from python.
# If an initvalue is given, the variable is given that default value in the
# module.
# Attributes can be added to or removed from individual variables using
# the + or - syntax.
# The type must be one of real, integer, logical, complex, or one of the
# defined derived types. Note that real defaults to double, integer and
# logical have the same length as a C long, and complex is double.
# Variables of derived type can be defined as a fortran pointer by prepending
# the type name with an underscore. (See definition of t2 below).
# Currently, the units are ignored.
#
# The definition of arrays has the same syntax as above except for the
# addition of the array shape.
#   varname(shape) type /initvalue/ [units] +attr1 -attr2 # documentation
# The shape takes the same syntax as is used in fortran. The array can be
# declared as a fortran pointer by prepending the type name with an underscore.
# Note that any array with a nonconstant shape must be a pointer.
# For nonpointer arrays, an initvalue must have the correct number of elements
# to fill the array (the fortran data statement syntax is used). For pointer
# arrays, initvalue must be a single value which is broadcast throughout the
# array upon allocation.
# The shape can contain integers from other packages, but these package names
# must be specified on the Forthon command line.
# Note that at the python level, the indexing of arrays always begins with 0,
# so an array with shape (3:5) will be accessed using indices 0, 1, and 2.
# Note that arrays with unspecified bounds (e.g. z(:)) will always be given
# the fassign attribute.
# Note that arrays of derived type are not permitted.
#
# The definition of subroutines has the following format.
#   subname(arglist) subroutine # documentation
# The arglist specifies the calling sequence and must be consistent with
# that in the source. The arglist consists of the list of input quantities,
# each followed by a colon and its type (e.g. x:real,i:integer). The type
# can be any of the types of scalars. For array arguments, the elemental type
# is given.
# Though subroutines are listed inside of modules, this does not imply that
# the fortran module will contain the subroutine. Actually, because
# fortran compilers do name mangling of subroutines in modules, the
# subroutines must be outside of a module.
#
# Functions can be defined using the format
#   funname(arglist) funtype function # documentation
# The function type can only be one of real, integer, or logical.
#
# Derived types are defined similarly to modules, except that the name is
# preceded with %'s (instead of *'s). The % reflects the use of that character
# in fortran to refer to elements of a derived type. Initial values can be
# given, but since not all compilers can deal with them (it is a Fortran95
# feature), they are treated differently than for plain variables. The
# only difference is that non-dynamic arrays can only be given a scalar
# value (the same as dynamic arrays).
# For all array pointers in a derived type, any variables used in the shape
# must be an element of that derived type.
#
##############################################################################

{
# A block in between curly braces is optional. It can contain macros that
# can be used below, for example as an initial value. These are not accessible
# from python.
ANINITVALUE = 7.
}

****** Module1 test:
i integer /3/ # Sample integer variable
a real /ANINITVALUE/ # Sample real variable
d(3) real /3*10./ # Sample static array
n integer /0/ # Size of sample array pointer
x(0:n) _real /1/ # Sample array pointer
z(:) _real # Sample array pointer with undefined bounds.
xxx(:,:) _real # Sample multidimensional array
l1 logical /.false./ # Sample logical variable
realvar real /1./
varreal real /2./

%%%%% Type2:
ii integer
xx real

%%%%% Type1:
$ Sample derived type
j integer /7/ # Integer element of a derived type
b real # Real element of a derived type
e(10) real /8./ # Static array element of a derived type
m integer # Size of array pointer in derived type
y(0:m) _real /3./ # Array pointer element of a derived type
static2 Type2 # Pointer to derived type object of the same type
next _Type1 # Pointer to derived type object of the same type
prev _Type1 # Pointer to derived type object of the same type
xxx(:,:) _real

***** Module2:
t1 Type1 # Test derived type
t2 _Type1 # Test derived type pointer

***** Subroutines:
testsub1(ii:integer,aa:real) subroutine # Test basic call to fortran subroutine
testsub2(ii:integer,aa:real,dd:real) subroutine # Test setting of variable
                                                # in fortran.
testsub3(ii:integer,aa:real,nn:integer) subroutine # Test operating on
                                                   # derived type variables.
testsub33() subroutine # Extra subroutine
testsub4() subroutine # Extra subroutine
testsub5() subroutine # Test operations on array pointer with undefined bounds
testsub6(t:Type1) subroutine # Test passing derived type to fortran.
testsub10(ii:integer,aa(1:nn,3):real,nn:integer) subroutine
   # This subroutine is declared in a separate fortran file.

a1() subroutine
a2() subroutine
a3() subroutine

****** Stringtest:
tstring character*4 /"////"/ # test "/" in strings
cccc character*8
printchar8(cccc:string) subroutine
