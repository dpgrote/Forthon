# Created by David P. Grote, March 6, 1998
# Modified by T. B. Yang, May 19, 1998
# Parse the interface description file
# $Id: interfaceparser.py,v 1.22 2011/05/20 00:09:34 grote Exp $

# This reads in the entire variable description file and extracts all of
# the variable and subroutine information needed to create an interface
# to python (or other scripting language).

import sys
import fvars
import re
if sys.version[0] == '1':
    import regsub

attribute_pat = re.compile('[ \t\n]*([a-zA-Z_]+)')

def processfile(packname,filename,othermacros=[],timeroutines=0):

    # Open the variable description file
    varfile = open(filename,'r')

    # Read in the whole thing
    # (Make sure that there is a newline character at the end)
    text = varfile.read() + '\n'
    varfile.close()

    # Check if any other files where included in the list
    textother = []
    for a in othermacros:
        f = open(a,'r')
        t = f.read()
        f.close()
        textother = textother + [t]

    # Get the package name (first line of file) and strip it from the text
    i = re.search('[\n \t#]',text).start()
    line = text[:i]
    text = text[i+1:].strip()

    # Check to make sure that the package name is correct
    if packname != line:
        print """

    Warning: the package name in the file does not agree
    with the file name. Make sure that the first line of the
    variable description file is the package name.
    file name = %r
    packagename = %r

    One possible reason this error is that the variable description
    file is in dos format. If this is so, change the format to match
    your system. (This can be done in vi by opening the file and
    typing ":set fileformat=unix" and then saving the file.)

    """%(packname,line)
        return []

    # Remove all line continuation marks
    text = re.sub('\\\\','',text)

    # Get rid of any initial comments and blank lines
    while len(text) > 0 and text[0] ==  '#':
        text = text[re.search('\n',text).start()+1:].strip()

    # Get macros (everthing between curly braces) from other files and
    # prepend to current file (inside curly braces).
    for t in textother:
        m = t[re.search('\n{',t).start()+2:re.search('\n}',t).start()+1]
        if m:
            if text[0] == '{':
                text = '{' + m + text[1:]
            else:
                text = '{' + m + '}\n' + text

    # Deal with statements in between initial curly braces
    # (Only macro statements are used, everything else is ignored.)
    if len(text) > 0 and text[0] == '{':
        while text[0] != '}':
            if 'a' <= text[0] and text[0] <= 'z' or 'A' <= text[0] and text[0] <= 'Z':
                macro = text[0:re.search('=',text).start()].strip()
                text = text[re.search('=',text).start()+1:].strip()
                value = text[0:re.search('[#\n]',text).start()].strip()
                value = repr(eval(value))
                if sys.version[0] == '1':
                    text = regsub.gsub('\<'+macro+'\>',value,text)
                else:
                    text = re.sub('(?<=\W)'+macro+'(?=\W)',value,text)
                text = re.sub('/'+macro+'/','/'+value+'/',text)
            text = text[re.search('\n',text).start()+1:].strip()
        text = text[re.search('\n',text).start()+1:].strip()

    # Get rid of any comments and blank lines
    while len(text) > 0 and text[0] ==  '#':
        text = text[re.search('\n',text).start()+1:].strip()

    # Parse rest of file, gathering variables

    # Create blank list to put the variables in.
    vlist = []

    # Create blank list to put the 'hidden' variables in.
    hidden_vlist = []

    # Create list of types
    typelist = []

    group = ''
    attributes = ''

    # The parser needs to distinguish between variable names and type names.
    # When readyfortype is 0, the next string found will be interpreted as a
    # variable name. If 1, then it will be a type name.
    readyfortype = 0

    # Parse rest of file
    hidden = 0
    istype = 0
    while text:
        text = text + '\n'

        # Check if group
        if text[0] == '*':
            istype = 0
            # Check the syntax
            g = re.match('\*+ (\w+)((?:[ \t]+\w+)*):',text)
            if g is None:
                g = re.match('(.*)',text)
                raise SyntaxError('Line defining the group is invalid\n%s'%g.group(1))
            # Then get new group name
            i = re.search(':',text).start()
            g = text[:i].split()
            group = g[1]
            # Include group name as an attribute
            attributes = ' '+' '.join(g[1:])+' '
            # Look for the 'hidden' attribute
            hidden = 0
            if re.search(' hidden ',attributes) != None:
                hidden = 1
            # Strip off group name and any comments
            text = text[i+1:].strip() + '\n'
            while text[0] ==  '#':
                text = text[re.search('\n',text).start()+1:].strip() + '\n'
            # Set i so that nothing is stripped off at end of 'if' block
            i = -1
            readyfortype = 0

        # Check if derived type
        elif text[0] == '%':
            istype = 1
            # Then get new type name
            i = re.search(':',text).start()
            tname = text[:i].split()[1]
            group = tname
            # Include group name as an attribute
            attributes = ' '+tname+' '
            # Create new instance of Ftype and append to the list
            ftype = fvars.Ftype(tname,attributes)
            typelist.append(ftype)
            # Strip off group name and any comments
            text = text[i+1:].strip() + '\n'
            while text[0] ==  '#':
                text = text[re.search('\n',text).start()+1:].strip() + '\n'
            # Set i so that nothing is stripped off at end of 'if' block
            i = -1
            readyfortype = 0

        # Check if variable is dynamic
        elif text[0] == '_':
            v.dynamic = 1
            i = 0

        # Check if type is real
        elif re.match('real\s',text):
            v.type = 'real'
            i = 3
            readyfortype = 0

        # Check if type is double
        elif re.match('double\s',text):
            v.type = 'double'
            i = 5
            readyfortype = 0

        # Check if type is float
        elif re.match('float\s',text):
            v.type = 'float'
            i = 4
            readyfortype = 0

        # Check if type is integer
        elif re.match('integer\s',text):
            v.type = 'integer'
            i = 6
            readyfortype = 0

        # Check if type is logical
        elif re.match('logical\s',text):
            v.type = 'logical'
            i = 6
            readyfortype = 0

        # Check if type is Filedes (temporary fix for now)
        elif re.match('Filedes\s',text):
            v.type = 'integer'
            i = 6
            readyfortype = 0

        # Check if type is Filename (assumed to a string of length 256)
        elif re.match('Filename\s',text):
            v.type = 'character'
            i = 7
            v.dims = ['256'] + v.dims
            readyfortype = 0

        # Check if type is character
        elif re.match('character[\s*]',text):
            v.type = 'character'
            v.array = 1
            i = re.search('[ \t\n]',text).start()
            if text[9] == '*':
                v.dims = [text[10:i]] + v.dims
            readyfortype = 0

        # Check if type is complex
        elif re.match('complex\s',text):
            v.type = 'complex'
            i = 6
            readyfortype = 0
           #if not v.dims:
           #  v.dims = ['1']

        # Check if variable is a function
        elif re.match('function\s',text):
            v.function = 'fsub'
            v.array = 0
            i = 7
            readyfortype = 0
            if timeroutines:
                # --- Add variable used to accumulate the time
                timerv = fvars.Fvars()
                timerv.name = v.name+'runtime'
                timerv.type = 'real'
                timerv.data = '/0./'
                timerv.unit = 'seconds'
                timerv.comment = 'Run time for function %s'%v.name
                timerv.group = group
                timerv.attr = attributes
                vlist.append(timerv)

        # Check if variable is a subroutine
        elif re.match('subroutine\s',text):
            v.function = 'fsub'
            v.array = 0
            v.type = 'void'
            i = 9
            readyfortype = 0
            if timeroutines:
                # --- Add variable used to accumulate the time
                timerv = fvars.Fvars()
                timerv.name = v.name+'runtime'
                timerv.type = 'real'
                timerv.data = '/0./'
                timerv.unit = 'seconds'
                timerv.comment = 'Run time for subroutine %s'%v.name
                timerv.group = group
                timerv.attr = attributes
                vlist.append(timerv)

        # Check if variable is a C subroutine (takes C ordered arrays)
        elif re.match('csubroutine\s',text):
            v.function = 'csub'
            v.array = 0
            v.type = 'void'
            i = 10
            readyfortype = 0
            if timeroutines:
                # --- Add variable used to accumulate the time
                timerv = fvars.Fvars()
                timerv.name = v.name+'runtime'
                timerv.type = 'real'
                timerv.data = '/0./'
                timerv.unit = 'seconds'
                timerv.comment = 'Run time for C subroutine %s'%v.name
                timerv.group = group
                timerv.attr = attributes
                vlist.append(timerv)

        # Check if variable is a parameter, i.e. not writable
        elif re.match('parameter\s',text):
            if v.array or v.function:
                raise SyntaxError('%s: only scalar variables can be a parameter'%v.name)
            v.parameter = 1
            i = 8

        # Check if there are any dimensions
        elif text[0] == '(':
            v.array = 1
            i = findmatchingparenthesis(0,text,v.name)
            v.dimstring = text[0:i+1]
            v.dims = v.dims + convertdimstringtodims(v.dimstring)

        # Look for a data field
        elif text[0] == '/':
            i = 1
            # If the data is a string, skip over the string in case the
            # '/' character appears, as in a date for example.
            if   text[i] == '"': i = re.search('"',text[2:]).start() + 2
            elif text[i] == "'": i = re.search("'",text[2:]).start() + 2
            i = re.search('/',text[i:]).start() + i
            data = text[0:i+1]
            # Handle the old Basis syntax for initial logical values.
            if data[1:-1] == 'FALSE': data = '/.false./'
            if data[1:-1] == 'TRUE': data = '/.true./'
            v.data = data
            readyfortype = 0

        # Look for a unit field
        elif text[0] == '[':
            i = re.search(']',text).start()
            v.unit = text[1:i]
            readyfortype = 0

        # Look for a limited field
        elif text[0:7] == 'limited':
            j = re.search('\(',text).start()
            i = findmatchingparenthesis(j,text,v.name)
            v.limit = text[j:i+1]
            readyfortype = 0

        # Look for attribute to add
        elif text[0] == '+':
            m = attribute_pat.match(text[1:])
            i = m.end() + 1
            if i != -1:
                v.attr = v.attr + m.group(1) + ' '
            readyfortype = 0

        # Look for attribute to subtract
        elif text[0] == '-':
            m = attribute_pat.match(text[1:])
            if m != None:
                i = m.end() + 1
                if sys.version[0] == '1':
                    v.attr = regsub.sub('\<'+m.group(1)+'\>',' ',v.attr)
                else:
                    v.attr = re.sub('(?<=\W)'+m.group(1)+'(?=\W)',' ',v.attr,count=1)
            else:
                i = 0
            readyfortype = 0

        # Look for a set action flag
        elif re.match('SET\s',text):
            v.setaction = 1
            i = 2
            readyfortype = 0

        # Look for a get action flag
        elif re.match('GET\s',text):
            v.getaction = 1
            i = 2
            readyfortype = 0

        # Look for comment (and remove extra spaces)
        elif text[0] == '#':
            i = re.search('\n',text).start() - 1
            v.comment = v.comment + text[1:i+2].lstrip()
            readyfortype = 0

        # Look for private remark
        elif text[0] == '$':
            i = re.search('\n',text).start() - 1
            readyfortype = 0

        # This only leaves a variable name or a new type
        else:
            if not readyfortype:
                # Create new variable
                v = fvars.Fvars()
                if not hidden and not istype:
                    vlist.append(v)
                elif hidden:
                    hidden_vlist.append(v)
                elif istype:
                    ftype.addvar(v)
                i = re.search('[ (\t\n]',text).start() - 1
                v.name = text[:i+1]
                v.group = group
                v.attr = attributes
                readyfortype = 1
            else:
                readyfortype = 0
                i = re.search('[ \t\n]',text).start() - 1
                v.type = text[:i+1]
                v.derivedtype = 1

        # Strip off field which was just parsed
        try:
            text = text[i+1:].strip()
        except IndexError:
            text = ''

    def processvar(v):
        # Use implicit typing if variable type was not set.
        if v.type == '':
            v.type = 'real'
            if 'i' <= v.name[0] and v.name[0] <= 'n':
                v.type = 'integer'

        # Parse the dimensions of arrays
        if v.array:
            assert v.dims,"%s dimensions not set properly"%v.name
            dims = v.dims
            v.dims = []
            for d in dims:
                sd = d.split(':')
                fd = fvars.Fdims()
                if len(sd) == 1:
                    fd.low = '1'
                    fd.high = sd[0]
                else:
                    fd.low = sd[0]
                    fd.high = sd[1]
                # Check for dimensions which are expressions
                if v.dynamic:
                    if fd.high == '':
                        # Flag the dynamic arrays which do not have dimensions
                        # specified. Give it the fassign attribute since it is
                        # assumed that a shapeless variable will be assigned
                        # to in fortran.
                        v.dynamic = 3
                        v.attr = v.attr + ' fassign '
                    elif (re.search(fd.low ,'[/\*\+\-]') != None or
                          re.search(fd.high,'[/\*\+\-]') != None):
                        # Flag the dynamic arrays which have an expression for a dimension
                        v.dynamic = 2
                else:
                    # For static arrays, evaluate the dimensions or remove the
                    # expression
                    if re.search(fd.low ,'[/\*\+\-]') != None:
                        fd.low = repr(eval(fd.low))
                    if re.search(fd.high,'[/\*\+\-]') != None:
                        fd.high = repr(eval(fd.high))
                v.dims = v.dims + [fd]
            #v.dims.reverse()

        # Set v.args if variable is a function
        if v.function:
            # Remove a possible blank first argument
            if len(v.dims) > 0 and v.dims[0]=='': del v.dims[0]
            # Extract type from arguments, this includes possible dimension
            # descriptions of arrays
            # --- dimvars holds the list of arguments that are used in array
            # --- dimensions
            v.dimvars = []
            for a in v.dims:
                fa = fvars.Fargs()
                v.args = v.args + [fa]
                # --- parse the pattern name(dim1:dim2,...):type
                # --- Only the name must be given, the other parts are optional
                ma = re.search("[:(]|\Z",a)
                fa.name = a[:ma.start()]
                if ma.group() == '(':
                    i = findmatchingparenthesis(ma.start(),a,v.name)
                    fa.dimstring = a[ma.start():i+1]
                    dimlist = fa.dimstring[1:-1].split(',')
                    fa.dims = processargdimvars(dimlist,v.dimvars)
                    a = a[i+1:]
                    ma = re.search("[:]|\Z",a)
                else:
                    fa.dimstring = ''
                    fa.dims = []
                # --- Set the argument type
                if ma.group() == ':':
                    fa.type = a[ma.start()+1:]
                else:
                    # --- Use implicit typing if the type was not specified
                    fa.type = 'real'
                    if 'i' <= fa.name[0] and fa.name[0] <= 'n':
                        fa.type = 'integer'
            # --- Change the list of dimvars from names of the arguments to
            # --- references to the Fargs instance and position number
            i = 0
            for arg in v.args:
                if arg.name in v.dimvars:
                    assert arg.type in ['integer'],"Error in %s:Variables used in array dimensions for argument list must be integer type"%v.name
                    v.dimvars[v.dimvars.index(arg.name)] = [arg,i]
                i += 1
            # --- Make sure that only names that are arguments are used
            # --- This checks for leftover names that wern't replaced by the
            # --- above loop.
            for var in v.dimvars:
                if type(var) == type(''):
                    raise SyntaxError("%s: Only subroutine arguments can be specified as dimensions. Bad argument name is %s"%(v.name,var))
            # --- Empty out dims which is no longer needed (so it won't cause
            # --- confusion later).
            v.dims = []

        # Clean up the comment, removing extra spaces and replace " with '
        if v.comment:
            v.comment = re.sub(' +',' ',v.comment)
            v.comment = re.sub('"',"'",v.comment)
            v.comment = v.comment.strip()

    # Do further processing on dims or arguments list, and check variable type
    for v in vlist: processvar(v)
    for t in typelist:
        for v in t.vlist: processvar(v)

    # Return the list
    return (vlist, hidden_vlist, typelist)

def findmatchingparenthesis(i,text,errname):
    # --- Note that text[i] should be "(".
    p = 1
    while p > 0:
        i = i + 1
        try:
            if text[i] == '(':
                p = p + 1
            elif text[i] == ')':
                p = p - 1
        except IndexError:
            print 'Error in subscript of variable '+errname
    return i

def convertdimstringtodims(dimstring):
    # --- Remove the beginning and ending parenthesis
    d = dimstring[1:-1]
    # --- Strip out all white space
    d = re.sub('[ \t\n]','',d)
    # --- Remove optional argument signifying if it is the first character
    if len(d) > 0 and d[0] == ';':
        d = d[1:]
    # --- Remove other optional arguments
    d = re.sub(';',',',d)
    # --- Carefully search through dimstring, looking for comma separated
    # --- arguments and for parenthetical blocks (that may contain commas
    # --- as well as other text, numbers, math symbols, and parenthesis).
    dimlist = []
    while len(d) > 0:
        m = re.search("[,(]",d)
        if m is not None and m.group() == '(':
            i0 = findmatchingparenthesis(m.start(),d,d) + 1
            m = re.search("[,(]",d[i0:])
        else:
            i0 = 0
        if m is None:
            i = len(d)
        elif m.group() == ',':
            i = i0 + m.start()
        dimlist = dimlist + [d[:i]]
        d = d[i+1:]
    return dimlist

def processargdimvars(dims,dimvars):
    # --- For each dimensions, finds the low and high pieces and finds all
    # --- of the names which appear.
    dimlist = []
    for d in dims:
        fd = fvars.Fdims()
        dimlist = dimlist + [fd]
        # --- Find the low and high pieces, which are separated by a colon
        sd = d.split(':')
        if len(sd) == 1:
            fd.low = '1'
            fd.high = sd[0]
        else:
            fd.low = sd[0]
            fd.high = sd[1]
         # --- This is not really needed since it can be done in C too.
         #try:
         #  low = repr(eval(fd.low))
         #  fd.low = low
         #except:
         #  pass
         #try:
         #  high = repr(eval(fd.high))
         #  fd.high = high
         #except:
         #  pass
        # --- Add any variable names to the list of variables in dimensions
        for dim in [fd.low,fd.high]:
            sl = re.split('[ ()/\*\+\-]',dim)
            for ss in sl:
                if re.search('[a-zA-Z]',ss) != None:
                    if ss not in dimvars: dimvars.append(ss)
    return dimlist

