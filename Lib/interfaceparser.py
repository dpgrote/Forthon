# Created by David P. Grote, March 6, 1998
# Modified by T. B. Yang, May 19, 1998
# Parse the interface description file
# $Id: interfaceparser.py,v 1.3 2004/02/10 17:59:14 dave Exp $

# This reads in the entire variable description file and extracts all of
# the variable and subroutine information needed to create an interface
# to python (or other scripting language).

import sys
import string
import fvars
import re
if sys.version[0] == '1':
  import regsub

attribute_pat = re.compile('[ \t\n]*([a-zA-Z_]+)')

def processfile(packname,filename,othermacros=[]):

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
  text = string.strip(text[i+1:])

  # Check to make sure that the package name is correct
  if packname != line:
    print 'Warning: the package name in the file does not agree'
    print 'with the file name.'
    return []

  # Remove all line continuation marks
  text = re.sub('\\\\','',text)

  # Get rid of any initial comments and blank lines
  while text[0] ==  '#':
    text = string.strip(text[re.search('\n',text).start()+1:])

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
  if text[0] == '{':
    while text[0] != '}':
      if 'a' <= text[0] and text[0] <= 'z' or 'A' <= text[0] and text[0] <= 'Z':
        macro = string.strip(text[0:re.search('=',text).start()])
        text = string.strip(text[re.search('=',text).start()+1:])
        value = string.strip(text[0:re.search('[#\n]',text).start()])
        value = repr(eval(value))
        if sys.version[0] == '1':
          text = regsub.gsub('\<'+macro+'\>',value,text)
        else:
          text = re.sub('(?<=\W)'+macro+'(?=\W)',value,text)
        text = re.sub('/'+macro+'/','/'+value+'/',text)
      text = string.strip(text[re.search('\n',text).start()+1:])
    text = string.strip(text[re.search('\n',text).start()+1:])

  # Get rid of any comments and blank lines
  while text[0] ==  '#':
    text = string.strip(text[re.search('\n',text).start()+1:])

  # Parse rest of file, gathering variables

  # Create blank list to put the variables in.
  vlist = []

  # Create blank list to put the 'hidden' variables in.
  hidden_vlist = []

  # Create list of types
  typelist = []

  group = ''
  attributes = ''

  # Parse rest of file
  hidden = 0
  istype = 0
  readyfortype = 0
  while text:
    text = text + '\n'

    # Check if group
    if text[0] == '*':
      istype = 0
      # Then get new group name
      i = re.search(':',text).start()
      g = string.split(text[:i])
      group = g[1]
      # Include group name as an attribute
      attributes = ' '+string.join(g[1:])+' '
      # Look for the 'hidden' attribute
      hidden = 0
      if re.search(' hidden ',attributes) != None:
	hidden = 1
      # Strip off group name and any comments
      text = string.strip(text[i+1:]) + '\n'
      while text[0] ==  '#':
        text = string.strip(text[re.search('\n',text).start()+1:]) + '\n'
      # Set i so that nothing is stripped off at end of 'if' block
      i = -1
      readyfortype = 0

    # Check if derived type
    elif text[0] == '%':
      istype = 1
      # Then get new type name
      i = re.search(':',text).start()
      tname = string.split(text[:i])[1]
      group = tname
      # Include group name as an attribute
      attributes = ' '+tname+' '
      # Create new instance of Ftype and append to the list
      ftype = fvars.Ftype(tname,attributes)
      typelist.append(ftype)
      # Strip off group name and any comments
      text = string.strip(text[i+1:]) + '\n'
      while text[0] ==  '#':
        text = string.strip(text[re.search('\n',text).start()+1:]) + '\n'
      # Set i so that nothing is stripped off at end of 'if' block
      i = -1
      readyfortype = 0

    # Check if variable is dynamic
    elif text[0] == '_':
      v.dynamic = 1
      i = 0

    # Check if type is real
    elif text[0:4] == 'real':
      v.type = 'real'
      i = 3
      readyfortype = 0

    # Check if type is integer
    elif text[0:7] == 'integer':
      v.type = 'integer'
      i = 6
      readyfortype = 0

    # Check if type is logical
    elif text[0:7] == 'logical':
      v.type = 'logical'
      i = 6
      readyfortype = 0

    # Check if type is Filedes (temporary fix for now)
    elif text[0:7] == 'Filedes':
      v.type = 'integer'
      i = 6
      readyfortype = 0

    # Check if type is Filename (assumed to a string of length 256)
    elif text[0:8] == 'Filename':
      v.type = 'character'
      i = 7
      v.dims = ['256'] + v.dims
      readyfortype = 0

    # Check if type is character
    elif text[0:9] == 'character':
      v.type = 'character'
      v.array = 1
      i = re.search('[ \t\n]',text).start()
      if text[9] == '*':
        v.dims = [text[10:i]] + v.dims
      readyfortype = 0

    # Check if type is complex
    elif text[0:7] == 'complex':
      v.type = 'complex'
      i = 6
      readyfortype = 0
     #if not v.dims:
     #  v.dims = ['1']

    # Check if variable is a function
    elif text[0:8] == 'function':
      v.function = 1
      v.array = 0
      i = 7
      readyfortype = 0

    # Check if variable is a subroutine
    elif text[0:10] == 'subroutine':
      v.function = 1
      v.array = 0
      v.type = 'void'
      i = 9
      readyfortype = 0

    # Check if there are any dimensions
    elif text[0] == '(':
      v.array = 1
      i = 0
      p = 1
      while p > 0:
        i = i + 1
        try:
          if text[i] == '(':
            p = p + 1
          elif text[i] == ')':
            p = p - 1
        except IndexError:
          print 'Error in subscript of variable '+v.name
      v.dimstring = text[0:i+1]
      d = text[1:i]
      # Strip out all white space
      d = re.sub(' ','',d)
      d = re.sub('\t','',d)
      d = re.sub('\n','',d)
      if len(d) > 0 and d[0] == ';':
        d = d[1:]
      d = re.sub(';',',',d)
      d = string.splitfields(d,',')
      #v.dims.append(d)
      v.dims = v.dims + d

    # Look for a data field
    elif text[0] == '/':
      i = re.search('/',text[1:]).start() + 1
      v.data = text[0:i+1]
      readyfortype = 0

    # Look for a unit field
    elif text[0] == '[':
      i = re.search(']',text).start()
      v.unit = text[1:i]
      readyfortype = 0

    # Look for a limited field
    elif text[0:7] == 'limited':
      j = re.search('\(',text).start()
      i = j
      p = 1
      while p > 0:
        i = i + 1
        try:
          if text[i] == '(':
            p = p + 1
          elif text[i] == ')':
            p = p - 1
        except IndexError:
          print 'Error in subscript of variable '+v.name
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

    # Look for comment (and remove extra spaces)
    elif text[0] == '#':
      i = re.search('\n',text).start() - 1
      v.comment = v.comment + text[1:i+1] + ' '
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
      text = string.strip(text[i+1:])
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
        sd = string.splitfields(d,':')
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
      # Extract type from arguments
      for a in v.dims:
        sa = string.splitfields(a,':')
        fa = fvars.Fargs()
        fa.name = sa[0]
        if len(sa) > 1:
          fa.type = sa[1]
        else:
          fa.type = 'real'
          if 'i' <= fa.name[0] and fa.name[0] <= 'n':
            fa.type = 'integer'
        v.args = v.args + [fa]
      v.dims = []

    # Clean up the comment, removing extra spaces and replace " with '
    if v.comment:
      v.comment = re.sub(' +',' ',v.comment)
      v.comment = re.sub('"',"'",v.comment)

  # Do further processing on dims or arguments list, and check variable type
  for v in vlist: processvar(v)
  for t in typelist:
    for v in t.vlist: processvar(v)

  # Return the list
  return (vlist, hidden_vlist, typelist)
