"""Routine timer and profiler
These are generic classes which provide timing information about routine
calls and traces.
ForthonProfiler: accumulates function level timings and optionally prints
                 traces on function calls and returns
enablelinetracing: enables line level tracing
disablelinetracing: disables line level tracing
"""
import sys
import time
import linecache


def ForthonTimerdoc():
    import ForthonTimer
    print ForthonTimer.__doc__


#############################################################################
# --- These two classes create a profiler which keeps
# --- track on function executions and timings of them. Instances of the
# --- first class stores the data for each function. The second class does
# --- the setup and contains the profiler.
class ForthonTimings:
    """
    This is a holder for function calls and timing information. Each instance
    holds the timing info for a subroutine and references to the instances for its
    caller and all of its callees.
     - level: depth of routine call from the toplevel
     - name: name of the routine
     - parent: class instance of the caller of this routine
    """

    def __init__(self, level, name, parent=None):
        self.level = level
        self.name = name
        self.parent = parent
        self.time = 0.
        self.ncalls = 0
        self.subtimers = {}
        self.renew()

    def renew(self):
        """
        This restarts the timing clock and increments the number of calls made to
        this function.
        """
        self.starttime = time.clock()
        self.ncalls += 1

    def newtimer(self, name):
        """
        This returns a timer to be used for a callee. If the callee has already be
        called before, return that timer with the time reset. Otherwise create a
        new one and return it.
        """
        if name in self.subtimers:
            self.subtimers[name].renew()
        else:
            self.subtimers[name] = ForthonTimings(self.level+1, name, self)
        return self.subtimers[name]

    def stoptimer(self):
        """
        Stop the timing clock and accumulate the time. This assumes that this is
        called on a function return, so it returns the timer instance of the caller
        (the parent).
        """
        self.endtime = time.clock()
        self.time = self.time + self.endtime - self.starttime
        return self.parent

    def out(self, maxlevel, mintime=0.):
        """
        Prints info about the function and all of its callees, up to the input level.
        """
        if self.level > maxlevel:
            return
        if self.time > mintime:
            print "%2d%s%s %d %f"%(self.level, self.level*'  ', self.name, self.ncalls, self.time)
        for v in self.subtimers.itervalues():
            v.out(maxlevel, mintime)


class ForthonProfiler:
    """
    This class installs a profiler which gathers timing information for all python
    functions and methods called. It keeps a list of ForthonTimings instances
    for top level functions (those called from the same name space where the
    instance is created, or from the interpreter). Each of those contain all of
    the info about their callees.
    Argument:
      - trace=0: when true, a trace of the routines called and returned from will
                 be actively printed
      - tracelevel=None: when specified, trace is only printed in levels less than
                         the given value. The trace option is automatically
                         switched on.
    """

    _ninstances = 0

    def __init__(self, trace=0, tracelevel=None):
        if ForthonProfiler._ninstances > 0:
            raise RuntimeError("Only one instance allowed.")
        ForthonProfiler._ninstances = 1
        self.trace = trace
        self.tracelevel = tracelevel
        if self.tracelevel is not None:
            self.trace = 1
        self.restart()

    def restart(self):
        self.root = None
        self.level = 0
        self.finished = 0
        sys.setprofile(self.profiler)

    def finish(self):
        """
        Turn off the profiling. This also stops the clock of the top level routine.
        """
        if self.finished:
            return
        sys.setprofile(None)
        self.root.stoptimer()
        self.finished = 1

    def profiler(self, frame, event, arg):
        """
        This is the profile routine. It creates the instances of ForthonTimings for each
        routine called and starts and stops the timers.
        """
        # --- Get the name of the routine
        name = frame.f_code.co_name
        # --- If name is not known, this could mean that this was called from
        # --- inside the sys.setprofile routine, or from some other odd place,
        # --- like from fortran. Skip those cases.
        if name == '?':
            return
        # --- Turn the profiler off during these operations (though this is
        # --- probably unecessary since the sys package should already do this).
        sys.setprofile(None)
        # --- Create an instance of the timer for the toplevel if it hasn't
        # --- already been done.
        if self.root is None:
            self.root = ForthonTimings(0, "toplevel")
            self.timer = self.root
        if event == 'return' and self.level > 0:
            self.level = self.level - 1
            self.timer = self.timer.stoptimer()
        if self.trace:
            if self.tracelevel is None or self.tracelevel > self.level:
                # --- The flush is added so that anything that was printed to stdout
                # --- or stderr get outputed now so that the print outs occur
                # --- at the proper time relative to the trace.
                sys.stdout.flush()
                sys.stderr.flush()
                print "%s %s %s %s %d"%(self.level*'  ', event, name, frame.f_code.co_filename, frame.f_lineno)
        if event == 'call':
            self.level = self.level + 1
            self.timer = self.timer.newtimer(name)
        # --- Turn the profiler back on
        sys.setprofile(self.profiler)

    def out(self, maxlevel=2, mintime=0.):
        """
        Print out timing info.
         - maxlevel=2: only prints timings up to the given call depth
         - mintime=0.: only prints timings greater than or equal to the given value
        """
        self.finish()
        self.root.out(maxlevel, mintime)


###############################################################################
# --- Thanks to Andrew Dalke
# --- See http://www.dalkescientific.com/writings/diary/archive/2005/04/20/tracing_python_code.html
def traceit(frame, event, arg):
    if event == "line":
        lineno = frame.f_lineno
        try:
            filename = frame.f_globals["__file__"]
        except KeyError:
            filename = ''
        if (filename.endswith(".pyc") or filename.endswith(".pyo")):
            filename = filename[:-1]
        name = frame.f_globals["__name__"]
        line = linecache.getline(filename, lineno)
        print "%s:%s: %s" % (name, lineno, line.rstrip())
    return traceit


def enablelinetracing():
    """
    Enables line by line tracing.
    """
    sys.settrace(traceit)


def disablelinetracing():
    """
    Disables line by line tracing.
    """
    sys.settrace(None)
