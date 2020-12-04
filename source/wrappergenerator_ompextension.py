#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:06:47 2020

@author: jguterl
"""
import re
from . import fvars

try:
    from colorama import Back, Style
    CYAN = Back.CYAN
    RESET_ALL = Style.RESET_ALL
except ImportError:
    CYAN = ''
    RESET_ALL = ''

class PyWrap_OMPExtension():
    def OMPInit(self,ompactive=False,ompdebug=False):
        self.ListThreadPrivateVars=[]
        self.ompactive=ompactive
        self.ompdebug=ompdebug
        if self.ompactive or self.ompdebug:
            print('{color} Adding OPENMP variables and routines for package: {}{reset}'.format(self.pkgname,color=CYAN,reset=RESET_ALL))
    def ProcessOMPVarList(self):
        """
        Author: Jerome Guterl (JG)
        Read in and parse a file which must be 
        """
        self.ompvarlist=[]
        # First check that the file can be found and read
        if self.ompvarlistfile is not None:
            try: 
                f=open(self.ompvarlistfile)
            except:
                # JG: We throw an error so the compilation will fail and the user will know that something
                # is wrong. It is not easy to identify a simple warning during the compilation of the
                # forthon packages.
                raise IOError("{color}Could not open/read the ompvarlistfile file :{}{reset}".format(self.ompvarlistfile,color=CYAN,reset=RESET_ALL))
                
            with f:
                for line in f:
                    line = line.split('#', 1)[0]
                    line = line.rstrip()
                    if len(line)>0:
                        self.ompvarlist.append(line)
            
            print('{color} List of variables requested to be threaded {reset}:{}'.format(self.ompvarlist,color=CYAN,reset=RESET_ALL))
    
    def ProcessDim(self,S):
        if S.count('(')>0:
            Str=S.split('(')[1].split(')')[0] 
            Dims=Str.split(',')
            for i in range(len(Dims)):
                if Dims[i].count(':')<1:
                    Dims[i]='1:'+Dims[i]
            S='('+','.join(Dims)+')'
        return S
    
    def GetSize(self,S):
        
        if S.count('(')>0:
            Out=[]
            Str=S.split('(')[1].split(')')[0] 
            Dims=Str.split(',')
            for i in range(len(Dims)):
                if Dims[i].count(':')<1:
                    if int(Dims[i])!=1:
                        raise ValueError('Dimension not correct:{}',S)
                    Out.append('('+Dims[i]+')')
                else:    
                    Out.append('('+Dims[i].split(':')[1]+'-'+Dims[i].split(':')[0]+'+1'+')')
            Out='*'.join(Out)
        else:            
            Out='1'
        return Out
            
            
    def writef90OMPCopyHelper(self):
        self.fw('module OmpCopy{}'.format(self.pkgname))
        # self.fw('  use OMPSettings')
        # First we add all the use(group) necessary to have access to the data
        self.fw('contains' ) 
        ompcommoncopy=[]
        
        for a in self.alist:
            if a.threadprivate and a.type != 'character':
                self.fw('subroutine OmpCopyPointer{}'.format(a.name)) 
                
                # Group with dimension first 
                groupsadded=[]
                groups = self.dimsgroups(a.dimstring)
                for g in groups:
                    if g not in groupsadded:
                        self.fw('  use ' + g)
                        groupsadded.append(g)
                # groups to access variables                
                if a.group not in groupsadded:
                    self.fw('  use ' + a.group)
                    groupsadded.append(a.group)
                self.fw('  integer:: tid,omp_get_thread_num') 
                # declare variable to be copied in threads:
                S=self.prefixdimsf(re.sub('[ \t\n]', '', a.dimstring))
                S=self.ProcessDim(S)
                self.fw('  ' + fvars.ftof(a.type) + '::{}copy{}'.format(a.name,S))        
                #write common block
                ompcommoncopy.append(a.name+'copy')
                if a.dynamic:
                    self.fw('if ({}.le.size({})) then'.format(self.GetSize(S),a.name))
                self.fw('{}copy{}={}{}  '.format(a.name,S,a.name,S))
                
                # self.fw('if (OMPCopyDebug.gt.0) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP::: Starting parallel construct to copy variable {}'".format(a.name))
                # self.fw('endif')    
                self.fw('!$omp parallel private(tid) firstprivate({}copy)'.format(a.name))#' copyin(/comompcopy/)')
                self.fw('tid=omp_get_thread_num()')
                self.fw('if (tid.gt.0) then')
                # self.fw('if (OMPCopyDebug.gt.0) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP::: Thread #',tid,' : copying...'")
                # self.fw('endif')
                self.fw('{}{}={}copy{}'.format(a.name,S,a.name,S)) 
                self.fw('endif ')         
                self.fw('!$omp end parallel')
                # self.fw('if (OMPCopyDebug.gt.0) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP::: End of parallel construct to copy variable {}'".format(a.name))
                # self.fw('endif')           
                if a.dynamic:
                    self.fw('endif')
                self.fw('  return')
                
                self.fw('end subroutine OmpCopyPointer{}'.format(a.name)) 
        self.fw('subroutine OmpCopyPointer{}'.format(self.pkgname))
        for a in self.alist:
                if a.threadprivate and a.type != 'character':
                    # self.fw('if (OMPCopyDebug.gt.0) then')
                    # self.fw("write(*,*) '#Master::: Calling routine to copy variable {}'".format(a.name))
                    # self.fw('endif')
                    self.fw('call OmpCopyPointer{}'.format(a.name))
        self.fw('end subroutine OmpCopyPointer{}'.format(self.pkgname))   
        self.fw('subroutine OmpCopyScalar{}'.format(self.pkgname))
        ompcommonscalar=[]
        groupsadded=[]
        for s in self.slist: 
            if s.threadprivate and s.type != 'character':
                if s.group not in groupsadded:
                    self.fw('  use ' + s.group)
                    groupsadded.append(s.group)
                ompcommonscalar.append(s.name)
                
        self.fw('integer::tid,omp_get_thread_num')
        if len(ompcommonscalar)>0:
            for s in self.slist:
                if s.name in ompcommonscalar:
                    self.fw('  ' + fvars.ftof(s.type) + '::{}copy'.format(s.name))
        if len(ompcommonscalar)>0:
            ompcommonscalarcopy=[s+'copy' for s in ompcommonscalar]
            self.fw('common /ompcommonscalarcopy/ '+','.join(ompcommonscalarcopy))
            for s in self.slist:
                if s.name in ompcommonscalar:
                    self.fw('{}copy={}'.format(s.name,s.name))

            self.fw('!$omp parallel private(tid) firstprivate(/ompcommonscalarcopy/)')
            self.fw('tid=omp_get_thread_num()')
            self.fw('if (tid.gt.0) then')
            for s in self.slist:
                if s.name in ompcommonscalar:
                    self.fw('{}={}copy'.format(s.name,s.name))
            # self.fw('if (OMPCopyDebug.gt.0) then')
            if self.ompdebug:
                self.fw("write(*,*) '#OMP::: Thread #',tid,' : copying scalar...'")
            # self.fw('endif')
            self.fw('endif')
            self.fw('!$omp end parallel')
        self.fw('end subroutine OmpCopyScalar{}'.format(self.pkgname))
        self.fw('end module OmpCopy{}'.format(self.pkgname) )  
                 
                
    def ThreadedNullify(self,a):
            if a.threadprivate:
                self.fw('!$omp parallel private(tid)') 
                self.fw('tid=omp_get_thread_num()')
                # self.fw('if (OMPAllocDebug.gt.0) then')
                if self.ompdebug:
                    self.fw("write(*,'(a,i3)') '# Nullifying {} in thread #',tid".format(a.name))
                # self.fw("endif")
                self.fw(' nullify({})'.format(a.name)) 
                self.fw('!$omp end parallel')
            else:
                self.fw(' nullify({})'.format(a.name)) 
            
                
    def ThreadedAssociation(self,a):
            if a.threadprivate:
                self.fw('  integer:: tid,omp_get_thread_num')
                Dim=a.dimstring.count(',')+1
                L=[':' for i in range(Dim)]
                if len(L)==1:
                    Str=L[0]
                else:
                    Str=','.join(L)
                self.fw('  ' + fvars.ftof(a.type) + ', target,allocatable,save:: pcopy__({})'.format(Str))
                self.fw('  ' + fvars.ftof(a.type) + ', pointer::ptop__('+Str+'),ptopcopy__('+Str+')')
            if a.threadprivate:
                self.fw('!$omp threadprivate(pcopy__)')
                # self.fw('if (OMPAllocDebug.gt.1) then')
                if self.ompdebug:
                    self.fw("write(*,*) '##########OMP: association of the pointer: {}'".format(a.name))
                    self.fw("write(*,*) '#OMP: is pcopy allocated:',allocated(pcopy__)")
                # self.fw('endif')
                self.fw('if (.not.allocated(pcopy__)) then')
                self.fw('allocate(pcopy__'+self.prefixdimsf(re.sub('[ \t\n]', '', a.dimstring))+')')                    
                self.fw('endif')
                self.fw('ptop__=>p__')
                self.fw('ptopcopy__=>pcopy__')
                # self.fw('if (OMPAllocDebug.gt.1) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP: Are ptop__,ptopcopy__ associated:',associated(ptop__),associated(ptopcopy__)")
                #self.fw('endif')
                S=self.prefixdimsf(re.sub('[ \t\n]', '', a.dimstring))
                self.fw('if (associated(ptop__)) then')
                self.fw('pcopy__'+self.ProcessDim(S)+'=p__'+self.ProcessDim(S))
                self.fw('endif')
                # self.fw('if (OMPAllocDebug.gt.1) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP: beginning of parallel construct for association'")
                # self.fw('endif')
# pcopy__ must be copied in so eahc thread has an allocated target for the pointer. Note that deallocation called after the parallel construct only deallocte pcopy__ for the master thread but not for the other threads. That way, the py array object corresponding to the memory allocated in the subroutine for the master thread can be freed. But we do not care about memory allocated in the other threads since the pyarray object has not link to these memory locations. Note that the pointer and the target are persistent between references to parallel constructs for each threads. Memory in threads is freed when thread is killed.  
                self.fw('!$omp parallel private(tid) copyin(pcopy__)') 
                self.fw('tid=omp_get_thread_num()')
                self.fw('if (tid.eq.0) then')
                # self.fw('if (OMPAllocDebug.gt.0) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP::: Associating {} in master thread'".format(a.name))
                # self.fw('endif')
                self.fw('  ' + a.name + ' => p__')
                
                self.fw('else')
                # self.fw('if (OMPAllocDebug.gt.0) then')
                if self.ompdebug:
                    self.fw("write(*,'(a,i3)') '#OMP::: Associating {} in thread #',tid".format(a.name))
                # self.fw('endif')
                self.fw('if (associated(ptop__)) then')
                self.fw('  ' + a.name + ' => pcopy__')
                self.fw('else')
                self.fw("nullify({})".format(a.name)) 
                self.fw('endif')
                self.fw('endif')
                # self.fw('if (OMPAllocDebug.gt.1) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP::: Thread #',tid,' : location of {}:',loc({})".format(a.name,a.name))
                # self.fw('endif')
                self.fw('!$omp end parallel')
                # self.fw('if (OMPAllocDebug.gt.1) then')
                if self.ompdebug:
                    self.fw("write(*,*) '#OMP: End of parallel construct for association'")
                # self.fw('endif')
                self.fw('nullify(ptop__)')
                self.fw('nullify(ptopcopy__)') 
                self.fw('if (allocated(pcopy__)) then')
                # if pcopy_ is not deallocated then the freearray c function cannot free the 
                # array pointer pya in gallot or gchange   
                self.fw('deallocate(pcopy__)')                    
                self.fw('endif')
            else:
                self.fw('  ' + a.name + ' => p__')
                
    def DeclareThreadPrivate(self,g):      
        for s in self.slist:
            if s.group == g:                    
                if s.dynamic:
                    if s.threadprivate==1:
                        self.fw('!$omp threadprivate('+s.name+')')
                        self.ListThreadPrivateVars.append(s.name)
                else:
                    if s.threadprivate==1:
                        self.fw('!$omp threadprivate('+s.name+')')
                        self.ListThreadPrivateVars.append(s.name)
            
        for a in self.alist:
            if a.group == g:
                    if a.type == 'character':
                        pass
                    else:
                        if a.threadprivate:
                            self.fw('!$omp threadprivate('+a.name+')')
                            self.ListThreadPrivateVars.append(a.name)
    def PrintListThreadPrivateVars(self):
        with open('ListThreadPrivateVariables_{}.txt'.format(self.pkgname),'w') as f:
            for v in self.ListThreadPrivateVars:
                f.write(v+'\n')
            f.write('total={}\n'.format(len(self.ListThreadPrivateVars)))
    
                            
    def writef90OMPInitHelper(self):
        groupslist=[]
        self.fw('subroutine OmpInitZero{}'.format(self.pkgname))
        for a in self.alist:
            if a.threadprivate and a.type != 'character': 
                # groups to access variables                
                if a.group not in groupslist:
                    groupslist.append(a.group)
                    
        for s in self.slist: 
            if s.threadprivate and s.type != 'character':
                if s.group not in groupslist:
                    groupslist.append(s.group)
                
        for g in groupslist:
            self.fw('  use ' + g)
        self.fw('!$omp parallel')    
        for a in self.alist:
            if a.threadprivate and a.type != 'character':
                self.fw('{}=0'.format(a.name))
        for s in self.slist:
            if s.threadprivate and s.type != 'character':
                self.fw('{}=0'.format(s.name))
        self.fw('!$omp end parallel')        
                
        self.fw('end subroutine OmpInitZero{}'.format(self.pkgname) )
