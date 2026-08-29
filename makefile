#!/usr/bin/make

RESET:=\e[0m
RED:=\e[1;31m
GREEN:=\e[1;32m
BLUE:=\e[1;34m
YELLOW:=\e[1;33m
CYAN:=\e[1;36m
MAGENTA:=\e[1;35
END:=

SHELL := /usr/bin/bash
OS := $(shell uname -s)
HOSTNAME := $(shell hostname)

PROJECT:=aic-foxtrot


CUDAS := \
	$(wildcard *.cu) \
	$(END)
CUDAS := $(sort $(filter-out t%, $(CUDAS)))
ifeq ($(HOSTNAME),blue) # blue has AMD graphics cards.
CUDAS := $(sort $(filter-out 14d-dist-shmem.cu sample-dshmem.cu, $(CUDAS)))
endif


HIPS := \
	$(CUDAS:.cu=.hip) \
	$(wildcard *.hip) \
	$(END)
HIPS := $(sort $(filter-out t%, $(HIPS)))

SRCS := \
	$(wildcard *.cpp) \
	$(CUDAS) \
	$(HIPS) \
	$(END)
SRCS := $(sort $(filter-out t%, $(SRCS)))

ifeq ($(OS),Linux)
OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)
OBJS := $(OBJS:.hip=.o)
OBJS := $(sort $(OBJS))
else # cygwin
OBJS := $(SRCS:.cpp=.obj)
OBJS := $(OBJS:.cu=.obj)
OBJS := $(OBJS:.hip=.obj)
OBJS := $(sort $(OBJS))
endif

ifeq ($(OS),Linux)
EXES := $(OBJS:.o=.exe)
else # cygwin
EXES := $(OBJS:.obj=.exe)
endif

NVCC = nvcc
NVCCFLAGS = \
	-Xcompiler -O2 \
	-Xptxas -O3 \
	-O3 \
	-arch=native \
	-lcublas \
	$(END)

#	-Xcompiler -O2 \
#	-Xptxas -O3 \
#	-O3 \
#	-gencode arch=compute_120,code=sm_120 \
#	-gencode arch=compute_120,code=\"sm_120,compute_120\" \

HIPCC = hipcc
HIPCCFLAGS = \
	-O2 \
	-Wno-unused-value \
	-lhipblas \
	$(END)

ifeq ($(OS),Linux)
CXX = clang++
CXXFLAGS = -O2
LDFLAGS += # -lvulkan -lglfw -lGL -lX11
else # cygwin
endif

.TARGETS:	.cu .hip .cpp .c .obj .exe .comp .spv

ifeq ($(OS),Linux)
ifeq ($(HOSTNAME),blue) # blue has AMD graphics cards.
%.hip:	%.cu
	../hipify-perl -o $@ $^
%.exe:	%.hip
	$(HIPCC) $^ -o $@ $(HIPCCFLAGS)
endif
ifeq ($(HOSTNAME),gold) # gold has NVIDIA graphics cards.
%.exe:	%.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)
endif
%.o:	%.cpp
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
%.o:	%.c
	$(COMPILE.c) $(OUTPUT_OPTION) $<
%.exe:	%.o
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@
%.exe:	%.cpp
	$(LINK.cc) $^ $(LOADLIBES) $(LDLIBS) -o $@
%.comp.spv:	%.comp
	glslc -o $@ $^
else # cygwin
%.obj:	%.cpp
	$(COMPILE.c) /c /Fo$@ $<
%.obj:	%.c
	$(COMPILE.c) /c /TC /Fo$@ $<
%.exe:	%.obj
	$(LINK.obj) $^ $(LOADLIBES) $(LDLIBS) /OUT:$@
endif


ifeq ($(HOSTNAME),blue)
default:	$(HIPS) $(EXES) 
else 
default:	$(EXES) 
endif

show:
	@echo "CUDAS = $(CUDAS)"
	@echo "HIPS = $(HIPS)"
	@echo "SRCS = $(SRCS)"
	@echo "OBJS = $(OBJS)"
	@echo "EXES = $(EXES)"

clean:
	/bin/rm -f *.exp *.ilk *.pdb *.idb *.iobj *.ipdb
	/bin/rm -f *.obj
	/bin/rm -f $(OBJS)

clobber: clean
	/bin/rm -f *.exe *.exe.manifest
	/bin/rm -f $(EXES)

# common commands
#
wc:
	wc $(SRCS)

ident:
	ident $(SRCS)

rcsinit:
	@for src in $(SRCS) ; do \
		if [ ! -f RCS/$${src},v ] ; then\
			rcs -i -U $${src} ; \
			ci -f -u $${src} ; \
		fi \
	done

rcsdiff:
	@for src in $(SRCS) ; do \
		if ! rcsdiff -q $${src} ; then \
			echo "$${src}: different from previous version" ; \
		fi \
	done

ci:
	@for src in $(SRCS) ; do \
		if [ ! -f RCS/$${src},v ] ; then \
			rcs -i -U $${src} ; \
			ci -f -u $${src} ; \
		fi ; \
		if ! rcsdiff -q $${src} ; then \
			ci -f -u $${src} ; \
		fi \
	done
	

__dontuse_rcsinit:
	rcs -i -U $(SRCS)

__dontuse_ci:
	ci -f -u $(SRCS)

__dontuse_co:
	co -f -u -r1.5 $(SRCS)

