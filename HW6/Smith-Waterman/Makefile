CXX = g++ 
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3

NVCXX = nvcc 
NVCXXFLAGS = -g --ptxas-options=-v -std=c++11  -O3


CUDALIB = /usr/local/cuda/lib64
CUDAINC = /usr/local/cuda/include
SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all:  bin/sw

bin:
	mkdir -p bin


bin/sw: $(OBJS) $(CUOBJS) driver/driver.o
	mkdir -p bin
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
			    @echo "Compiled "$<" successfully!"


.PHONY:	test clean

$(OBJS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
			mkdir -p obj
	    @echo $(CXX) $(CXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"


$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
			mkdir -p cuobj
	    @echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"

clean: 
	rm -f $(CUOBJS) $(OBJS) 
	rm -rf bin/*

#########################
# Submit
##########################
run: ./bin/sw
	mkdir -p result
	./bin/sw -p -t /nfs/home/mgp2021_data/sw/query_2.fasta /nfs/home/mgp2021_data/sw/target_2.fasta > result/local.out


submit_small:
	mkdir -p result
	condor_submit sw_small.cmd

submit_mid:
	mkdir -p result
	condor_submit sw_mid.cmd

submit_large:
	mkdir -p result
	condor_submit sw_large.cmd
