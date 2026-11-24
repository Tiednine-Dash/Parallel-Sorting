# Compiler
CXX := nvcc

# Compiler flags
CXXFLAGS := -O3 -Xptxas -O3 -Xcompiler -Wall,-Werror,-fopenmp -std=c++23
# TESTFLAGS := $(CXXFLAGS) -DTESTING

# Linker
LINK := $(CXX)

# Libraries
LDLIBS := -lgomp

# Executables
EXEC := sorts

#############################################################
# Main program
#############################################################

$(EXEC) : $(EXEC).o
	$(LINK) $^ $(LDLIBS) -o $@

$(EXEC).o : $(EXEC).cu Timer.hpp
	$(CXX) $(CXXFLAGS) -c $(EXEC).cu


#############################################################

.PHONY : clean

clean :
	@$(RM) $(EXEC) $(TEST)
	@$(RM) *.o *.d *~
	@echo "Cleanup complete."
