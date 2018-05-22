CXX=g++
CXXFLAGS=-O2 -std=c++11 
LDFLAGS=
SOURCES=main.cpp hello.cpp factorial.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=hello

all: cpu-bridges

cpu-bridges: src/cpu/cpu-bridges.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf cpu-bridges
