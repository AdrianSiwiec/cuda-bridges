CXX=g++
CXXFLAGS=-O2 -std=c++11 
LDFLAGS=
SOURCES=main.cpp hello.cpp factorial.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=hello

all: cpu-bridges networkrepository-parser

cpu-bridges: src/cpu/cpu-bridges.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

networkrepository-parser: networkrepository-parser.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf cpu-bridges networkrepository-parser
