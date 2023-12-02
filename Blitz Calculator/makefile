# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++11

# Source files and headers
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)
HDRS := $(wildcard *.h)

# Target executable
TARGET = compiled.exe

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)

# Rule to build .o files
%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Clean target
clean:
	del /Q $(OBJS) $(TARGET) *.exe