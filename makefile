## Author: Jesse Flores

# Define the C++ compiler to use
CXX = g++

# CXXFLAGS for a STATIC build.
CXXFLAGS = -std=c++17 -O2 -Wall -DSFML_STATIC

# SFML PATH
SFML_PATH = ./sfml

# FLAGS & LIBRARIES for SFML 3 (STATIC Debug Build)
CXXFLAGS += -I$(SFML_PATH)/include

# Linker flags for a STATIC debug build.
# This links the '-s-d' libraries and all their required dependencies.
LDFLAGS = -L$(SFML_PATH)/lib \
          -lsfml-graphics-s-d \
          -lsfml-window-s-d \
          -lsfml-system-s-d \
          -lfreetyped \
          -lopengl32 \
          -lgdi32 \
          -lwinmm

# TARGETS
TARGET_TRAIN = train
TARGET_GUI = gui

# SOURCE & OBJECT FILES
TRAIN_SRCS = main.cpp neural_network.cpp mnist_reader.cpp
TRAIN_OBJS = $(TRAIN_SRCS:.cpp=.o)

GUI_SRCS = gui.cpp neural_network.cpp
GUI_OBJS = $(GUI_SRCS:.cpp=.o)

# Default target
all: $(TARGET_TRAIN)

# BUILD RULES
$(TARGET_TRAIN): $(TRAIN_OBJS)
	@echo "Linking Training App (Static)..."
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

gui: $(GUI_OBJS)
	@echo "Linking GUI App (Static)..."
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	-del /f /q $(TARGET_TRAIN).exe $(TARGET_GUI).exe *.o 2>nul