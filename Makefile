CC := g++
CC_FLAGS := -Wall -Wextra -std=c -I$(SRC_DIR) -g -fsanitize=address
BUILD_DIR := ./build
SRC_DIR := ./src
#SRCS := $(wildcard $(SRC_DIR)/*.cpp)
SRCS := $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
DEPS := $(OBJECTS:.o=.d)
TARGET := demo

run: $(TARGET)
	@echo "Running... ./$(TARGET)"
	@./$(TARGET)
hello:
	@echo "hi"

compile: $(OBJS) 
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $(OBJS)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CC_FLAGS) -c  -o $@ $<

$(TARGET): $(OBJS)
	$(CC) $(CC_FLAGS) -o $@ $^

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)/*.o
	@rm -rf build/*/
	@rm -rf $(TARGET)