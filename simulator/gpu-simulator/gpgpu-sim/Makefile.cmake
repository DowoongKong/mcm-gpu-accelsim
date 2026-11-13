# GPGPU-Sim Makefile - Wrapper for CMake build system
# This forwards all make commands to the build directory

.PHONY: all clean install debug distclean

BUILD_DIR := build

all:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Creating build directory..."; \
		mkdir -p $(BUILD_DIR); \
		cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release; \
	fi
	@cd $(BUILD_DIR) && $(MAKE)

debug:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Creating build directory..."; \
		mkdir -p $(BUILD_DIR); \
		cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Debug; \
	fi
	@cd $(BUILD_DIR) && $(MAKE)

clean:
	@if [ -d "$(BUILD_DIR)" ]; then \
		cd $(BUILD_DIR) && $(MAKE) clean; \
	fi

distclean:
	@rm -rf $(BUILD_DIR)

install:
	@cd $(BUILD_DIR) && $(MAKE) install
