# Complete CMake Expert Guide

## Table of Contents
1. [Introduction to CMake](#introduction)
2. [Basic Concepts](#basic-concepts)
3. [Your First CMake Project](#first-project)
4. [Variables and Properties](#variables-properties)
5. [Targets and Dependencies](#targets-dependencies)
6. [Finding and Using Libraries](#finding-libraries)
7. [Custom Commands and Targets](#custom-commands)
8. [Generator Expressions](#generator-expressions)
9. [Testing with CTest](#testing)
10. [Packaging with CPack](#packaging)
11. [Advanced Topics](#advanced-topics)
12. [Best Practices](#best-practices)
13. [Common Patterns](#common-patterns)
14. [Troubleshooting](#troubleshooting)

---

## 1. Introduction to CMake {#introduction}

CMake is a cross-platform build system generator. It generates native build files (Makefiles, Visual Studio projects, Xcode projects, etc.) from platform-independent CMakeLists.txt files.

### Key Benefits:
- **Cross-platform**: Works on Windows, macOS, Linux, and more
- **IDE-agnostic**: Generates files for various IDEs and build systems
- **Dependency management**: Handles complex dependency graphs
- **Testing integration**: Built-in testing framework (CTest)
- **Packaging support**: Create installers and packages (CPack)

### Build Process Overview:
```
CMakeLists.txt → CMake → Build System Files → Build Tool → Executable/Library
```

---

## 2. Basic Concepts {#basic-concepts}

### Key Terms:
- **Source Tree**: Directory containing source code and CMakeLists.txt files
- **Build Tree**: Directory where build artifacts are created
- **Target**: Something that can be built (executable, library, custom target)
- **Generator**: Tool that creates build system files (Unix Makefiles, Ninja, Visual Studio)
- **Configuration**: Debug, Release, RelWithDebInfo, MinSizeRel

### CMake Commands Structure:
```cmake
command(ARGUMENT1 ARGUMENT2 ...)
command(TARGET_NAME
    KEYWORD1 arg1 arg2
    KEYWORD2 arg3
)
```

---

## 3. Your First CMake Project {#first-project}

### Simple Executable Example:

**Directory structure:**
```
my_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── math_utils.cpp
│   └── math_utils.h
└── build/           # Create this for out-of-source builds
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.16)

# Project declaration
project(MyProject
    VERSION 1.0.0
    DESCRIPTION "My first CMake project"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add executable target
add_executable(my_app
    src/main.cpp
    src/math_utils.cpp
)

# Add include directory
target_include_directories(my_app PRIVATE src)
```

**src/main.cpp:**
```cpp
#include <iostream>
#include "math_utils.h"

int main() {
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;
    return 0;
}
```

**src/math_utils.h:**
```cpp
#pragma once
int add(int a, int b);
int multiply(int a, int b);
```

**src/math_utils.cpp:**
```cpp
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

### Building the Project:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

---

## 4. Variables and Properties {#variables-properties}

### Variable Types:
1. **Normal Variables**: Scoped to current directory
2. **Cache Variables**: Persist across cmake runs
3. **Environment Variables**: From system environment

### Setting Variables:
```cmake
# Normal variable
set(MY_VAR "value")
set(MY_LIST "item1" "item2" "item3")

# Cache variable (appears in CMake GUI)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

# Environment variable
set(ENV{PATH} "/new/path:$ENV{PATH}")
```

### Using Variables:
```cmake
# Variable substitution
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
message(STATUS "Source directory: ${SOURCE_DIR}")

# List operations
list(APPEND MY_LIST "item4")
list(LENGTH MY_LIST LIST_SIZE)
```

### Important Built-in Variables:
```cmake
# Directories
CMAKE_SOURCE_DIR          # Top-level source directory
CMAKE_BINARY_DIR          # Top-level build directory
CMAKE_CURRENT_SOURCE_DIR  # Current CMakeLists.txt directory
CMAKE_CURRENT_BINARY_DIR  # Current build directory

# Compiler and flags
CMAKE_CXX_COMPILER        # C++ compiler path
CMAKE_CXX_FLAGS           # C++ compiler flags
CMAKE_BUILD_TYPE          # Debug, Release, etc.

# System information
CMAKE_SYSTEM_NAME         # Operating system
CMAKE_SYSTEM_PROCESSOR    # Target processor
```

### Properties:
```cmake
# Set property on target
set_target_properties(my_target PROPERTIES
    CXX_STANDARD 17
    OUTPUT_NAME "custom_name"
)

# Get property
get_target_property(STANDARD my_target CXX_STANDARD)
```

---

## 5. Targets and Dependencies {#targets-dependencies}

### Target Types:

#### Executable Targets:
```cmake
add_executable(my_app src/main.cpp)

# Conditional compilation
if(WIN32)
    target_sources(my_app PRIVATE src/windows_specific.cpp)
elseif(UNIX)
    target_sources(my_app PRIVATE src/unix_specific.cpp)
endif()
```

#### Library Targets:
```cmake
# Static library
add_library(math_lib STATIC
    src/math_utils.cpp
    src/advanced_math.cpp
)

# Shared library
add_library(graphics_lib SHARED
    src/renderer.cpp
    src/shader.cpp
)

# Interface library (header-only)
add_library(header_lib INTERFACE)
target_include_directories(header_lib INTERFACE include)
```

### Target Properties:
```cmake
# Include directories
target_include_directories(my_target
    PUBLIC include          # Consumers get these
    PRIVATE src            # Only this target gets these
    INTERFACE interface    # Only consumers get these
)

# Compile definitions
target_compile_definitions(my_target
    PUBLIC -DAPI_VERSION=2
    PRIVATE -DDEBUG_MODE
)

# Compile options
target_compile_options(my_target
    PRIVATE -Wall -Wextra
    PUBLIC -fPIC
)

# Link libraries
target_link_libraries(my_target
    PUBLIC math_lib        # Links and propagates to dependents
    PRIVATE graphics_lib   # Links but doesn't propagate
    INTERFACE boost::headers  # Only propagates to dependents
)
```

### Modern CMake Target Usage:
```cmake
# Create a library with modern interface
add_library(mylib
    src/mylib.cpp
    src/internal.cpp
)

# Set up the interface
target_include_directories(mylib
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src
)

target_compile_features(mylib PUBLIC cxx_std_17)

# Create executable that uses the library
add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE mylib)
```

---

## 6. Finding and Using Libraries {#finding-libraries}

### Using find_package():
```cmake
# Find required packages
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Use the found packages
target_link_libraries(my_app PRIVATE
    Boost::system
    Boost::filesystem
    ${OpenCV_LIBS}
    Threads::Threads
)
```

### pkg-config Integration:
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

target_link_libraries(my_app PRIVATE ${GTK3_LIBRARIES})
target_include_directories(my_app PRIVATE ${GTK3_INCLUDE_DIRS})
target_compile_options(my_app PRIVATE ${GTK3_CFLAGS_OTHER})
```

### Custom Find Modules:
**cmake/FindMyLib.cmake:**
```cmake
# Find MyLib
find_path(MYLIB_INCLUDE_DIR mylib.h
    PATHS /usr/local/include /opt/mylib/include
)

find_library(MYLIB_LIBRARY
    NAMES mylib
    PATHS /usr/local/lib /opt/mylib/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MyLib DEFAULT_MSG
    MYLIB_LIBRARY MYLIB_INCLUDE_DIR
)

if(MYLIB_FOUND)
    if(NOT TARGET MyLib::MyLib)
        add_library(MyLib::MyLib UNKNOWN IMPORTED)
        set_target_properties(MyLib::MyLib PROPERTIES
            IMPORTED_LOCATION "${MYLIB_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MYLIB_INCLUDE_DIR}"
        )
    endif()
endif()
```

### FetchContent (CMake 3.14+):
```cmake
include(FetchContent)

# Fetch from Git
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

# Fetch from URL
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)

FetchContent_MakeAvailable(googletest json)

# Use the fetched dependencies
target_link_libraries(my_test PRIVATE gtest_main)
target_link_libraries(my_app PRIVATE nlohmann_json::nlohmann_json)
```

---

## 7. Custom Commands and Targets {#custom-commands}

### Custom Commands:
```cmake
# Generate a file during build
add_custom_command(OUTPUT generated_file.cpp
    COMMAND ${CMAKE_COMMAND} -E echo "Generated content" > generated_file.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating source file"
    VERBATIM
)

# Add generated file to target
add_executable(my_app main.cpp generated_file.cpp)

# Run command after building target
add_custom_command(TARGET my_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
        ${CMAKE_CURRENT_SOURCE_DIR}/config.ini
        $<TARGET_FILE_DIR:my_app>
    COMMENT "Copying configuration file"
)
```

### Custom Targets:
```cmake
# Custom target that always runs
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
)

# Custom target with dependencies
add_custom_target(docs
    COMMAND doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    DEPENDS my_app
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation"
)
```

---

## 8. Generator Expressions {#generator-expressions}

Generator expressions are evaluated during build system generation and provide powerful conditional logic:

### Common Generator Expressions:
```cmake
# Configuration-based
target_compile_definitions(my_app PRIVATE
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release>:RELEASE_BUILD>
)

# Platform-based
target_sources(my_app PRIVATE
    src/main.cpp
    $<$<PLATFORM_ID:Windows>:src/windows.cpp>
    $<$<PLATFORM_ID:Linux>:src/linux.cpp>
)

# Compiler-based
target_compile_options(my_app PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Target properties
target_include_directories(mylib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Boolean logic
target_compile_definitions(my_app PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:GNU_DEBUG>
)
```

### Advanced Generator Expressions:
```cmake
# Join lists
set(SOURCES file1.cpp file2.cpp file3.cpp)
target_sources(my_app PRIVATE $<JOIN:${SOURCES}, >)

# Filter lists
set(ALL_LIBS lib1 lib2_debug lib3)
target_link_libraries(my_app PRIVATE 
    $<$<NOT:$<CONFIG:Debug>>:$<FILTER:${ALL_LIBS},EXCLUDE,.*_debug>>
)

# Target file operations
add_custom_command(TARGET my_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:my_app>
        ${CMAKE_CURRENT_BINARY_DIR}/backup/$<TARGET_FILE_NAME:my_app>
)
```

---

## 9. Testing with CTest {#testing}

### Basic Testing Setup:
```cmake
# Enable testing
enable_testing()

# Simple test
add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE math_lib)

add_test(NAME math_test COMMAND test_math)
```

### Advanced Testing:
```cmake
# Test with arguments
add_test(NAME parametric_test 
    COMMAND my_test --input ${CMAKE_CURRENT_SOURCE_DIR}/test_data.txt
)

# Test properties
set_tests_properties(math_test PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All tests passed"
    FAIL_REGULAR_EXPRESSION "FAILED|ERROR"
    ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}"
)

# Conditional tests
if(BUILD_PERFORMANCE_TESTS)
    add_test(NAME performance_test COMMAND performance_benchmark)
    set_tests_properties(performance_test PROPERTIES
        LABELS "performance;slow"
        TIMEOUT 300
    )
endif()
```

### Google Test Integration:
```cmake
find_package(GTest REQUIRED)

add_executable(unit_tests
    tests/test_math.cpp
    tests/test_string.cpp
)

target_link_libraries(unit_tests PRIVATE
    GTest::gtest_main
    my_lib_under_test
)

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(unit_tests)
```

### Running Tests:
```bash
# Build and run all tests
cmake --build . --target test

# Or use ctest directly
ctest

# Run specific tests
ctest -R "math.*"  # Run tests matching pattern
ctest -L "unit"    # Run tests with specific label
ctest -j 4         # Run tests in parallel
```

---

## 10. Packaging with CPack {#packaging}

### Basic CPack Setup:
```cmake
# Set package information
set(CPACK_PACKAGE_NAME "MyApplication")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "My awesome application")
set(CPACK_PACKAGE_VENDOR "My Company")

# Installation rules
install(TARGETS my_app
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES config.ini DESTINATION etc)
install(DIRECTORY docs/ DESTINATION share/doc)

# Include CPack
include(CPack)
```

### Platform-specific Packaging:
```cmake
# Windows NSIS installer
if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "My Application")
    set(CPACK_NSIS_PACKAGE_NAME "MyApp")
    set(CPACK_NSIS_CONTACT "support@mycompany.com")
endif()

# Linux packages
if(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR "DEB;RPM")
    
    # Debian package
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "maintainer@mycompany.com")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-all-dev")
    
    # RPM package
    set(CPACK_RPM_PACKAGE_REQUIRES "boost-devel")
endif()

# macOS bundle
if(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_FORMAT "UDBZ")
endif()
```

### Creating Packages:
```bash
# Build packages
cmake --build . --target package

# Or use cpack directly
cpack -G ZIP     # Create ZIP archive
cpack -G DEB     # Create Debian package
cpack -G NSIS    # Create Windows installer
```

---

## 11. Advanced Topics {#advanced-topics}

### CMake Modules and Functions:

**cmake/MyUtilities.cmake:**
```cmake
# Function definition
function(add_library_with_alias target_name)
    set(options SHARED STATIC INTERFACE)
    set(oneValueArgs ALIAS NAMESPACE)
    set(multiValueArgs SOURCES HEADERS)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Determine library type
    if(ARG_SHARED)
        set(lib_type SHARED)
    elseif(ARG_STATIC)
        set(lib_type STATIC)
    elseif(ARG_INTERFACE)
        set(lib_type INTERFACE)
    else()
        set(lib_type "")
    endif()
    
    # Create library
    add_library(${target_name} ${lib_type} ${ARG_SOURCES} ${ARG_HEADERS})
    
    # Create alias
    if(ARG_ALIAS)
        set(alias_name ${ARG_ALIAS})
    else()
        set(alias_name ${target_name})
    endif()
    
    if(ARG_NAMESPACE)
        add_library(${ARG_NAMESPACE}::${alias_name} ALIAS ${target_name})
    endif()
endfunction()

# Macro definition (different scoping rules)
macro(set_compiler_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endmacro()
```

**Usage:**
```cmake
include(cmake/MyUtilities.cmake)

add_library_with_alias(math_utils STATIC
    SOURCES src/math.cpp
    HEADERS include/math.h
    ALIAS math
    NAMESPACE MyProject
)

set_compiler_warnings(math_utils)
# Now you can use MyProject::math
```

### Export and Import:

**Creating exportable targets:**
```cmake
# Create library
add_library(MyLib src/mylib.cpp)
target_include_directories(MyLib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install library and headers
install(TARGETS MyLib
    EXPORT MyLibTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Export targets
install(EXPORT MyLibTargets
    FILE MyLibTargets.cmake
    NAMESPACE MyLib::
    DESTINATION lib/cmake/MyLib
)

# Create config file
configure_package_config_file(
    cmake/MyLibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MyLibConfig.cmake
    INSTALL_DESTINATION lib/cmake/MyLib
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MyLibConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MyLibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MyLibConfigVersion.cmake
    DESTINATION lib/cmake/MyLib
)
```

**Using exported targets:**
```cmake
find_package(MyLib REQUIRED)
target_link_libraries(my_app PRIVATE MyLib::MyLib)
```

### Precompiled Headers (CMake 3.16+):
```cmake
# Create precompiled header
target_precompile_headers(my_app PRIVATE
    <iostream>
    <vector>
    <string>
    <memory>
    "common.h"
)

# Reuse PCH across targets
target_precompile_headers(another_target REUSE_FROM my_app)
```

### Unity Builds (CMake 3.16+):
```cmake
# Enable unity build
set_target_properties(my_app PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 8
)
```

---

## 12. Best Practices {#best-practices}

### 1. Modern CMake (3.15+) Principles:
```cmake
# ✅ Good - Target-centric approach
add_library(mylib src/mylib.cpp)
target_include_directories(mylib PUBLIC include)
target_compile_features(mylib PUBLIC cxx_std_17)

# ❌ Bad - Directory-wide settings
include_directories(include)
set(CMAKE_CXX_STANDARD 17)
```

### 2. Proper Variable Usage:
```cmake
# ✅ Good - Scoped variables
function(my_function)
    set(LOCAL_VAR "value")  # Function scope
    set(PARENT_VAR "value" PARENT_SCOPE)  # Parent scope
endfunction()

# ❌ Bad - Global variables everywhere
set(GLOBAL_VAR "value")  # Pollutes global namespace
```

### 3. Generator Expression Usage:
```cmake
# ✅ Good - Precise conditions
target_compile_definitions(my_app PRIVATE
    $<$<CONFIG:Debug>:DEBUG_MODE>
)

# ❌ Bad - Build-time checks only
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(my_app PRIVATE DEBUG_MODE)
endif()
```

### 4. Proper Dependency Management:
```cmake
# ✅ Good - Use targets
find_package(Boost REQUIRED COMPONENTS system)
target_link_libraries(my_app PRIVATE Boost::system)

# ❌ Bad - Manual paths and flags
target_link_libraries(my_app PRIVATE ${Boost_LIBRARIES})
target_include_directories(my_app PRIVATE ${Boost_INCLUDE_DIRS})
```

### 5. Project Structure:
```
project/
├── CMakeLists.txt          # Main CMake file
├── cmake/                  # CMake modules and utilities
│   ├── FindCustomLib.cmake
│   └── ProjectUtilities.cmake
├── src/                    # Source files
│   └── CMakeLists.txt      # Source-specific CMake
├── include/                # Public headers
│   └── myproject/
├── tests/                  # Test files
│   └── CMakeLists.txt      # Test-specific CMake
├── docs/                   # Documentation
├── examples/               # Usage examples
└── build/                  # Build directory (not in VCS)
```

---

## 13. Common Patterns {#common-patterns}

### 1. Header-Only Library:
```cmake
add_library(header_only_lib INTERFACE)
target_include_directories(header_only_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(header_only_lib INTERFACE cxx_std_17)
```

### 2. Conditional Compilation:
```cmake
option(ENABLE_FEATURE_X "Enable feature X" ON)
option(BUILD_TESTS "Build test suite" OFF)

if(ENABLE_FEATURE_X)
    target_compile_definitions(my_app PRIVATE FEATURE_X_ENABLED)
    target_sources(my_app PRIVATE src/feature_x.cpp)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### 3. Compiler-Specific Settings:
```cmake
# Set warnings based on compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(my_app PRIVATE
        -Wall -Wextra -Wpedantic -Werror
        $<$<CONFIG:Debug>:-g3 -O0>
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
    )
elseif(MSVC)
    target_compile_options(my_app PRIVATE
        /W4 /WX
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
endif()
```

### 4. Version Information:
```cmake
# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.h
    @ONLY
)

# version.h.in content:
# #define PROJECT_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
# #define PROJECT_VERSION_MINOR @PROJECT_VERSION_MINOR@
# #define PROJECT_VERSION_PATCH @PROJECT_VERSION_PATCH@
# #define PROJECT_VERSION_STRING "@PROJECT_VERSION@"
```

### 5. Resource Embedding:
```cmake
# Function to embed resources
function(embed_resource target resource_file)
    get_filename_component(resource_name ${resource_file} NAME_WE)
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${resource_name}_resource.cpp)
    
    add_custom_command(OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND}
        -DRESOURCE_FILE=${resource_file}
        -DOUTPUT_FILE=${output_file}
        -DRESOURCE_NAME=${resource_name}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedResource.cmake
        DEPENDS ${resource_file}
        COMMENT "Embedding ${resource_file}"
    )
    
    target_sources(${target} PRIVATE ${output_file})
endfunction()
```

---

## 14. Troubleshooting {#troubleshooting}

### Common Issues and Solutions:

#### 1. Target Not Found:
```cmake
# Error: Cannot specify link libraries for target "my_app" which is not built by this project
# Solution: Make sure target exists and is defined in current scope
add_executable(my_app src/main.cpp)  # Define before using
target_link_libraries(my_app PRIVATE some_lib)
```

#### 2. Include Path Issues:
```cmake
# ❌ Wrong - Absolute paths
target_include_directories(my_app PRIVATE /absolute/path/to/headers)

# ✅ Correct - Relative or generator expressions
target_include_directories(my_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
```

#### 3. Library Linking Problems:
```cmake
# Check if library was found
find_package(MyLib)
if(NOT MyLib_FOUND)
    message(FATAL_ERROR "MyLib not found!")
endif()

# Use imported targets when available
if(TARGET MyLib::MyLib)
    target_link_libraries(my_app PRIVATE MyLib::MyLib)
else()
    target_link_libraries(my_app PRIVATE ${MyLib_LIBRARIES})
    target_include_directories(my_app PRIVATE ${MyLib_INCLUDE_DIRS})
endif()
```

#### 4. Cross-Platform Issues:
```cmake
# Handle different file extensions
if(WIN32)
    set(EXECUTABLE_EXTENSION ".exe")
    set(LIBRARY_PREFIX "")
    set(LIBRARY_EXTENSION ".dll")
else()
    set(EXECUTABLE_EXTENSION "")
    set(LIBRARY_PREFIX "lib")
    set(LIBRARY_EXTENSION ".so")
endif()
```

### Debugging Techniques:

#### 1. Print Variables:
```cmake
# Print variable value
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# Print all variables
get_cmake_property(variable_names VARIABLES)
foreach(variable_name ${variable_names})
    message(STATUS "${variable_name}: ${${variable_name}}")
endforeach()
```

#### 2. Debug Target Properties:
```cmake
# Print target properties
get_target_property(includes my_app INCLUDE_DIRECTORIES)
message(STATUS "my_app includes: ${includes}")

get_target_property(libs my_app LINK_LIBRARIES)
message(STATUS "my_app libraries: ${libs}")
```

#### 3. Verbose Output:
```bash
# Verbose build
cmake --build . --verbose

# Or with make
make VERBOSE=1

# Show compile commands
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
```

---

## Advanced Tips and Tricks

### 1. CMake Cache Manipulation:
```cmake
# Force cache values
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# Clear cache entry
unset(OLD_VARIABLE CACHE)
```

### 2. Custom Properties:
```cmake
# Define custom property
define_property(TARGET PROPERTY MY_CUSTOM_PROPERTY
    BRIEF_DOCS "My custom property"
    FULL_DOCS "Detailed description of my property"
)

# Set and get custom property
set_target_properties(my_app PROPERTIES MY_CUSTOM_PROPERTY "value")
get_target_property(value my_app MY_CUSTOM_PROPERTY)
```

### 3. Configure-time vs Build-time:
```cmake
# Configure time - runs when cmake is called
message(STATUS "Configuring project")
file(GLOB sources "src/*.cpp")

# Build time - runs when building
add_custom_command(OUTPUT timestamp.txt
    COMMAND ${CMAKE_COMMAND} -E echo "Built at: $<TIMESTAMP>" > timestamp.txt
    VERBATIM
)
```

This comprehensive guide covers most aspects of CMake from beginner to expert level. Practice with real projects and gradually implement more advanced features as you become comfortable with the basics. The key to CMake mastery is understanding the target-based approach and leveraging modern CMake features for maintainable, cross-platform build systems.