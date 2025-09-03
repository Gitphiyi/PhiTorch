# TO BUILD CMAKE: cmake -S . -B build
# make -j //to compile in build folder
cmake_minimum_required(VERSION 3.20)
project(PhiTorch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

# kernel compilation
set(SHADER_DIR ${PROJECT_SOURCE_DIR}/src/kernels)
set(TENSOR_METAL ${SHADER_DIR}/tensor.metal)
set(TENSOR_AIR   ${CMAKE_BINARY_DIR}/tensor.air)
set(TENS_METALLIB ${CMAKE_BINARY_DIR}/tens.metallib)

add_custom_command(
  OUTPUT ${TENSOR_AIR}
  COMMAND xcrun -sdk macosx metal -std=metal3.0
          -c ${TENSOR_METAL}
          -o ${TENSOR_AIR}
          -I ${SHADER_DIR}                  # if you #include other .metal headers
  DEPENDS ${TENSOR_METAL}
  BYPRODUCTS ${TENSOR_AIR}
  COMMENT "Compiling Metal shader: tensor.metal -> tensor.air"
  VERBATIM
)
add_custom_command(
  OUTPUT ${TENS_METALLIB}
  COMMAND xcrun -sdk macosx metallib ${TENSOR_AIR} -o ${TENS_METALLIB}
  DEPENDS ${TENSOR_AIR}
  BYPRODUCTS ${TENS_METALLIB}
  COMMENT "Linking Metal library: tens.metallib"
  VERBATIM
)
add_custom_target(tens_metallib ALL DEPENDS ${TENS_METALLIB})

add_custom_command(
  TARGET tens_metallib POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${TENS_METALLIB}" "${CMAKE_BINARY_DIR}/examples/tens.metallib"
  COMMENT "Copy tens.metallib to build/examples/"
)


# metal lib
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE
  ${PROJECT_SOURCE_DIR}/extern/metal-cpp
)
find_library(FW_METAL       Metal)
find_library(FW_FOUNDATION  Foundation)
find_library(FW_QUARTZCORE  QuartzCore)
find_library(FW_METALKIT    MetalKit)

target_link_libraries(metal-cpp INTERFACE
  ${FW_METAL}
  ${FW_FOUNDATION}
  ${FW_QUARTZCORE}
  ${FW_METALKIT}
)
# tens lib
add_library( tens STATIC 
    src/tensor/Tensor.cpp
    src/GpuScheduler.cpp
)
target_include_directories(tens PUBLIC ${PROJECT_SOURCE_DIR}/include) #directories refers to folder so any #include refers to the include folder
target_link_libraries(tens PUBLIC metal-cpp) 
add_dependencies(tens tens_metallib)

target_compile_definitions(tens PUBLIC
  TENS_METALLIB_PATH="${CMAKE_BINARY_DIR}/tens.metallib"
)

# add_executable(main main.cpp)
# target_link_libraries(main PRIVATE tens)

add_subdirectory(examples)

# --- Testing ---
# RUN TESTS: ctest
# RUN SPECIFIC TEST: ctest -R compiler_tests //only run compiler tests. follows name in file not filename
# include(CTest)          # sets BUILD_TESTING option
# if(BUILD_TESTING)
#   add_subdirectory(tests)
# endif()