set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

add_executable(tensor tensor.cpp)
target_link_libraries(tensor PRIVATE tens)
target_include_directories(tensor PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_executable(nn nn.cpp)
target_link_libraries(nn PRIVATE tens)
target_include_directories(nn PRIVATE "${PROJECT_SOURCE_DIR}/include")
