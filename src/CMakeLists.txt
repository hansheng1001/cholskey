# 编译出cholesky
add_executable(cholesky cholesky.cu)

# target_link_libraries(cholesky my_api_shared_lib)

target_include_directories(cholesky PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
target_include_directories(cholesky PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
target_include_directories(cholesky PRIVATE ${CMAKE_SOURCE_DIR}/inc)

target_link_libraries(cholesky curand cusolver cublas)