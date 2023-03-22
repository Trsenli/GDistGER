# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_BUILD_TYPE=Release")

#============  Release 版本一定要把这两个加上！ =============
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# add_app_exec(huge_walk)
add_cuda_exec(huge_walk)
#add_app_exec(biased_walk)
#add_app_exec(dynamic_walk)
