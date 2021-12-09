# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
# when hislicon SDK was installed, toolchain was installed in the path as below:
set(CMAKE_C_COMPILER /opt/linux/x86-arm/aarch64-mix210-linux/bin/aarch64-mix210-linux-gcc)
set(CMAKE_CXX_COMPILER /opt/linux/x86-arm/aarch64-mix210-linux/bin/aarch64-mix210-linux-g++)

find_path(GCC_PATH gcc)
find_path(GXX_PATH g++)
if(NOT ${GCC_PATH} STREQUAL "GCC_PATH-NOTFOUND" AND NOT ${GXX_PATH} STREQUAL "GXX_PATH-NOTFOUND")
    set(FLATC_GCC_COMPILER ${GCC_PATH}/gcc)
    set(FLATC_GXX_COMPILER ${GXX_PATH}/g++)
endif()

# set searching rules for cross-compiler
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

#set(CMAKE_CXX_FLAGS "-march= -mfloat-abi=softfp -mfpu=neon-vfpv4  ${CMAKE_CXX_FLAGS}")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
