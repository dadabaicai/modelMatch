cmake_minimum_required(VERSION 2.8...4.1.1)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)


# debug or release
SET(CMAKE_BUILD_TYPE "Release")


if(MSVC)
    # Enable the fused AVX2 response-map path and OpenMP row parallelism.
    add_compile_options(/EHsc /utf-8 /arch:AVX2)
else()
    # arm or x86
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86")
        set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
    else()
        set(PLATFORM_COMPILE_FLAGS "-march=native")
    endif()

    set(COMMON_COMPILE_FLAGS "-Wall -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

find_package(OpenMP REQUIRED)


# opencv
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} E:/cwn/OpenCV/build/install/x64/vc17/lib)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")


# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp speedUp.cpp test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

