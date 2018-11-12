set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp")
#include_directories(${CMAKE_SOURCE_DIR}/../../src)
include_directories(${CMAKE_SOURCE_DIR}/src/sqlite)
include_directories(${CMAKE_SOURCE_DIR}/src/ncnn)
include_directories(${CMAKE_SOURCE_DIR}/src/)
FIND_LIBRARY(SQLITE_LIB sqlite3  ${CMAKE_SOURCE_DIR}/lib/)
FIND_LIBRARY(DLIB dlib ${CMAKE_SOURCE_DIR}/lib/)
FIND_LIBRARY(ncnn_LIB ncnn ${CMAKE_SOURCE_DIR}/lib/)#${CMAKE_SOURCE_DIR}/../../build-android-aarch64/src)

option(NOT_ONLY_CRNN "build for crnn" OFF)

if(NOT_ONLY_CRNN)
    set(SRC_LIST interface_crnn.cpp face_recognization.cpp  dlib_preprocess.cpp extract_face_feature.cpp insert_feature.cpp update_feature.cpp)
else()
    set(SRC_LIST interface_crnn.cpp)
endif()

add_library(vision STATIC ${SRC_LIST})
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

add_executable(demo_crnn demo_crnn.cpp)
target_link_libraries(demo_crnn vision ${ncnn_LIB} -ldl -lpthread)

if(NOT_ONLY_CRNN)
add_executable(demo_insert_feature demo_insert_feature.cpp)
add_executable(demo demo.cpp)
add_executable(create_db db.cpp create_db.cpp)
target_link_libraries(demo vision ${ncnn_LIB} ${SQLITE_LIB} ${DLIB} -ldl -lpthread)
target_link_libraries(demo_insert_feature vision ${ncnn_LIB} ${SQLITE_LIB} ${DLIB} -ldl -lpthread)
target_link_libraries(create_db vision ${ncnn_LIB} ${SQLITE_LIB} ${DLIB} -ldl -lpthread)
endif()

