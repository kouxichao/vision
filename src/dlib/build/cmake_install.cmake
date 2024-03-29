# Install script for directory: /home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/libdlib.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dlib" TYPE DIRECTORY FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.cmake$" REGEX "/[^/]*\\_tutorial\\.txt$" REGEX "/cassert$" REGEX "/cstring$" REGEX "/fstream$" REGEX "/iomanip$" REGEX "/iosfwd$" REGEX "/iostream$" REGEX "/istream$" REGEX "/locale$" REGEX "/ostream$" REGEX "/sstream$" REGEX "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build" EXCLUDE)
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dlib" TYPE FILE FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/config.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dlib" TYPE FILE FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/revision.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib/dlib.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib/dlib.cmake"
         "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/CMakeFiles/Export/lib/cmake/dlib/dlib.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib/dlib-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib/dlib.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib" TYPE FILE FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/CMakeFiles/Export/lib/cmake/dlib/dlib.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib" TYPE FILE FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/CMakeFiles/Export/lib/cmake/dlib/dlib-release.cmake")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/dlib" TYPE FILE FILES
    "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/config/dlibConfig.cmake"
    "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/config/dlibConfigVersion.cmake"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/dlib-1.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/examples/dlib-19.15/dlib/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
