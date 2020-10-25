#------------------------------------------------------------------------------
# Copyright 2020 CryptoGraphics
#
#  This program is free software; you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the Free
#  Software Foundation; either version 2 of the License, or (at your option)
#  any later version.  See COPYING for more details.
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# enable MSVC_RUNTIME_LIBRARY property
#if(MSVC)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

#endif()

project(VerthashMiner CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Auto configuration for windows msvc platform
if(MSVC)
    SET(INCLUDE_DIRECTORIES
        "${CMAKE_SOURCE_DIR}/deps/OpenCL/include"
        "${CMAKE_SOURCE_DIR}/deps/jansson/include"
        "${CMAKE_SOURCE_DIR}/deps/curl/include")

    SET(LIBRARY_PATHS 
        "${CMAKE_SOURCE_DIR}/deps/OpenCL/lib"
        "${CMAKE_SOURCE_DIR}/deps/jansson/lib"
        "${CMAKE_SOURCE_DIR}/deps/curl/lib")

    # Jansson
    find_path(JANSSON_INCLUDE_PATH
              NAMES jansson.h
              PATHS ${INCLUDE_DIRECTORIES})
    find_library(JANSSON_LIBRARY
                 NAMES jansson
                 PATHS ${LIBRARY_PATHS})
    # Curl
    find_path(CURL_INCLUDE_DIR
              NAMES curl/curl.h
              PATHS ${INCLUDE_DIRECTORIES})
    find_library(CURL_LIBRARY_DEBUG
                 NAMES libcurl_a_debug
                 PATHS ${LIBRARY_PATHS})
    find_library(CURL_LIBRARY_RELEASE
                 NAMES libcurl_a
                 PATHS ${LIBRARY_PATHS})
    # OpenCL
    find_path(OpenCL_INCLUDE_DIR
              NAMES CL/opencl.h
              PATHS ${INCLUDE_DIRECTORIES})
    find_library(OpenCL_LIBRARY
                 NAMES OpenCL
                 PATHS ${LIBRARY_PATHS})
endif()
# -----------------------------------------------------------------------------
# Disable some warning for MSVC compiler
if(MSVC)
   add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS)
endif()


#------------------------------------------------------------------------------
# GENERIC options

SET(VERTHASH_RESULT_VALIDATION "Extended" CACHE STRING "library selected at CMake configure time")
SET_PROPERTY(CACHE VERTHASH_RESULT_VALIDATION PROPERTY STRINGS Simple Extended Full) 

if(VERTHASH_RESULT_VALIDATION STREQUAL Extended)
    add_definitions(-DVERTHASH_EXTENDED_VALIDATION)
endif()

if(VERTHASH_RESULT_VALIDATION STREQUAL Full)
    add_definitions(-DVERTHASH_FULL_VALIDATION)
endif()


#option(VERTHASH_EXTENDED_VALIDATION "Extended 64 bit device side nonce validation" ON)

#if(VERTHASH_EXTENDED_VALIDATION)
#    add_definitions(-DVERTHASH_EXTENDED_VALIDATION)
#endif()

#option(VERTHASH_FULL_VALIDATION "Extended 64 bit device side nonce validation" ON)
#
#if(VERTHASH_FULL_VALIDATION)
#    add_definitions(-DVERTHASH_FULL_VALIDATION)
#endif()


#------------------------------------------------------------------------------
# CUDA
#option(USE_CUDA "Use CUDA" ON)

set(CUDA_FOUND FALSE)
mark_as_advanced(CUDA_SDK_ROOT_DIR)
find_package(CUDA)

if(CUDA_FOUND)
    option(USE_CUDA "Use CUDA" ON)
    add_definitions(-DHAVE_CUDA)
else()
    option(USE_CUDA "Use CUDA" OFF)
endif()

if(USE_CUDA)
    message("INFO: Using CUDA version ${CUDA_VERSION}")
else()
    message("INFO: Building without CUDA support")
endif()


if(CUDA_FOUND AND USE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        --ptxas-options=-v
# generate ptx for
        -gencode arch=compute_35,code=sm_35;
        -gencode arch=compute_50,code=sm_50;
        -gencode arch=compute_52,code=sm_52;
        -gencode arch=compute_60,code=sm_60;
        -gencode arch=compute_61,code=sm_61;
        -gencode arch=compute_70,code=sm_70;
        -gencode arch=compute_75,code=sm_75;
# generate SASS for(performance will not depend on driver)
        #-gencode arch=compute_35,code=compute_35;
        #-gencode arch=compute_50,code=compute_50;
        #-gencode arch=compute_52,code=compute_52;
        #-gencode arch=compute_60,code=compute_60;
        #-gencode arch=compute_61,code=compute_61;
        #-gencode arch=compute_70,code=compute_70;
        #-gencode arch=compute_75,code=compute_75;

        -lineinfo;
    )

    if (CUDA_VERSION_MAJOR LESS 11)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_30,code=sm_30;
        #-gencode arch=compute_30,code=compute_30;
        )
    else()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_80,code=sm_80;
        #-gencode arch=compute_80,code=compute_80;
        )
    endif()

    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};)
    set(CUDA_NVCC_FLAGS_MINSIZEREL ${CUDA_NVCC_FLAGS_MINSIZEREL}; -O1; -DNDEBUG)
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE}; -O3;)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO}; -O2; -DNDEBUG)
endif()

#------------------------------------------------------------------------------
# Fix compile error with some configurations
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MTd")
endif()

#------------------------------------------------------------------------------
# OpenCL
find_package( OpenCL REQUIRED )
if(NOT OpenCL_FOUND)
    message(SEND_ERROR "Failed to find OpenCL library")
   # return()
else()
    include_directories(${OpenCL_INCLUDE_DIR})
endif()

#------------------------------------------------------------------------------
# JANSSON
find_package(jansson REQUIRED)
if(NOT JANSSON_FOUND)
    message(SEND_ERROR "Failed to find jansson library")
   # return()
else()
    include_directories(${JANSSON_INCLUDE_PATH})
endif()

#------------------------------------------------------------------------------
# CURL
# display in all menus
#mark_as_advanced(CLEAR CURL_INCLUDE_DIR CURL_LIBRARY)
if(MSVC)
option(USE_CURL_STATIC "USE_CURL_STATIC" ON)
else()
option(USE_CURL_STATIC "USE_CURL_STATIC" OFF)
endif()


if(USE_CURL_STATIC)
    add_definitions(-DCURL_STATICLIB)
endif()

find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    message(SEND_ERROR "Failed to find libcurl")
    return()
else()
    include_directories(${CURL_INCLUDE_DIR})
endif()

#------------------------------------------------------------------------------
# OpenSSL
if (UNIX)
    option(USE_OPENSSL "Use OPENSSL" ON)
else()
    option(USE_OPENSSL "Use OPENSSL" OFF)
endif()

if(USE_OPENSSL)
    option(USE_OPENSSL_STATIC "USE_OPENSSL_STATIC" OFF)
    if(USE_OPENSSL_STATIC)
        set(OPENSSL_USE_STATIC_LIBS, TRUE)
        if(MSVC)
            set(OPENSSL_MSVC_STATIC_RT, TRUE)
        endif()
    endif()

    find_package(OpenSSL REQUIRED)
    if(NOT OPENSSL_FOUND)
        message(SEND_ERROR "Failed to find OpenSSL library")
        return()
    else()
        include_directories(${OPENSSL_INCLUDE_PATH})
    endif()

endif()

# -----------------------------------------------------------------------------
if(CUDA_FOUND AND USE_CUDA)
CUDA_ADD_EXECUTABLE(VerthashMiner
    src/external/sha3/sha3.cpp
    src/external/tinycthread/tinycthread.cpp
    src/vhCore/ConfigFile.cpp
    src/vhCore/Util.cpp
    src/vhCuda/verthash.cu
    src/main.cpp
)
else()
add_executable(VerthashMiner
    src/external/sha3/sha3.cpp
    src/external/tinycthread/tinycthread.cpp
    src/vhCore/ConfigFile.cpp
    src/vhCore/Util.cpp
    src/main.cpp
)
endif()


# -----------------------------------------------------------------------------

target_include_directories(VerthashMiner PRIVATE src)
target_compile_features(VerthashMiner PUBLIC cxx_std_11)




if(MSVC)
    set_property(TARGET VerthashMiner PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    target_link_libraries(VerthashMiner
                          ${JANSSON_LIBRARY} ${CURL_LIBRARIES} ${OpenCL_LIBRARY}
                          wldap32 crypt32 Ws2_32 Normaliz)
else()
    target_link_libraries(VerthashMiner
                          ${JANSSON_LIBRARY} ${CURL_LIBRARIES} ${OpenCL_LIBRARY})

    if (UNIX)
        target_link_libraries(VerthashMiner
                              pthread)
        if (USE_OPENSSL)
            target_link_libraries(VerthashMiner
                                  ${OPENSSL_LIBRARIES})
            if(USE_OPENSSL_STATIC)
                target_link_libraries(VerthashMiner
                                      rt)
            endif()                                  
        endif()
    endif()
endif()


             
# -----------------------------------------------------------------------------
install(
    TARGETS VerthashMiner
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)

INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/src/kernels/" DESTINATION "${CMAKE_INSTALL_PREFIX}/kernels")

