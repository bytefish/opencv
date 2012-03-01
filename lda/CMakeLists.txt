CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(ldacv)

############################## Eigen ##############################
set(WITH_EIGEN OFF CACHE BOOL "Use Eigen3.")
if(WITH_EIGEN)
    find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
            PATHS "/usr/local/include/eigen3" "/opt/include/eigen3" "/usr/include/eigen3"
            DOC "The path to Eigen3 headers")
    if(EIGEN_INCLUDE_PATH)
        include_directories(${EIGEN_INCLUDE_PATH})
        set(HAVE_EIGEN 1)
    endif()
endif()

############################## OpenCV ##############################
#SET(OpenCV_DIR /path/to/your/opencv/installation) # probably needs to be set
FIND_PACKAGE(OpenCV REQUIRED)

############################## Fisherfaces #########################
INCLUDE_DIRECTORIES(BEFORE ${PROJECT_SOURCE_DIR}/include)
ADD_EXECUTABLE(lda src/main.cpp src/subspace.cpp src/fisherfaces.cpp src/helper.cpp)
TARGET_LINK_LIBRARIES(lda ${OpenCV_LIBS})
