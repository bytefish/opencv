CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(ldacv)

############################## OpenCV ##############################
#SET(OpenCV_DIR /path/to/your/opencv/installation) # probably needs to be set
FIND_PACKAGE(OpenCV REQUIRED)

############################## Fisherfaces #########################
INCLUDE_DIRECTORIES(BEFORE ${PROJECT_SOURCE_DIR}/include)
ADD_EXECUTABLE(lda src/main.cpp src/subspace.cpp src/fisherfaces.cpp src/helper.cpp)
TARGET_LINK_LIBRARIES(lda ${OpenCV_LIBS})
