CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(eigenfacescv)
SET(CMAKE_BUILD_TYPE Release)
# you probably need to set this
#SET(OpenCV_DIR /path/to/your/opencv/installation)
FIND_PACKAGE(OpenCV REQUIRED) # http://opencv.willowgarage.com
INCLUDE_DIRECTORIES(BEFORE ${PROJECT_SOURCE_DIR}/include)
ADD_EXECUTABLE(eigenfaces src/main.cpp  src/eigenfaces.cpp src/helper.cpp)
TARGET_LINK_LIBRARIES(eigenfaces ${OpenCV_LIBS})
