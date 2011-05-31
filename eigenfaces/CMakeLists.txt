PROJECT(eigencv)
# General
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE(eigenfaces main.cpp Eigenfaces.cpp)
# OpenCV
TARGET_LINK_LIBRARIES(eigenfaces ${OpenCV_LIBS})
