CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(name "misc opencv samples")
project(misc_opencv_samples)

## If you didn't install OpenCV in a system-wide search path,
## you probably need to set the OpenCV_DIR for the find_package
## macro of CMake to function properly:
#SET(OpenCV_DIR /path/to/your/opencv/installation)

## Packages you'll need for this samples to work. You can 
## obtain an OpenCV copy at: http://opencv.willowgarage.com.
find_package(OpenCV REQUIRED) 

## Probably I should write a macro to compile the samples. But... I don't mind repeating
## myself here:


### The pca_demo shows how to perform a PCA in OpenCV. It's also shown how to display
### the famous Eigenfaces from the input data. You can either load the images in a 
### hardcoded manner or load it from a CSV file, just look up the comments in code:
add_executable(pca_demo pca_demo.cpp)
target_link_libraries(pca_demo opencv_core opencv_imgproc opencv_highgui)

### The skin_color_demo can be called with an image, and will show the regions of skin 
### color found in the image. Based on simple thresholding rules:
add_executable(skin_color_demo skin_color.cpp)
target_link_libraries(skin_color_demo opencv_core opencv_imgproc opencv_highgui)

### The tan_triggs_preprocessing can be called with an image, and will display 
### the illumination normalized image:
add_executable(tan_triggs_preprocessing tan_triggs.cpp)
target_link_libraries(tan_triggs_preprocessing opencv_core opencv_imgproc opencv_highgui)
