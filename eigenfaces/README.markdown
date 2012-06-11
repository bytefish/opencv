# !!!!!!!!!!!!!!!!!!!!!!!!!!! # 
# 					NOTE 							#
# !!!!!!!!!!!!!!!!!!!!!!!!!!! # 

## PLEASE USE LIBFACEREC ##

This repository is here for educational purpose, as it might be interesting to someone. Please use libfacerec at:

* http://www.github.com/bytefish/libfacerec

This contains all these algorithms with Unit Tests and a nice API.

# bytefish/opencv/eigenfaces #

This project implements the Eigenfaces method as described in: Turk and Pentland, "Eigenfaces for recognition.", Journal of Cognitive Neuroscience
3 (1991), 71–86.

## Building the Project ##

This project comes as a [CMake project](http://www.cmake.org), so compiling the project is as easy as writing (assuming you are in this folder):

```
philipp@mango:~/some/dir/eigenfaces$ mkdir build
philipp@mango:~/some/dir/eigenfaces$ cd build
philipp@mango:~/some/dir/eigenfaces/build$ cmake ..
philipp@mango:~/some/dir/eigenfaces/build$ make
philipp@mango:~/some/dir/eigenfaces/build$ ./eigenfaces filename.ext
```

And if you are in Windows using [MinGW](http://www.mingw.org) it may look like this:

```
C:\some\dir\eigenfaces> mkdir build
C:\some\dir\eigenfaces> cd build
C:\some\dir\eigenfaces\build> cmake -G "MinGW Makefiles" ..
C:\some\dir\eigenfaces\build> mingw32-make
C:\some\dir\eigenfaces\build> eigenfaces.exe filename.ext
```

You probably have to set the `OpenCV_DIR` variable if it wasn't added by your installation, see [Line 5 in the CMakeLists.txt](https://github.com/bytefish/opencv/blob/master/eigenfaces/CMakeLists.txt#L5) how to do this. If you have problems working with CMake or installing OpenCV, you probably want to read my guide on [Face Recognition with OpenCV2](http://www.bytefish.de/blog/face_recognition_with_opencv2). 

## Using the Project ##

The project comes with an example, please have a look at the [main.cpp](https://github.com/bytefish/opencv/blob/master/eigenfaces/src/main.cpp) on how to use the classes. You need some data to make the examples work, sorry but I really can't include those face databases in my repository. I have thoroughly commented the code and reworked it lately, to make its usage simpler. So if anything regarding the classes is unclear, please read the comments.

In the example I use a CSV file to read in the data, it's the easiest solution I can think of right now. However, if you know a simpler solution please ping me about it. Basically all the CSV file needs to contain are lines composed of a _filename_ followed by a _;_ followed by the _label_ (as **integer number**), making up a line like this: `/path/to/image.ext;0`.

Think of the _label_ as the subject (the person) this image belongs to, so same subjects (persons) should have the same _label_. An example CSV file for the [AT&T Facedatabase](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) is [given here](https://github.com/bytefish/opencv/blob/master/eigenfaces/at.txt), which looks like this (assuming I've extracted the database to `/home/philipp/facerec/data/at`, file is without `...` of course):

```
/home/philipp/facerec/data/at/s1/1.pgm;0
/home/philipp/facerec/data/at/s1/2.pgm;0
...
/home/philipp/facerec/data/at/s2/1.pgm;1
/home/philipp/facerec/data/at/s2/2.pgm;1
...
/home/philipp/facerec/data/at/s40/1.pgm;39
/home/philipp/facerec/data/at/s40/2.pgm;39
```

Once you have a CSV file with **valid** _filenames_ and _labels_, you can run the demo by simply starting the demo with the path to the CSV file as parameter:

```
./eigenfaces /path/to/your/csvfile.ext
```

Or if you are in Windows:

```
eigenfaces.exe /path/to/your/csvfile.ext
```

## License ##

All code is put under a [BSD license](http://www.opensource.org/licenses/bsd-license), so feel free to use it for your projects.
