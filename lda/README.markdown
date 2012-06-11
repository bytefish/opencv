# !!!!!!!!!!!!!!!!!!!!!!!!!!! # 
# 					NOTE 							#
# !!!!!!!!!!!!!!!!!!!!!!!!!!! # 

## PLEASE USE LIBFACEREC ##

This repository is here for educational purpose, as it might be interesting to someone. Please use libfacerec at:

* http://www.github.com/bytefish/libfacerec

This contains all these algorithms with Unit Tests and a nice API.

# bytefish/opencv/lda #


This project implements the Fisherfaces method as described in: P. Belhumeur, J. Hespanha, and D. Kriegman, "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection", IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(7):711--720, 1997.

## Building the Project ##

This project has no additional dependencies, so compiling the project is as easy as writing (assuming you are in this folder):

```
philipp@mango:~/some/dir/lda$ mkdir build
philipp@mango:~/some/dir/lda$ cd build
philipp@mango:~/some/dir/lda/build$ cmake ..
philipp@mango:~/some/dir/lda/build$ make
philipp@mango:~/some/dir/lda/build$ ./lda filename.ext
```

And if you are in Windows using [MinGW](http://www.mingw.org) it may look like this:

```
C:\some\dir\lda> mkdir build
C:\some\dir\lda> cd build
C:\some\dir\lda\build> cmake -G "MinGW Makefiles" ..
C:\some\dir\lda\build> mingw32-make
C:\some\dir\lda\build> lda.exe filename.ext
```

You probably have to set the `OpenCV_DIR` variable if it wasn't added by your installation, see [Line 5 in the CMakeLists.txt](https://github.com/bytefish/opencv/blob/master/lda/CMakeLists.txt#L5) how to do this. If you have problems working with CMake or installing OpenCV, you probably want to read my guide on [Face Recognition with OpenCV2](http://www.bytefish.de/blog/face_recognition_with_opencv2). 

## Using the Project ##

The project comes with an example, please have a look at the [main.cpp](https://github.com/bytefish/opencv/blob/master/lda/src/main.cpp) on how to use the classes. You need some data to make the examples work, sorry but I really can't include those face databases in my repository. I have thoroughly commented the code and reworked it lately, to make its usage simpler. So if anything regarding the classes is unclear, please read the comments.

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
./lda /path/to/your/csvfile.ext
```

Or if you are in Windows:

```
lda.exe /path/to/your/csvfile.ext
```

## License ##

All code is put under a [BSD license](http://www.opensource.org/licenses/bsd-license), so feel free to use it for your projects.
