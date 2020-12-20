To run the main.py in python script standalone folder:

Set up libraries: dlib, numpy, the latest opencv-contrib-python, sys, 
scipy.spatial.qhull's Delaunay, 

First: Use command line.
Make sure the videos are in the Resources folder.
Configure system arguments to be: 
1) path to video file 1 in Resources folder 
2) path to video file 2 in Resources folder

For example: python main.py 'Resources/video1 'Resources/video2'

This will create a window with the video loop.


If this does not work, Pycharm environment might be required:
https://www.jetbrains.com/pycharm/download/#section=windows

In this case, configure the environment with the libraries using: 
File>settings>Project:main.py>Python Interpreter> Click the '+' >> Add packages

Run>Edit Configurations..>Parameters>> Add 'Resources/video1 'Resources/video2'

And then run main.py




