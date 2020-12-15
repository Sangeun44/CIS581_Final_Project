Detailed readme: https://github.com/Sangeun44/Face_Swapping/blob/main/README.md

To run the main.py in python script standalone folder:

Set up libraries: dlib, numpy, the latest opencv-contrib-python
Use command line.
Make sure the videos are in the Resources folder.
Configure system arguments to be: 
1) path to video file in Resources folder for the basis 
2) path to video file 2 in Resources folder

For example: python main.py 'Resources/video1 'Resources/video2'

This will create a window with the video loop.


If this does not work, Pycharm environment might be required:
https://www.jetbrains.com/pycharm/download/#section=windows

Use the project environment in Final_Project folder.

In this case, configure the environment with the libraries using: 
File>settings>Project:main.py>Python Interpreter> Click the '+' >> Add packages

Run>Edit Configurations..>Parameters>> Add 'Resources/video1 'Resources/video2'

And then run main.py




