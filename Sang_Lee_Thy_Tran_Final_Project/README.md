# Face Swapping
**University of Pennsylvania, CIS 581: Computer Vision**
* Sangeun Lee: [personal website](https://leesang.myportfolio.com/), [email](eunsang@sas.upenn.edu)
* Thy (Tea) Tran: [LinkedIn](https://www.linkedin.com/in/thy-tran-97a30b148/), [personal website](https://tatran5.github.io/), [email](thytran316@outlook.com)
* Tested on: Windows 10, i7-8750H @ 2.20GHz 22GB, GTX 1070

![](images/sethMeyersOnHughGrant.gif)

## To Run the Program
Make sure dlib, OpenCV2, and NumPy are installed.

Configure system arguments to be: 1) path to video file in Resources folder for the basis and 2) video file with the face to be swapped into the first video.

To run the main.py:

Set up libraries: dlib, numpy, the latest opencv-contrib-python

Use command line.

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



## Results

### Seth Meyers and Hugh Grant
| Face 1 on face 2| Face 2 on face 1 |
|---|---|
|![](Videos/sethMeyersOnHughGrant.gif)|![](Videos/hughGrantOnSethMeyers.gif)|

|Original video 1 | Original video 2 |
|---|---|
|![](Videos/original/hughGrant.gif)|![](Videos/original/sethMeyers.gif)|

### Frank Underwood and Homelander
| Face 1 on face 2| Face 2 on face 1 |
|---|---|
|![](Videos/frankUnderwoodOnHomelander.gif)|![](Videos/homelanderOnFrankUnderwood.gif)|

|Original video 1 | Original video 2 |
|---|---|
|![](Videos/original/frankUnderwood.gif)|![](Videos/original/homelander.gif)|

### MrRobot and Frank Underwood
|Face 1 on face 2 | Original video 1| Original video 2|
|---|---|---|
|![](Videos/mrRobotOnFrankUnderwood.gif)|![](Videos/original/mrRobot.gif)|![](Videos/original/frankUnderwood.gif)|

## Process

### Facial Landmarks Detection

Detect faces and facial landmarks in the source and replacement videos. We use OpenCV dlib library to do this. 

In this image, the blue points are feature points/landmarks, while the green line is a convexhull surrounding the most outer points of the feature landmarks.

<img src="Videos/intermediate/landmark.PNG" width="300">

### Feature extraction

From the above, we apply a mask to extract only the face from the first video to later applied on the second video.

<img src="Videos/intermediate/faceExtract.PNG" width="300">

### Delaunay triangulation

Because the first face might not have the same dimension, orientation or resolution as the second face, we need to perform Delaunay triangulation on the first face and second face. 
Then, we warp corresponding triangles on the first face on the second face. 

| Full face triangulation | Corresponding triangles between two videos |
|---|---|
|![](Videos/intermediate/triangulation.PNG)|![](Videos/intermediate/tri_mapping.gif)|

### Face swapping & blending

After assembling all of the transformed triangles into one convexhull, we can put it onto the second video (frame by frame). We also perform seamless cloning from OpenCV2 to blend the first face to the head of the other video to make the face swapping look more natural.

|Face swapping with no blending| Face swapping with blending|
|---|---|
|![](Videos/intermediate/homelanderOnFrankUnderwood_noBlending.gif)|![](Videos/homelanderOnFrankUnderwood.gif)|

## Performance analysis

While our implementation works well with most of the tested videos, it has a hard time detecting features correctly in scenes where there is little difference in colors between non-face and face.

|Feature detection on distinctive background| Feature detection when background has similar values to face|
|---|---|
|![](Videos/intermediate/frankUnderwood_landmarks.gif)|![](Videos/intermediate/mrRobot_landmarks.gif)|

Our current implementation does not perform well on swapping faces with different skin tones. Especially in the case of pasting face with a darker skin on a head of lighter skin, this is much more obvious.

Our current implementation also does not handle changing light environment extremely well either. If the pasted face has changing light colors, it is likely not matching with the rest of the body in the video pasted on. If the pasted face does not have changing light colors, the face overall still matches with the rest of the lighting in the scene pretty well, as seen below. However, there are details on the face that should have been brighter or darker, but our implementation fails to do so as seen below

| Dark-skin-tone face on light-skin-tone head | Light-skin-tone-face on dark-skin-tone head |
|---|---|
|![](Videos/mrRobotOnSethMeyers.gif)|![](Videos/sethMeyersOnMrRobot.gif)|

| Dynamic-lighting face on non-dynamic-lighting head | Non-dynamic-lighting face on dynamic-lighting head|
|---|---|
|![](Videos/details/homelanderOnFrankUnderwood_weirdLighting.PNG)|![](Videos/details/frankUnderwoodOnHomelander_lighting.PNG)|

Some other minor details include the blending between triangles mapped from one face to another. There are at times small black marks in the output. In addition, occasionally the blending results in center top of the swapped face has brighter lighting than the rest of the face despite not being a feature in its original video.

|Black marks and bright center top|
|---|
|<img src="Videos/details/hughGrantOnSethMeyers_blackMarksAndBrightCentertop.PNG" width="300">|

## Potential improvements
The quality of face blending could be refined to by using gradient domain blending or any other technique that makes the swapped faces look real with their new bodies. Compensation for exposure, lighting and shadows, poses, skin tone, etc. as well as optical flow techniques to robustly track faces could further improve the project.
|---|---|

## Resources
- Pycharm 2020.2.4
- numPy
- [dlib](http://dlib.net/)
- [dlib HOG](https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py)
- [shape_predictor_68_face_landmarks.dat from dlib](https://github.com/davisking/dlib-models)
- [OpenCV-Python](https://docs.opencv.org/3.4/d2/d42/tutorial_face_landmark_detection_in_an_image.html)
- Python 3.7.6
