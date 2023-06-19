# Parking Space Counter 

*** First of all, personal vehicles are not the future. It is crucial to reduce their usage in order to mitigate the carbon footprint in our atmosphere.***

## Overview
Monitoring and analyzing the occupancy of parking spaces in a parking lot can indeed offer valuable insights into the growth and activity of nearby businesses. However, the high cost associated with installing a dedicated parking count system often deters free and non-gated parking lots from pursuing such solutions. This project aims to explore the feasibility of applying machine learning techniques to count parked cars using existing surveillance videos.

## Challenge
In the task of counting the number of cars in a video frame, various approaches such as object detection techniques, object tracking, and object segmentation can be employed. However, the primary challenge lies in differentiating between parked cars and cars that are in motion. Our focus is on identifying and counting the cars that are parked in the designated parking stalls, excluding those that are actively driving or moving within the area.

Frame:

Mask:

## Solution
The solution is to apply a binary mask with the background as black and the foreground, which will be the shapes of each parking stall, as white, on each video frame. The mask can be done by either implementing code or by graphic software (Inkscape, for example). By applying the mask, we now can focus only on the designated parking stall areas, and able elevering any classification machine learning method to determine if a stall whether is empty of occupied. 

## Result
![Alt Text](output/output.gif)

## Video source
```
L. Mou and X. X. Zhu, "Vehicle instance segmentation from aerial image and video using a multi-task learning residual fully convolutional network," IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 11, pp. 6699-6711, 2018.
```
You can download the data from https://drive.google.com/file/d/1Wq_1J8oVHNwL5iq5KF-zUszzb3x1GWeG/view
