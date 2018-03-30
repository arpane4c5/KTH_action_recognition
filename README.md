# Recognizing action in KTH dataset using person's movement features.
## Steps

1. For all the videos, extract the person boundaries, get the centroid and obtain the speed and distance over a window of frames. 

2. The segmentation of person uses haarcascade, followed by Detectron (Mask-RCNN).

3. Learn a model on the motion features. 

4. Evaluate on the validation set.
