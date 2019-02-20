# DRONE_CODE_REPOSITORY

This Repository is for sharing the code related to the Autonomous Drone Project that we are working on 

I have uploaded the code for the repository and I will include links here to the depenedencies for each code sample

Explanations of Files: 

Test_Model_Drone.py -> Allows you to test the outputs of the neural net vs the true positives or negatives of the filepath that stores your positive and negative files

Read CSV for Values: Allows you to extract all the learning to fly by crashing data and create the new files based on this data

drone_CSV_Extract -> Uses Read_CSV_for_Values and actually loads the images into positive and negative samples

Drone_Lenet_Move -> Meant for moving the drone, can control automatically if a shutdown is needed


Dependencies:

Python2.7 for libardrone

libardrone: https://github.com/venthur/python-ardrone

cv2: pip install cv2

Python3:
Keras: pip install keras







