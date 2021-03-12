"# Sidewalk-scooter-detection" 

#System description

The System includes two modules one to detect the scooter and track it and the other to detect the sidewalk and road.


#Object detection :

We built an object detection model using retinanet to detect the electric scooters and in conjunction used trackers to track them. A tracker is assigned once the scooter is detected and the position of the tracker is adjusted based on the object detection algorithm. When there is a threshold overlap between tracker and object detection algorithm the tracker is adjusted to depict the bounding box of the object detection model.


We built a semantic segmentation model by using a Mapillary dataset that segments the pixels into the sidewalk and the street. The sidewalk is checked with the bounding box obtained from object detection and f the overlap is over a certain threshold then the bicycle is classified as in street or in the road.






#System Model:



#Dependencies and running the code

1. The model to run scooter detection. You will need to upload the model to the /model inside santamonica folder folder that will be created by the notebook after completing all the set up steps.
2. The camvid_tiny.zip folder which is contains all the dependencies required to perform sidewalk semantic segmentation.


Additional dependencies are found in s3 smc-trafficcam-aa bucket under github_dependencies/


Steps to set up the code and start running-

Add camvid.tiny folder in AWS>santamonica
Add model in the models folder in AWS>santamonica>models
Go to the AWS folder and run the command “Python manage.py runserver”
The website can be found at displayed local address

