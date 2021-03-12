from django.shortcuts import render
from django.http import HttpResponseRedirect

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from .models import Video
from .forms import VideoForm

from django.conf import settings
from django.contrib.auth.decorators import login_required
import keras
print("hello1")
# import keras_retinanet


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import cv2
import sys
import imutils
from PIL import Image as PImage
from fastai.vision import Image
from fastai.vision import pil2tensor

from fastai.vision.learner import *
from fastai.vision import *
from fastai.vision import Image
import math

from keras_retinanet import models

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.visualization import draw_box, draw_caption

from keras_retinanet.utils.colors import label_color

from keras_retinanet.utils.gpu import setup_gpu
@login_required(login_url='/admin')
def showvideo(request):

    lastvideo= Video.objects.last()

    videofile= lastvideo.videofile if lastvideo else None
   

    form= VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
            form.save()
            return HttpResponseRedirect('/processed')

   
        
    context= {'videofile': videofile,
                'form': form,}
 
      
    return render(request, 'videos.html', context)



@login_required(login_url='/admin')
def showvideop(request):
  
    
    lastvideo= Video.objects.last()

    videofile= lastvideo.videofile if lastvideo else None
    

    form= VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()


    #S3 Bucket directory and video name
    

    VIDEO_NAME='media/'+str(videofile)

  
 

    # use this to change which GPU to use
    #gpu = 0

    # set the modified tf session as backend in keras
    #setup_gpu(gpu)
    model_path = os.path.join('santamonica/models', 'model_2_17_2020.h5')

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    labels_to_names = {0: 'sidewalk', 1: 'street'}
   
    cap = cv2.VideoCapture(VIDEO_NAME)
  
    frameRate = cap.get(5) #get frame rate of video
    Frames_Per_Second = .5
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    PADDING = round(frameRate)* 2 #two seconds before and after the detection

    frame_number = 0
    write_range = 0
    write_threshold = PADDING
    newFrames_list = [1]
    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret != True:
            break;
            
        frame_number += 1
        scooter_num = 0
        write_range -= 1
        if ( (frame_number % math.floor(frameRate//Frames_Per_Second) == 0) and (write_range <= 0) ):
            new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # preprocess image for network
            network_frame = preprocess_image(new_frame)
            network_frame, scale = resize_image(new_frame)

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(network_frame, axis=0))

            # correct for image scale
            boxes /= scale

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.50:
                    break
                scooter_num += 1;
            if(scooter_num != 0):
                write_range = PADDING
                for i in range(-1*write_threshold, write_threshold+1):
                    frame_num = frame_number + i
                    if(frame_num <= 0):
                        continue
                    if(not newFrames_list[-1] >= frame_num):
                        newFrames_list.append(frame_num)
        print("\r", "{}/{} frames completed".format(frame_number, totalFrames), end="")    

    cap.release()
    cv2.destroyAllWindows()

  
    


    print( "Video will be reduced to ",(len(newFrames_list)/totalFrames)*100, " of the original length")

    
    cap = cv2.VideoCapture(VIDEO_NAME)
   
    fourcc = cv2.VideoWriter_fourcc(*'XMPG') #refrence this doc: http://www.fourcc.org/codecs.php if using a different video format
   
    print(newFrames_list)
    frameRate = cap.get(5) #get frame rate of video
    frameCount = len(newFrames_list)
    ret, frame = cap.read()
    height, width, layers = frame.shape
    size = (width,height)
    out = cv2.VideoWriter("MLREDUCED"+VIDEO_NAME,fourcc, frameRate, size) 

    frame_number = 0
    current_frame = 0
    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret != True:
            break;
        frame_number += 1
        if frame_number == newFrames_list[current_frame]:
            print("hi")
            current_frame +=1
            out.write(frame)
        print("\r", "{}/{} frames completed".format(current_frame, frameCount), end="") 
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()


    # <br/>
    # <h1>TRACKING</h1>

    # <h5>Initialize helper functions and classes</h5>

    # In[ ]:

  
    path=Path('santamonica/camvid_tiny') #setup path
    path_lbl = path/'labels'
    path_img = path/'images'
    fnames = get_image_files(path_img)
    fnames[:3]
    img_f = fnames[0]
    img = open_image(img_f)
    img.show(figsize=(5,5))
    lbl_names = get_image_files(path_lbl)
    lbl_names[:3]
    get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
    mask = open_mask(get_y_fn(img_f))
    mask.show(figsize=(5,5), alpha=1)
    src_size = np.array(mask.shape[1:])   #mask data
    src_size,mask.data
    codes = np.loadtxt(path/'codes.txt', dtype=str)
    data = (SegmentationItemList.from_folder(path_img)
            .split_by_rand_pct()
            .label_from_func(get_y_fn, classes=codes)
            .transform(get_transforms(), tfm_y=True, size=256)
            .databunch(bs=2, path=path)
            .normalize(imagenet_stats))
    name2id = {v:k for k,v in enumerate(codes)}
    void_code = name2id['Void']


    


    def acc_camvid(input, target):     #defining accuracy
        target = target.squeeze(1)
        mask = target != void_code
        return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    from fastai.vision.models import resnet34
    learn=unet_learner(data,resnet34, metrics=acc_camvid) #creating the architecture with imagenet weights
    learn.data.single_ds.tfmargs['size'] = None
    learn.load('stage-2')   #Loading the trained Model in this case it is stage-2 modek which is stored under camvid-tiny/models
    def infer(x):
        #returrns [ROW][COL]
        start = time.time()
        img = open_image(x)
        c=learn.predict(img)
        print(c[0])
        print(c[1][0][0][0])
        c[0].show()
        print("processing time: ", time.time() - start)
        return c[1][0]

    scooters = []
    sidewalkTable = None
    global counter_street
    counter_street = 0
    global counter_sidewalk
    counter_sidewalk = 0
    class Tracker:
        Id = None
        tracker = None
        box=[]
        status=False
        labelType = None
        correctionByDetector = False
        attemptedToRemove = False
        
        def __init__(self, frame, bbox):
            global counter_street 
            global counter_sidewalk
            self.Id = counter_street
            self.tracker = cv2.TrackerCSRT_create()
            self.box = bbox;
            self.status = self.tracker.init(frame, bbox)
            self.labelType = sidewalkOverlap(sidewalkTable, bbox)
            if(self.labelType == "street"):
                counter_street +=1
            else:
                counter_sidewalk +=1
            self.correctionByDetector = True
            
        def updateTracking(self, frame):
            self.status, self.box = self.tracker.update(frame)
            self.correctionByDetector = False
            if not self.status:
                self.attemptedToRemove = True
                self.removeSelf() 
                
        def reinitialze(self, frame, bbox):
            global counter_street
            global counter_sidewalk
            self.correctionByDetector = True
            self.tracker = cv2.TrackerCSRT_create()
            self.box = bbox;
            self.status = self.tracker.init(frame, bbox)
            newLabel = sidewalkOverlap(sidewalkTable, bbox)   
            #If scooter switches from street to sidewalk, change its count to sidewalk
            if(self.labelType != newLabel):
                if(self.labelType == "street"):
                    counter_street -=1
                    counter_sidewalk +=1
            self.labelType = newLabel
                
        def removeSelf(self):
            if (self.attemptedToRemove):
                scooters.remove(self)#remove current instance from objects list
            else:
                self.attemptedToRemove = True
                
        def writeFrame(self, frame):
            if self.status:
                p1 = (int(self.box[0]), int(self.box[1]))
                p2 = (int(self.box[0] + self.box[2]), int(self.box[1] + self.box[3]))
                if(self.labelType == 'street'):
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                else:
                    cv2.rectangle(frame, p1, p2, (255,255,0), 2, 1)

    # In[ ]:
    def sidewalkOverlap(sidewalkTable, bbox):
        #sidewalkTable e.g. [ROW][COL]
        # bbox. e.g. (730, 170, 20, 40)
        #returns "street" or "sidewalk"
        
        #Percent of area to check for overlap with sidewalk starting from the bottom of the bounding box
        percentArea = .25
        
        #What percentage of scooter pixels in the checked area should determine a sidewalk overlap
        overlapThreshold = .20
        
        startX = bbox[0] - 1
        startY = bbox[1] - 1
        endX = bbox[0]+bbox[2] -1
        endY = bbox[1]+bbox[3] -1
        
        #How many columns are being checked
        colQty = endX-startX
        
        #Shortened sidewalk table
        table = sidewalkTable[startY:endY]
        
        #Area of the portion used to calculate overlap
        area = colQty*(endY-startY)*percentArea
        
        rowCounter = 0
        sidewalkCounter = 0
        for index in range(len(table)):
            rowCounter+=1
            row = table[-(index+1)]
            newRow = row[startX:endX]
        
            for col in newRow:
                if col == 19:
                    sidewalkCounter+=1
                
            if(rowCounter >= colQty*percentArea):
                break
        
        percentOverlap = sidewalkCounter/area
  

        if(percentOverlap>=overlapThreshold):
            return "sidewalk"
        return "street"
        

    HIGH_CONFIDENCE_THRESHOLD=.95
    LOW_CONFIDENCE_THRESHOLD=.2

    def inferSidewalk(frame):
        #returrns [ROW][COL]
        new_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pil_img = PImage.fromarray(new_frame.astype('uint8'), 'RGB')
        pil_img = pil2tensor(pil_img,np.float32)
        inf_img = Image(pil_img.div_(255))
        c=learn.predict(inf_img)
        return c[1][0]
        
    def detector(frame):
        highConfidenceBoxes=[]
        lowConfidenceBoxes=[]
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # preprocess image for network
        network_frame = preprocess_image(new_frame)
        network_frame, scale = resize_image(new_frame)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(network_frame, axis=0))

        boxes /= scale
        detectorHash={}
        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score<LOW_CONFIDENCE_THRESHOLD:
                break
            bx = box.astype(int)
            bbox = (bx[0], bx[1], bx[2]-bx[0], bx[3]-bx[1])
            tup= (bbox,labels_to_names[label],score)
            
            if score>=HIGH_CONFIDENCE_THRESHOLD:
                highConfidenceBoxes.append(tup)
                color = label_color(label+5)
                draw_box(frame, bx, color=color)
            else:
                if len(highConfidenceBoxes)==0:
                    lowConfidenceBoxes.append(tup)   
                else:
                    doesOverlap = False
                    for HCbox in highConfidenceBoxes:
                        if overlap(HCbox[0],tup[0], False) >=.8: #Filter overlaped boxes
                            doesOverlap = True
                            break
                    if not doesOverlap:
                        color = label_color(label+14)
                        draw_box(frame, bx, color=color)
                        lowConfidenceBoxes.append(tup)
        
        detectorHash['high_confidence']=highConfidenceBoxes
        detectorHash['low_confidence']=highConfidenceBoxes+lowConfidenceBoxes
        return detectorHash

    def overlap(i, j, isTracker):
        #box1
        bb1_x1 = i[0]
        bb1_x2 = i[0] + i[2]
        bb1_y1 = i[1]
        bb1_y2 = i[1] + i[3]
        
        #box2
        if isTracker:
            bb2_x1 = j.box[0]
            bb2_x2 = j.box[0] + j.box[2]
            bb2_y1 = j.box[1]
            bb2_y2 = j.box[1] + j.box[3]
        else:
            bb2_x1 = j[0]
            bb2_x2 = j[0] + j[2]
            bb2_y1 = j[1]
            bb2_y2 = j[1] + j[3]
            
        if bb1_x1 >= bb1_x2 or bb1_y1 >= bb1_y2 or bb2_x1 >= bb2_x2 or bb2_y1 >= bb2_y2:
            return -1
        
    
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1_x1, bb2_x1)
        y_top = max(bb1_y1, bb2_y1)
        x_right = min(bb1_x2, bb2_x2)
        y_bottom = min(bb1_y2, bb2_y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left+1) * (y_bottom - y_top+1)

        # compute the area of both AABBs
        bb1_area = (bb1_x2 - bb1_x1+1) * (bb1_y2 - bb1_y1+1)
        bb2_area = (bb2_x2 - bb2_x1+1) * (bb2_y2 - bb2_y1+1)

    #     iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        if intersection_area == bb2_area:
            return 1
        
        overlap = intersection_area/bb1_area
        return overlap

    def filterbox(frame,detectorHash):
        highConfDetectorBoxes = detectorHash['high_confidence']
        lowConfDetectorBoxes = detectorHash['low_confidence']
        
        #checks to see if the overlap is significant
        if len(scooters) == 0:
            for i in highConfDetectorBoxes:
                scooters.append( Tracker( frame, i[0]) )
        else:
            for i in lowConfDetectorBoxes:
                referenced = False #has this box been referenced to an existing tracker

                for j in scooters:
                    if j.correctionByDetector:
                        continue
                    if overlap(i[0], j, True)>0.5:
                        j.reinitialze(frame, i[0])
                        referenced = True
                        break

                if not(referenced) and (i[2]>=HIGH_CONFIDENCE_THRESHOLD):
                    scooters.append( Tracker( frame, i[0] ) )
                    
                    
            for j in scooters: #remove scooter trackers that weren't detected by the scooter
                if j.correctionByDetector:
                    continue
                j.removeSelf()
                    
                        

   

    # import keras
    VIDEO_ADDRESS= VIDEO_NAME
    # VIDEO_ADDRESS= 'broadway-3rd.mpg'

    #Insert output video name belo
    vout= 'processed/'+str(videofile).split('/')[-1]
    OUTPUTVIDEO_ADDRESS= 'media/processed/'+ str(videofile).split('/')[-1]

    video = cv2.VideoCapture(VIDEO_ADDRESS)

    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    frameRate = video.get(5) #get frame rate of video
    ret, frame = video.read()
    height, width, layers = frame.shape
    size = (width,height)
    totalFramesInVideo = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    #Initialize output video
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video = cv2.VideoWriter(OUTPUTVIDEO_ADDRESS,fourcc, frameRate, size)

    # Exit if video not opened.


    # In[ ]:


    #The number of frames to run the detector after
    detectorRoutine = 30
    sidewalkDetectionRoutine = 5000

    while (video.isOpened()):
        frameId = video.get(1) #current frame number
        print("\r", "{}/{} frames completed".format(frameId, totalFramesInVideo), end="")  
        
        # Read a new frame
        ok, frame = video.read()
        
        if not ok:
            break

        timer = cv2.getTickCount()
        
        # get updated location of objects in subsequent frames
        for scooter in scooters:
            scooter.updateTracking(frame)

        #run sidewalk detector on routine
        if(frameId-1) % sidewalkDetectionRoutine == 0:
            sidewalkTable = inferSidewalk(frame)
            
        #run the detector on routine or when an object is lost
        if (frameId-1) % detectorRoutine == 0:
            bboxes = detector(frame); #get new bounding boxes and label types as tuples from detector model
            filterbox(frame, bboxes); #filter any overlaps in the objects on the frame
            

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        for scooter in scooters:
            scooter.writeFrame(frame)
        
        # Display tracker type on frame
        cv2.putText(frame,"Street Count: " + str(int(counter_street)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),2);
        cv2.putText(frame,"Sidewalk Count: " + str(int(counter_sidewalk)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),2);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2);
        
        output_video.write(frame)
    #     if(frameId > 7000):
    #         break;
            

        
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

    
    context= {'videofile': vout,
            'form': form,'st_c':counter_street,'si_c':counter_sidewalk} 
   
 
    return render(request, 'uploaded.html',context)