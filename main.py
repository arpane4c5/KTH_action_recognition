"""
Author: Anjali, Shreya, Sakshi and Arpan
Description: Action Recognition in KTH dataset using movement features
"""

import cv2
import numpy as np 
import math
import os
import matplotlib.pyplot as plt

# Parameters
DATASET = "/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset"
DEST_FOLDER = "/home/hadoop/VisionWorkspace/KTH_action_recognition"
SEQUENCES_FILE = "kth_sequences.txt"
TRAIN = "kth_actions_train"
VALIDATION = "kth_actions_validation"
TEST = "kth_actions_test"


# Iterate over all the videos in the folder and extract the features corresponding to
# the action sequences. Returns a matrix of values.
def extract_features(srcVideoFolder, locations, partition="train"):
    # read the cascade classifier
    cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    # all the video files of the partition set
    video_list = sorted(os.listdir(srcVideoFolder))
    # set the window size for which speed and distance has to be calculated
    win_size = 7
    train_features_list = []
    for vid in video_list:
        srcVideo=os.path.join(srcVideoFolder,vid)
        key = vid.rsplit('_',1)[0]
        action_seq_str = locations[key]
        #fil=vid[:-3]
        #fil=fil+"txt"
        #destFile=os.path.join(destFolder,fil)
        # return a matrix of features for a single video
        train_features_list.append(getVidFeature(srcVideo, action_seq_str, cascade))


def getVidFeature(srcVideo, action_seq_str, cascade):
    #read the video
    cap=cv2.VideoCapture(srcVideo)
    #initialize bg subtractor
    foreground_background =cv2.createBackgroundSubtractorMOG2()
    
    if not cap.isOpened():
        import sys
        print "Error in reading the video file ! Abort !"
        sys.exit(0)
    
    ####################################################
    # extract the start and end frame nos from action_seq_str 
    # Eg string = '1-75, 120-190, 235-310, 355-435'
    start_frames = []
    end_frames = []
    for marker in action_seq_str.split(','):
        temp = marker.split('-')        # ' 120-190'
        start_frames.append(int(temp[0]))   # 120
        end_frames.append(int(temp[1]))
    # check
    if len(start_frames)!=len(end_frames):
        print "Error in reading the frame markers from file ! Abort !"
        sys.exit(0)
    ####################################################
    ## Either above block has to be commented or below block
    ####################################################
    # For all the frames of the video
    #start_frames = [0]
    #end_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
    ####################################################
    
    for i, stime in enumerate(start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, stime)
        ret, prev_frame = cap.read()
        
        # Apply bg subtraction
        fgmask = fgbg.apply(prev_frame)
        
        # convert frame to GRAYSCALE
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        stime = stime + 1
        # iterate over the frames to find the optical flows
        while(cap.isOpened() and stime <= end_frames[i]):
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)      # shape is (120, 160)
            fgmask_sums.append(np.sum(fgmask))
            #print "fgmask shape : "+str(fgmask.shape)
            
            if not ret:
                break
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print "RGB Frame shape :" + str(frame.shape)+" Gray FS : "+str(curr_frame.shape)
            
            #cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, 
            #                iterations, poly_n, poly_sigma, flags[, flow])
            # prev(y,x)~next(y+flow(y,x)[1], x+flow(y,x)[0])
            flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            print "For frames: ("+str(stime-1)+","+str(stime)+") :: shape : "+str(flow.shape)
            #print "Flow stats (mean[mag], min, max) : "+str((np.mean(flow[:,0]), np.min(flow[:,0]), np.max(flow[:,0])))
            #print "Flow stats (mean[orient], min, max) : "+str((np.mean(flow[:,1]), np.min(flow[:,1]), np.max(flow[:,1])))
            #cv2.imshow('Current Frame (Gray)', curr_frame)
            #cv2.imshow('Prev Frame(Gray)', prev_frame)
            
            # Visualization
            vis_vectors = draw_flow(prev_frame, flow, 8)
            vis_bgr = draw_flow_bgr(flow, frame, flow_means_x, flow_means_y)
            #cv2.imshow('Flow Vis', vis_vectors)
            #cv2.imshow('Flow BGR', vis_bgr)
            X_list.append(vis_bgr)       # append to the list
            #y_list.append(curr_label)
            #cv2.imshow('RGB Frame', frame)
            #cv2.imwrite("frame_"+str(stime)+".jpg", curr_frame)
            #if not(abs(flow_means_x[-1])>300 and abs(flow_means_y[-1])>300):
            #    cv2.imwrite(os.path.join(os.getcwd(),"walk_frames/CurrFrame_"+str(stime)+".jpg"), curr_frame)
            if fgmask_sums[-1]<80000:
                cv2.imwrite(os.path.join(os.getcwd(),"walk_frames/CurrFrame_"+str(stime)+".jpg"), curr_frame)
            stime = stime + 1
            prev_frame = curr_frame
            # uncomment following block of code to write visualizations to files
            #keyPressed = waitTillEscPressed()    
#            if keyPressed==0:       # write to file
#                cv2.imwrite(os.path.join(os.getcwd(),imgPrefix+"_OF_"+str(stime-1)+".jpg"),vis_bgr)
#                cv2.imwrite(os.path.join(os.getcwd(),imgPrefix+"_prevF_"+str(stime-1)+".jpg"), prev_frame)
#                cv2.imwrite(os.path.join(os.getcwd(),imgPrefix+"_currF_"+str(stime-1)+".jpg"), curr_frame)
#                cv2.imwrite(os.path.join(os.getcwd(),imgPrefix+"_OFVecs_"+str(stime-1)+".jpg"), vis_vectors)
    cap.release()            
    cv2.destroyAllWindows()
    #print flow_means_x      
    return
    

    
def get_person_haar(cascade):
    a = open(destFile, 'w')
    while True:
        ret,frame = cap.read()
        body = cascade.detectMultiScale(frame)
        #print(body)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5,5),np.uint8)
        #apply bg subtractor to get foreground mask 
        foreground_mask=foreground_background.apply(frame)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        hist,bins = np.histogram(foreground_mask.ravel(),256,[0,256])
        cv2.imshow("foreground mask",foreground_mask)
		
        #print(hist[0])
        sum=0
        for i in range(10):
            sum=sum+hist[255-i]
            
        if(sum>200): #no. of pixels of white 
          #print('object is doing motion')

            arr_size=7 #no of frames taking together
            xary,yary=np.zeros(arr_size),np.zeros(arr_size)

            for f in range(arr_size): #using consecutive arr_size frames 
                if(ret ==True):
                    cannyi=cv2.Canny(foreground_mask,10,250)
                    cv2.imshow('canny',cannyi)
                    cv2.waitKey(0)
			
                    #finding contours
                    im2, contours, hierarchy = cv2.findContours(cannyi,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
                    #bounding rectangle around a contour 
                    height,width = cannyi.shape
                    min_x, min_y = width, height
                    max_x = max_y = 0

                    #making a new image that will have only the center of rectangle 
                    new_image=np.zeros((height,width))

                    # computes the bounding box for the contour, and draws it on the frame,
                    for contour, hier in zip(contours, hierarchy[0]):
                        (x,y,w,h) = cv2.boundingRect(contour)
                        min_x, max_x = min(x, min_x), max(x+w, max_x)
                        min_y, max_y = min(y, min_y), max(y+h, max_y)
			   
                    cv2.rectangle(im2, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

                    x=int((min_x+max_x)/2)
                    y=int((min_y+max_y)/2)
			
                    cv2.circle(new_image, (x,y), 1,(255,0,0), thickness=-1, lineType=8, shift=0)
					
                    #cv2.drawContours(im2,contours,0,(0,255,0),2)			
                    cv2.imshow('contours with rectangle',im2)			
                    cv2.imshow('just the circle',new_image)
                    cv2.waitKey(0)
                    ret,frame = cap.read()

                    foreground_mask=foreground_background.apply(frame)
                    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break
                    xary[f]=x
                    yary[f]=y	
            #for loop ends here
            #applying linear regression
            c,m = estimate_coef(xary, yary)

            #calculating the correct y so as to get it on the line
            for i in range(arr_size):
                yary[i]=m*xary[i]+c
                #print('y is {} and x is {}'.format(yary[i],xary[i]))
            distance=0;
            for i in range(arr_size-1):
                distance=distance+math.sqrt((xary[i]-xary[i+1])*(xary[i]-xary[i+1])+(yary[i]-yary[i+1])*(yary[i]-yary[i+1]))
            speed=int(distance/arr_size)
            #speed=speed+ma th.sqrt((xary[arr_size-1]-xary[0])*(xary[arr_size-1]-xary[0])+(yary[arr_size-1]-yary[0])*(yary[arr_size-1]-yary[0]))
			
            a.write('%i\n' % speed)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    plot_from_test()
    cap.release()
    cv2.closeAllWindows()
    return 
    
    
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x) 
    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y) 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x) 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x 
    return(b_0, b_1)
 
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    #plt.scatter(x, y, color = "m",
    #          marker = "o", s = 30)
    # predicted response vector
    y_pred = b[0] + b[1]*x
    # plotting the regression line
    #plt.plot(x, y_pred, color = "g")
    # putting labels
    #plt.xlabel('x')
    #plt.ylabel('y')
    # function to show plot
    #plt.show()
    
def plot_from_test():
    X, Y = [], []
    for line in open('test.txt', 'r'):
        values = [float(s) for s in line.split()]
        X.append(line)
        Y.append(values[1])
    plt.plot(X, Y)
    plt.show()
    
# get the ground truth label (as a number) from the name of the video
def get_video_label(srcVid):
    if "boxing" in srcVid:
        return 0
    elif "handclapping" in srcVid:
        return 1
    elif "handwaving" in srcVid:
        return 2
    elif "jogging" in srcVid:
        return 3
    elif "running" in srcVid:
        return 4
    elif "walking" in srcVid:
        return 5
    
if __name__ == '__main__':
    import kth_read_sequences as sequences
    locations = sequences.read_sequences_file(os.path.join(DATASET, SEQUENCES_FILE))
    
    # extract features from the training set videos
    srcVideoFolder = os.path.join(DATASET, TRAIN)
    destFolder = DEST_FOLDER
    train_df = extract_features(srcVideoFolder, locations)
