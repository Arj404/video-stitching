"""
@author: arjavjain
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def detectAndDescribe(image):
	    descriptor = cv2.ORB_create()
	    (kps, features) = descriptor.detectAndCompute(image, None)
	    return (kps, features)


def matchKeyPointsBF(featuresA, featuresB, method):
	    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	    best_matches = bf.match(featuresA,featuresB)
	    rawMatches = sorted(best_matches, key = lambda x:x.distance)
	    #print("Raw matches (Brute force):", len(rawMatches))
	    return rawMatches


def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
	    kpsA = np.float32([kp.pt for kp in kpsA])
	    kpsB = np.float32([kp.pt for kp in kpsB])
	    
	    if len(matches) > 4:
	        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
	        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
	        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
	            reprojThresh)
	        return (matches, H, status)
	    else:
	        return None


def stitch(image1,image2):
	trainImg = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
	
	queryImg = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
	queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
	
	kpsA, featuresA = detectAndDescribe(trainImg_gray)
	kpsB, featuresB = detectAndDescribe(queryImg_gray)
	matches = matchKeyPointsBF(featuresA, featuresB, method='orb')
	img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:200],
	                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	
	M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=5)
	if M is None:
	    print("Error!")
	(matches, H, status) = M
	width = trainImg.shape[1] + queryImg.shape[1]
	height = trainImg.shape[0] + queryImg.shape[0]
	
	result = cv2.warpPerspective(trainImg, H, (width, height))
	result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
	
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	result = result[y:y + h, x:x + w]
	result_rgb = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	print(result_rgb.shape)
	return result_rgb


def video_stitch(url,f):
	cap = cv2.VideoCapture(url)
	count = 0
	f_list = []
	height = 0
	width = 0
	while cap.isOpened():
	    ret,frame = cap.read()
	    if ret == True:
	    	if (count == 0):
	    		result = frame
	    	elif(count%f==0):
	    		result = stitch(result,frame)
	    		f_list.append(result)
	    	(h, w) = result.shape[:2]
	    	if(h>height):
	    		height = h
	    	if(w>width):
	    		width = w
	    	r_frame = ResizeWithAspectRatio(result, height=300)
	    	#cv2.imshow('output',r_frame)
	    	count = count + 1
	    	if cv2.waitKey(10) & 0xFF == ord('q'):
	        	break
	    else:
	    	break
	cap.release()
	return f_list


v_url = './video.mp4'
frame_list = []
frame_list = video_stitch(v_url,10)
cv2.imshow('output2', frame_list[0])
#print(wi)
#out = cv2.VideoWriter('project.avi',0,1, (wi,he))
#for i in range(len(frame_list)):
 #   cv2.imshow('output2',frame_list[i])
  #  time.sleep(2)
    #out.write(frame_list[i])
   # print('frame added')
#out.release()

cv2.destroyAllWindows()  # destroy all the opened windows
