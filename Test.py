"""
@author: arjavjain
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

trainImg = cv2.imread('foto1.jpg')
trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2RGB)
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)

queryImg = cv2.imread('foto2.jpg')
queryImg = cv2.cvtColor(queryImg, cv2.COLOR_BGR2RGB)
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

def detectAndDescribe(image):
    descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)

kpsA, featuresA = detectAndDescribe(trainImg_gray)
kpsB, featuresB = detectAndDescribe(queryImg_gray)

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_matches = bf.match(featuresA,featuresB)
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

matches = matchKeyPointsBF(featuresA, featuresB, method='orb')
img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100],
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


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

M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
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
cv2.imwrite('result.jpg', result_rgb) 
plt.figure(figsize=(20,10))
plt.imshow(result)