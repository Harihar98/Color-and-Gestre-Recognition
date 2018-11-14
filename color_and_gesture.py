import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
        # read image
        ret, img = cap.read()

        # color recognition

        #converting frame(img i.e BGR) to HSV (hue-saturation-value)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #definig the range of red color
        red_lower=np.array([136,87,111],np.uint8)
        red_upper=np.array([180,255,255],np.uint8)

	#defining the Range of Blue color
        blue_lower=np.array([99,115,150],np.uint8)
        blue_upper=np.array([110,255,255],np.uint8)
	
	#defining the Range of yellow color
        yellow_lower=np.array([22,60,200],np.uint8)
        yellow_upper=np.array([60,255,255],np.uint8)

	#finding the range of red,blue and yellow color in the image
        red=cv2.inRange(hsv, red_lower, red_upper)
        blue=cv2.inRange(hsv,blue_lower,blue_upper)
        yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
	
	#Morphological transformation, Dilation
        kernal = np.ones((5 ,5), "uint8")

        red=cv2.dilate(red, kernal)
        res=cv2.bitwise_and(img, img, mask = red)

        blue=cv2.dilate(blue,kernal)
        res1=cv2.bitwise_and(img, img, mask = blue)

        yellow=cv2.dilate(yellow,kernal)
        res2=cv2.bitwise_and(img, img, mask = yellow)    


	#Tracking the Red Color
        (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
			
	#Tracking the Blsue Color
        (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

	#Tracking the yellow Color
        (_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  
            
	#cv2.imshow("Redcolour",red)
        cv2.imshow("Recognition",img)
    	#cv2.imshow("red",res)
        if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


        # gesture recognition
        
        # get hand data from the rectangle sub window on the screen
        #cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
        #print(cv2.getWindowImageRect('Recognition'))
        cv2.rectangle(img, (300,300), (0,0), (0,255,0),0)
        #crop_img = img[100:300, 100:300]
        crop_img = img[0:300, 0:300]
    
        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        # applying gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)
    
        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # show thresholded image
        cv2.imshow('Thresholded', thresh1)
    
        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')
    
        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
    
        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)
    
        # finding convex hull
        hull = cv2.convexHull(cnt)
    
        # drawing contours
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)
    
        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)
    
        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    
        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
    
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
    
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
    
            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0,0,255], -1)
            #dist = cv2.pointPolygonTest(cnt,far,True)
    
            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(crop_img,start, end, [0,255,0], 2)
            #cv2.circle(crop_img,far,5,[0,0,255],-1)
    
        # define actions required
        if count_defects == 1:
            cv2.putText(img,"Welcome!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
            str = "This is a basic hand gesture recogniser"
            cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img,"This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img,"Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img,"Hello World!!!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show appropriate images in windows
        cv2.imshow('Recognition', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break
    
