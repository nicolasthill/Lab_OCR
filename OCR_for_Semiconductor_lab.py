# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:47:54 2021

@author: nicol
"""
import cv2
import numpy as np
import glob as glob
import imutils
import random
import pandas as pd

from imutils.perspective import four_point_transform
from imutils import contours

image_names = glob.glob("images/*.jpg")
image_list = [cv2.imread(image) for image in image_names]


class reading():
    dist = 20
    def __init__(self, img, name):
        self.base_img = img
        self.dim_x = self.base_img.shape[1]
        self.dim_y = self.base_img.shape[0]
        self.name = name
        
    def initiate_settings(self, name = ''):
        try:
            temp = np.genfromtxt(self.name + '.csv', delimiter=',')
            self.x = int(temp[0])
            self.y = int(temp[1])
            self.w = int(temp[2])
            self.h = int(temp[3])
            self.angle = temp[4]
            self.base_img = imutils.rotate(self.base_img, self.angle)
        except OSError:
            self.x = int(self.dim_x/2)
            self.y = int(self.dim_x/2)
            self.h = int(self.dim_x/4)
            self.w = int(self.dim_x/4)
            self.angle = 0
            pass
    
    def view(self):
        cv2.imshow("Current View " + self.name, self.return_window())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def return_window(self):
        self.window = self.base_img[self.y:self.y+self.h, self.x:self.x+self.w]
        return self.window
        
    def move_x(self, value):
        self.x = int(self.x + value)

    def move_y(self, value):
        self.y = int(self.y + value)
    
    def stretch_w(self, factor):
        self.w = int(self.w * factor)
        
    def stretch_h(self, factor):
        self.h = int(self.h * factor)
    
    def select_view(self):
        
        EXIT = False
        while EXIT == False:
            self.view()
            user_input = input("Type U/D, L/R, +/-W, +/-H or EXIT: ")
            
            if user_input == "U":
                self.move_y(self.dist)
            elif user_input == "D":
                self.move_y(-self.dist)
            elif user_input == "L":
                self.move_x(-self.dist)
            elif user_input == "R":
                self.move_x(self.dist)
            elif user_input == "+W":
                self.stretch_w(1.25)
            elif user_input == "-W":
                self.stretch_w(0.75)
            elif user_input == "+H":
                self.stretch_h(1.25)
            elif user_input == "-H":
                self.stretch_h(0.75)
            elif user_input == "EXIT":
                break
            else:
                pass
            
    def save_settings(self):
        
        sett = np.array([self.x, self.y, self.w, self.h, self.angle])
        np.savetxt(self.name +".csv", sett, delimiter=",")
        
    def rotate(self, angle = None):
        if angle == None:
            angle = input("Please input rotation angle: ")
        self.base_img = imutils.rotate(self.base_img, angle)
        self.angle = self.angle + angle
        self.view()
        pass
    
def print_pic(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# prepare window makers
reading0 = reading(image_list[0], "voltage")
reading0.initiate_settings()
reading1 = reading(image_list[0], "current")
reading1.initiate_settings()
reading2 = reading(image_list[0], "temperature")
reading2.initiate_settings()

readings = [reading0, reading1, reading2]

# make window list
images = []
for image in image_list:
    windows = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for reading in readings:
        image = imutils.rotate(image, reading.angle)
        windows.append(image[reading.y:reading.y+reading.h,reading.x:reading.x+reading.w])
    images.append(windows)
    
DIGITS_LOOKUP = {
    (0, 0, 0, 0, 0, 0, 0): " ",
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 0): 9
}

def polygonArea(X, Y, n):
 
    # Initialze area
    area = 0.0
 
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
     
 
    # Return absolute value
    return int(abs(area / 2.0))

def print_pic(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

class raster():
    dist = 20
    def __init__(self, img, name, strength):
        self.base_img = img
        self.dim_x = self.base_img.shape[1]
        self.dim_y = self.base_img.shape[0]
        self.name = name
        self.coord_list = []
        self.x = 50
        self.y = 50
        self.w = 10
        self.h = 10
        self.n = 0
        self.spacing = 20
        self.strength = strength
        self.diag = 0
        
    def initiate_coords(self, name = ''):
        try:
            self.coord_list = np.genfromtxt(self.name + '.csv',dtype = int, delimiter=',')
        except OSError:
            print("ERROR")
            pass
        
    def initiate_params(self):
        self.x = self.coord_list[0][1]
        self.y = self.coord_list[0][0]
        self.w = int((self.coord_list[2][1]-self.coord_list[1][1] + self.coord_list[5][1]-self.coord_list[4][1])/2)
        self.h = int((self.coord_list[6][0]-self.y)/2)
        self.n = int(len(self.coord_list)/7) - 1
        if self.n!=0:
            self.spacing = self.coord_list[7][1]-self.coord_list[0][1]
        
    def update_coords(self):
        self.coord_list = []
        for i in range(self.n+1):
            self.coord_list.append([self.y, self.x+i*self.spacing])
            self.coord_list.append([self.y+int(self.h/2),self.x+i*self.spacing-int(self.w/2)-self.diag])
            self.coord_list.append([self.y+int(self.h/2),self.x+i*self.spacing+int(self.w/2)-self.diag])
            self.coord_list.append([self.y+self.h,self.x+i*self.spacing-self.diag])
            self.coord_list.append([self.y+int(3*self.h/2),self.x+i*self.spacing-int(self.w/2)-2*self.diag])
            self.coord_list.append([self.y+int(3*self.h/2),self.x+i*self.spacing+int(self.w/2)-2*self.diag])
            self.coord_list.append([self.y+2*self.h,self.x+i*self.spacing-2*self.diag])
    
    def view(self, image, n):
        cv2.imshow("Current View " + self.name, self.return_window(image, n))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def return_window(self, image,n):
        self.update_coords()
        window = image.copy()
        if n == 69:
            liste = self.coord_list
        else:
            liste = self.coord_list[7*n:7*n+7]
        for c in liste:
            window[c[0]:c[0]+5,c[1]:c[1]+5] = 255
        return window
        
    def move_x(self, value):
        self.x = int(self.x + value)

    def move_y(self, value):
        self.y = int(self.y + value)
    
    def stretch_w(self, factor):
        self.w = int(self.w * factor)
        
    def stretch_h(self, factor):
        self.h = int(self.h * factor)
        
    def add_n(self, value):
        self.n = self.n + value

    def save_coords(self):
        np.savetxt(self.name +".csv", self.coord_list, delimiter=",")
        
    def return_averages(self,image):
        chars =[]
        for n in range(self.n):
            char = []
            for c in range(7):
                
                char.append(cv2.mean(image[self.coord_list[c+7*n][0]:self.coord_list[c+7*n][0]+5,self.coord_list[c+7*n][1]:self.coord_list[c+7*n][1]+5])[0])
            chars.append(char)
        
        return chars
        

r0 = raster(images[0][0], "r0", 75)
r0.diag = 2
r0.initiate_coords()
r0.initiate_params()
r0.n = 7

r1 = raster(images[0][1], "r1", 35)
r1.diag = 2
r1.initiate_coords()
r1.initiate_params()
r1.n = 7

r2 = raster(images[0][2], "r2", 125)
r2.diag = 4
r2.initiate_coords()
r2.initiate_params()

rasters = [r0,r1,r2]

print("Lets Do THIS!")
image_data = []
for windows in images:
    numbers = []
    for w, window in enumerate(windows):
        #print("Looking at image ", image, " Window ", w)
        
        chars = rasters[w].return_averages(window)
        number = ""
        for index,char in enumerate(chars):
            #print(number)
            try:
                liste = []
                for i in char:
                    #print("Values is ", i, " and threshold ", rasters[w].strength, " thus we find ",i<rasters[w].strength)
                    liste.append(i<rasters[w].strength)
                    
                    
                tup = tuple(liste)
                digit = str(DIGITS_LOOKUP[tup])
                #print(digit)
                number += digit
                
            except KeyError:
                #print("does not correspond to a digit")
                number += "X"
        numbers.append(number)
        print("Result for this image: ", number)
        #rasters[w].view(window, index)
    image_data.append(numbers)
print(len(image_data), len(image_list))      
data = image_data.copy()
for i in data:
    i[0] = i[0][:2] + "." + i[0][-5:]
    i[1] = i[1][3:6] + "." + i[1][-1]
    if i[2][0] == " ":
        i[2] = i[2][:1] + "." + i[2][-3:]
    else:
        i[2] = i[2][:2] + "." + i[2][-2:]
    print(i)
df = pd.DataFrame(data)
df.to_csv("results.csv", sep = ';')
"""
for window in images:
    for window in windows:
        image = imutils.resize(window, height=100)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        print_pic(thresh)
        
        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            if w >= 20 and h >= 50:
                digitCnts.append(c)
                print_pic()
"""
    