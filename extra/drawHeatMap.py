from __future__ import division
import cv2
import numpy as np


in_file = 'test_nesting_length/results/errorNew.txt'


def addBorders(img, clr=[0,0,0], val=5):
    ht, wd = img.shape[:2]
    img = cv2.copyMakeBorder(img,val,val,val,val,cv2.BORDER_CONSTANT,value=clr)
    return img


with open(in_file) as f:
    f = f.readlines()

num_x = 8
num_y = 10
box_dim_x = 200
box_dim_y = 150

white_box_x = int(round(box_dim_x*0.5))
white_box_y = int(round(box_dim_y*0.25))
font = cv2.FONT_HERSHEY_DUPLEX

img = np.ndarray([num_y*box_dim_y, num_x*box_dim_x, 3], dtype='uint8')
height = img.shape[0]
print img.shape
img.fill(255)
img[:,:,0]=0
for line in f:
    nest_val = int(line[10])
    err_idx = line.find('error')
    len_idx = line.find('length')
    #print line[len_idx+9:err_idx-1], len_idx+8, err_idx-2
    length_val = int(line[len_idx+9:err_idx-1])
    error_val = 1 - float(line[err_idx+8:-1])
    #print nest_val, length_val
    #print nest_val, length_val,(nest_val-1)*box_dim,(nest_val)*box_dim, (length_val-1)*box_dim,(length_val)*box_dim
    #print int(round(255*error_val))

    box_end_y = height - (length_val-1)*box_dim_y
    box_start_y = height - (length_val)*box_dim_y
    box_start_x = (nest_val-1)*box_dim_x
    box_end_x = (nest_val)*box_dim_x
    box_center_y = int((box_start_y+box_end_y)/2)
    box_center_x = int((box_start_x+box_end_x)/2)

    #Color Image
    img[box_start_y:box_end_y, box_start_x:box_end_x, 1] = int(round(255*error_val))
    img[box_start_y:box_end_y, box_start_x:box_end_x, 0] = int(round(50*(1-error_val)))
    #img[box_start_y:box_end_y, box_start_x:box_end_x, 2] = int(round(255*error_val))

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


count_temp = 1
for line in f:
    nest_val = int(line[10])
    err_idx = line.find('error')
    len_idx = line.find('length')
    length_val = int(line[len_idx+9:err_idx-1])
    error_val = 1 - float(line[err_idx+8:-1])

    box_end_y = height - (length_val-1)*box_dim_y
    box_start_y = height - (length_val)*box_dim_y
    box_start_x = (nest_val-1)*box_dim_x
    box_end_x = (nest_val)*box_dim_x
    box_center_y = int((box_start_y+box_end_y)/2)
    box_center_x = int((box_start_x+box_end_x)/2)

    #Add white box
    white_box_start_y = int(box_center_y - white_box_y/2)
    white_box_end_y = int(box_center_y + white_box_y/2)
    white_box_start_x = int(box_center_x - white_box_x/2)
    white_box_end_x = int(box_center_x + white_box_x/2)

    img[white_box_start_y:white_box_end_y, white_box_start_x:white_box_end_x] = [255,255,255]

    #Add text
    error_val = int(100*(error_val))
    text_start_x = int(box_center_x - white_box_x/(2*1.5))
    text_start_y = int(box_center_y + white_box_y/(2*2))
    #print error_val, nest_val, length_val, count_temp, text_start_x, text_start_y
    print "nesting {} length {} error {}".format(nest_val, length_val, error_val)
    count_temp += 1
    cv2.putText(img,str(error_val)+'%',(text_start_x, text_start_y), font, 1.0,(0,0,0),2)
    #,cv2.LINE_AA





#add border
img = addBorders(img)

#add axis values
img = addBorders(img, [255,255,255], 60)

#add y values
for y in range(10,0,-1):
    pos_x = 10
    pos_y = 10+136+(10-y)*box_dim_y
    print 'putting ', y, pos_x, pos_y
    cv2.putText(img,str(y),(pos_x, pos_y), font, 1.0,(0,0,0),2)

#add x values
print img.shape
for x in range(1,9):
    pos_x = 10+130+(x-1)*box_dim_x
    pos_y = img.shape[0] - 10
    cv2.putText(img,str(x),(pos_x, pos_y), font, 1.0,(0,0,0),2)


cv2.imshow('img',img); cv2.waitKey()
#import matplotlib.pyplot as plt
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()



##Color scale
scale = np.ndarray([255, 20, 3], dtype='uint8')
scale.fill(255)
scale[:,:,0]=0
val = np.array([i for i in range(255)])
val = val.reshape(255,1)
val = np.repeat(val, 20, axis = 1)
scale[:,:,1] = val
scale[:,:,0] = 50*(1 - val/255.0)
scale = cv2.cvtColor(scale, cv2.COLOR_HSV2BGR)

scale = addBorders(scale,val=1)

cv2.imshow('scale',scale); cv2.waitKey()
import matplotlib.pyplot as plt
plt.imshow(scale)
plt.show()
