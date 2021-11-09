import cv2
import numpy as np

def loading_displaying_saving(path2image):
    img = cv2.imread(path2image, cv2.IMREAD_UNCHANGED)
    return img

def getColorLayerImg(img,code_chanel):
    chanel = img[:,:,code_chanel]
    new_image = np.zeros(img.shape)
    new_image[:,:,code_chanel] = chanel
    return new_image

def findObjects(img,min_value1,min_value2):
    
    ret,tresh_image1 = cv2.threshold(img,min_value1,255,cv2.THRESH_BINARY )
    ret,tresh_image2 = cv2.threshold(img,min_value2,255,cv2.THRESH_BINARY )
    
    return cv2.add(tresh_image1, tresh_image2)
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def setImageLayersByPresentationMethod(img,path2folder,prefix):
    # img = increase_brightness(img,30) 
    b,g,r = cv2.split(img)
    cv2.imwrite(path2folder + "/"+prefix+"_b_binary_v2_split.jpg",b)
    cv2.imwrite(path2folder + "/"+prefix+"_g_binary_v2_split.jpg",g)
    cv2.imwrite(path2folder + "/"+prefix+"_r_binary_v2_split.jpg",r)
    if(prefix == "rgb"):
      
        tmp = findObjects(g, 70,63)
        cv2.imwrite(path2folder+ "/binary/rgb/green_rgb_binary.jpg",tmp)
        contours, hierarchy = cv2.findContours(tmp, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

        image_copy = tmp.copy()
        output = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(22, 222, 30), thickness=4, lineType=cv2.LINE_AA)
  
        cv2.imshow('contours', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(path2folder+ "/binary/rgb/green_rgb_fields_new.jpg",image_copy - tmp)
      

    if(prefix == "lab"):
      
        tmp = findObjects(b, 85,73)
        cv2.imwrite(path2folder+ "/binary/lab/blue_lab_binary.jpg",tmp)

        contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = tmp.copy()

       
        output = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(22, 222, 30), thickness=3, lineType=cv2.LINE_AA)
        cv2.imwrite(path2folder+ "/binary/lab/blue_lab_fields_new.jpg",output - tmp)

def getImageLayers(image_src, path2folder):

    dark_lab = cv2.cvtColor(image_src, cv2.COLOR_BGR2LAB)
    setImageLayersByPresentationMethod(img,path2folder,"rgb")
    setImageLayersByPresentationMethod(dark_lab,path2folder,"lab")
    return;
    

img = cv2.imread("./images/default/robo3.jpg", cv2.IMREAD_UNCHANGED)

getImageLayers(img, "./images/results/")