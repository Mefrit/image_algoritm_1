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
  
    # ret,tresh_image1 = cv2.threshold(img,min_value1,255,cv2.THRESH_BINARY )
    # ret,tresh_image2 = cv2.threshold(img,min_value2,255,cv2.THRESH_BINARY )
    # robo 3
    return cv2.inRange(img,5, 50)
    # robo 2
    # return cv2.inRange(img,100, 170)
    # return cv2.add(tresh_image1, tresh_image2)

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def setImageLayersByPresentationMethod(img,path2folder,prefix):
    
  

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
  
      
        cv2.imwrite(path2folder+ "/binary/rgb/green_rgb_fields_new.jpg",image_copy - tmp)
      

    if(prefix == "lab"):

        # cv2.imshow('b2',   b )
        # cv2.imshow('r3',   r )
        # cv2.imshow('g4',   g)
        # # cv2.imshow('tmp', dark_lab)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()        
        tmp = findObjects( b + b + g  ,155,10)
        # tmp = findObjects( b +b + g -r ,255,0)
        
       
        # cv2.imshow(' b  + g',    b +b + g -r )
        cv2.imwrite(path2folder+ "/binary/lab/blue_lab_binary.jpg",tmp)  
  
       
        image_copy = tmp.copy()
        
       
        # output = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(22, 222, 30), thickness=3, lineType=cv2.LINE_AA)
        kernel = np.ones((3,3), 'uint8')
        contours, hierarchy = cv2.findContours( image_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours( image_copy, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
        dilate_img = cv2.dilate(image_copy, kernel, iterations=1)
        kernel = np.ones((3, 3), 'uint8')
        erode_img = cv2.erode(dilate_img, kernel, iterations=1)
        # dilate_img_v2 = prepareImageV2(erode_img)
        # kernel = np.ones((3,3), 'uint8')
        # dilate_img_v2 = cv2.dilate(dilate_img_v2, kernel, iterations=1)
        # kernel = np.ones((4, 4), 'uint8')
        # erode_img2 = cv2.erode(erode_img, kernel, iterations=1)
        # erode_img_medianBlur = cv2.medianBlur(erode_img, 3)
        # cv2.imshow('erode_img',   erode_img )
        # cv2.imshow('erode_img_medianBlur 3111',   erode_img_medianBlur )
        # canny = cv2.Canny(dilate_img, 10, 250)
        cv2.imshow('image_copy',erode_img)
        # cv2.imshow('tmp',   prepareImageV2(image_copy-tmp ))
        # cv2.imshow('tmp prepared',   (tmp) )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(path2folder+ "/binary/lab/blue_lab_fields_new.jpg",tmp)

def nothing(args):pass
def clahe(img_yuv, clipLimit=2.0, tileGridSize=(5,5)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 2])
    cv2.imshow('img ',  img_yuv )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)

    return img_output 
def prepareImageV2(img):
#         # img_blur_3 = cv2.medianBlur(img, 3)
    # img_blur_7 = cv2.medianBlur(img, 7)
    img = cv2.medianBlur(img,3)
    # img_blur_11 = cv2.GaussianBlur(img_blur_11, (3, 3), 0)
# 
    return img
def prepareImage(img):
    # img_blur_3 = cv2.medianBlur(img, 3)
    # img_blur_7 = cv2.medianBlur(img, 7)
    img_blur_11 = cv2.medianBlur(img, 9)
    img_blur_11 = cv2.GaussianBlur(img_blur_11, (3, 3), 0)

    return img_blur_11
def getImageLayers(img, path2folder):

    # cv2.imshow('tmp1', img)
    # cv2.imshow('tmp2', increase_brightness(img,300))\
    img = increase_brightness(img,255)
    dark_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # setImageLayersByPresentationMethod(img,path2folder,"rgb")
    # clahe(img) 
    setImageLayersByPresentationMethod(dark_lab,path2folder,"lab")
    return;
    

img = cv2.imread("./images/default/robo3.jpg", cv2.IMREAD_UNCHANGED)
img = prepareImage(img)
getImageLayers(img, "./images/results/")