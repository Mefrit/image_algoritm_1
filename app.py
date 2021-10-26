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
        # /contourIdx  что каждая точка в вашем контуре является отдельным контуром\типо толшина
        # thickness яркость
    #     for c in contours:
    #     #    img2 = img.copy()
    # #    cv2.waitKey(0)
    #         cv2.drawContours(image_copy, c, -1, (0, 255, 0), 2)
        output = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(22, 222, 30), thickness=4, lineType=cv2.LINE_AA)
        # # tmp= flood_fill_single(tmp, (110,110))
        cv2.imshow('contours', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(path2folder+ "/binary/rgb/green_rgb_fields_new.jpg",image_copy - tmp)
      

    if(prefix == "lab"):
      
        tmp = findObjects(b, 85,73)
        cv2.imwrite(path2folder+ "/binary/lab/blue_lab_binary.jpg",tmp)

        contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = tmp.copy()

        # cv2.drawContours( tmp, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
       
        output = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(22, 222, 30), thickness=3, lineType=cv2.LINE_AA)
        cv2.imwrite(path2folder+ "/binary/lab/blue_lab_fields_new.jpg",output - tmp)

# def flood_fill_single(im, seed_point):
#     """Perform a single flood fill operation.

#     # Arguments
#         image: an image. the image should consist of white background, black lines and black fills.
#                the white area is unfilled area, and the black area is filled area.
#         seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
#     # Returns
#         an image after filling.
#     """
#     pass1 = np.full(im.shape, 255, np.uint8)

#     im_inv = cv2.bitwise_not(im)

#     mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
#     _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

#     return pass1 
# def fill_hole(input_mask, point):
#     h, w = input_mask.shape
#     canvas = np.zeros((h + 2, w + 2), np.uint8)
#     canvas[1:h + 1, 1:w + 1] = input_mask.copy()

#     mask = np.zeros((h + 4, w + 4), np.uint8)

#     cv2.floodFill(canvas, mask, (0, 0), 1)
#     canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

#     return (~canvas | input_mask.astype(np.uint8))
def getImageLayers(image_src, path2folder):
    # cache=["blue","green","red"]
    # i = 0
    dark_lab = cv2.cvtColor(image_src, cv2.COLOR_BGR2LAB)


    setImageLayersByPresentationMethod(img,path2folder,"rgb")
    setImageLayersByPresentationMethod(dark_lab,path2folder,"lab")


    # while i < len(cache):
    #     new_image_bjr = getColorLayerImg(image_src,i)
    #     cv2.imwrite(path2folder + "/BGR-" + cache[i] + "-channel.jpg",new_image_bjr)
    #     if(cache[i] == "green"):
    #         findObjects(new_image_bjr, 60,path2folder+ "/binary/green_bgr_binary.jpg")
            
    #     new_image_lab = getColorLayerImg(dark_lab,i)
    #     cv2.imwrite(path2folder + "/LAB-" + cache[i] + "-channel.jpg",new_image_lab)
    #     if(cache[i] == "blue"):
    #         findObjects(new_image_lab,80, path2folder+ "/binary/blue_lab_binary.jpg")
    #     i+=
    # 1
    return;
    

img = cv2.imread("./images/default/robo1.jpg", cv2.IMREAD_UNCHANGED)

getImageLayers(img, "./images/results/")