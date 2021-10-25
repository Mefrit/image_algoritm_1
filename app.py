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

def getImageLayers(image_src, path2folder):
    cache=["blue","green","red"]
    
    i = 0
    dark_lab = cv2.cvtColor(image_src, cv2.COLOR_BGR2LAB)
    while i < len(cache):
        new_image = getColorLayerImg(image_src,i)
        cv2.imwrite(path2folder+"/BGR-" + cache[i] + "-channel.jpg",new_image)
        new_image = getColorLayerImg(dark_lab,i)
        cv2.imwrite(path2folder+"/LAB-" + cache[i] + "-channel.jpg",new_image)
        i+=1
   

    return;


img = cv2.imread("./images/default/robo1.jpg", cv2.IMREAD_UNCHANGED)

getImageLayers(img, "./images/results/")