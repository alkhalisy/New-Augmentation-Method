import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import os.path
import shutil
import random
#import imgaug
import albumentations as A


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
            rounds,
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def bg_remover_mediapip_deep(frame):
    # blured background of the frame preprossing using open cv and pixelmab
    change_background_mp = mp.solutions.selfie_segmentation
    change_bg_segment = change_background_mp.SelfieSegmentation()
    #sample_img = cv2.imread('media/sample11.jpg')
    RGB_sample_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = change_bg_segment.process(RGB_sample_img)
    binary_mask = result.segmentation_mask > 0.1
    binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))
    output_image = np.where(binary_mask_3, frame, 255)
    path = 'backgroundimages'
    file = random.choice(os.listdir("backgroundimages/"))
    bg_img = cv2.imread(os.path.join(path, file))
    bg_img = resized = cv2.resize(bg_img, (640,640), interpolation = cv2.INTER_AREA)

    #bg_img = cv2.imread('backgroundimages/1.jpg')
    output_image = np.where(binary_mask_3, frame, bg_img)
    #image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    # RGB_sample_img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    #cv2.imshow('E_proctoring', image)
    return output_image


#RGB_sample_img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

#cv2.imshow('E_proctoring', RGB_sample_img)
#CDir = os.getcwd()
#DSDir = CDir + "\dataset"
#print(DSDir)
for filename in os.listdir(os.getcwd()):


    if filename.endswith(".jpg"):
        image = cv2.imread(filename)
        # resize_image(image, 450, 400)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        transform = A.Compose([
            A.CLAHE(p=1),
            A.GaussNoise(),
            A.RandomGamma(p=1),
            A.OpticalDistortion(),
            A.HueSaturationValue(0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=1),
            A.RandomBrightnessContrast(),
        ])
        #random.seed(7)


        # result = kmeans_color_quantization(image, clusters=32) #image quantization using k-mean
        out_put_image = bg_remover_mediapip_deep(image)
        augmented_image = transform(image=out_put_image)['image']
        cv2.imwrite("removed_bg" + "/" + "aug" + filename, augmented_image)
        print(filename)
        # saturation_image(image, 100)
        # saturation_image(image, 150)
        # saturation_image(image, 200)

    if filename.endswith(".txt"):
        print("ok text")
        dest = shutil.copyfile(filename, "removed_bg" + "/" + "aug" + filename)
        # image = cv2.imread(filename)
        # resize_image(image, 450, 400)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # result = kmeans_color_quantization(image, clusters=32) #image quantization using k-mean
        # out_put_image = bg_remover_mediapip_deep(image)
        # cv2.imwrite("removed_bg" + "/" + "aug" + filename, out_put_image)
        print(filename)
        # saturation_image(image, 100)
        # saturation_image(image, 150)
        # saturation_image(image, 200)

