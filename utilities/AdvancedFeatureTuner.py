import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hsv

"""@package AdvancedFeatureTuner
# This is a class to demonstrate the advanced feature extraction methods,
# and to help with further tuning the model. Simply update the path below
# to that of an image to view the results of the filters and selected blighted pixels.
# Yellow represents 1 in the images, or a selected pixel.
"""

fileIn = 'C:/school/Ag-AI/images - Copy/blighted/DSC00200.JPG'

#Conservative values
lowRed = 165
highRed = 240
lowGreen = 160
highGreen = 200
lowBlue = 135
highBlue = 240

rgb_img = plt.imread(fileIn)
red = rgb_img[:, :, 0]
hsv_img = rgb2hsv(rgb_img)
hue_img = hsv_img[:, :, 0]
sat_img = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]

#saturation mask to isolate foreground
satMask = (sat_img > .11) | (value_img > .3)
#hue and value mask to remove additional brown from background
mask = (hue_img > .14) | (value_img > .48)
#healthy corn mask to remove healthy corn, leaving only blighted pixels
nonBlightMask = hue_img < .14

#get foreground
rawForeground = np.zeros_like(rgb_img)
rawForeground[mask] = rgb_img[mask]

#reduce brown in background
foreground = np.zeros_like(rgb_img)
foreground[satMask] = rawForeground[satMask]

#get blighted pixels from foreground
blightedPixels = np.zeros_like(rgb_img)
blightedPixels[nonBlightMask] = foreground[nonBlightMask]
#combine into one band
blightedHSV = np.bitwise_or(blightedPixels[:,:,0], blightedPixels[:,:,1])
blightedHSV = np.bitwise_or(blightedHSV, blightedPixels[:,:,2])

red = rgb_img[:, :, 0]
green = rgb_img[:, :, 1]
blue = rgb_img [:, :, 2]

binary_green = lowGreen < green
binary_blue = lowBlue < blue
binary_red = lowRed < red 

RGB_Blights = np.bitwise_and(binary_red, binary_green)
#'brown' pixels within each RGB threshold
RGB_Blights = np.bitwise_and(RGB_Blights, binary_blue)

HSV_and_RGB = np.bitwise_and(RGB_Blights, blightedHSV)

#get features
numForegroundPixels = np.count_nonzero(foreground)
numBlightedHSVPixels = np.count_nonzero(blightedHSV)
blightedHSVRatio = numBlightedHSVPixels / numForegroundPixels
num_RGB_blightedPixels = np.count_nonzero(RGB_Blights)
blightedRGBRatio = numForegroundPixels / num_RGB_blightedPixels
numBlightedBothPixels = np.count_nonzero(HSV_and_RGB)
blightedBothRatio = numForegroundPixels / numBlightedBothPixels

#display images
fig, axs = plt.subplots(3, 3)
axs[0, 0].imshow(rawForeground)
axs[0, 0].set_title('foreground')
axs[0, 0].axis('off')
axs[0, 1].imshow(foreground)
axs[0, 1].set_title('fore w/ mask')
axs[0, 1].axis('off')
axs[0, 2].imshow(blightedHSV)
axs[0, 2].set_title('HSV Blighted')
axs[0, 2].axis('off')

axs[1, 0].imshow(binary_red)
axs[1, 0].set_title('red')
axs[1, 0].axis('off')
axs[1, 1].imshow(binary_green)
axs[1, 1].set_title('green')
axs[1, 1].axis('off')
axs[1, 2].imshow(binary_blue)
axs[1, 2].set_title('blue')
axs[1, 2].axis('off')
axs[2, 0].imshow(rgb_img)
axs[2, 0].set_title('original')
axs[2, 0].axis('off')
axs[2, 1].imshow(RGB_Blights)
axs[2, 1].set_title('all clr masks')
axs[2, 1].axis('off')
axs[2, 2].imshow(HSV_and_RGB)
axs[2, 2].set_title('Blighted HSVandRGB')
axs[2, 2].axis('off')


