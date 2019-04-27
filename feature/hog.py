__author__ = 'sachh'
#import cv2
#import glob
import sys
sys.path.append('../')

from skimage import color, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import skimage.io
from config import Project

#load the dataset

#converting the RGB image to sillhouette
def get_1d_2d_hog(img):
    #img.shape returns (345, 456, 3) for RGB
    if len(img.shape) >= 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    hog_image_1d, hog_image_2d = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                     cells_per_block=(1, 1), visualise=True)
    return hog_image_1d, hog_image_2d


def get_hog(img):
    """
    :param img: the 2d rbg image, represented by numpy
    :return: list of feature
    """
    hog_image_1d, hog_image_2d = get_1d_2d_hog(img)
    hog = list(hog_image_1d)
    res = [int(x * 100) for x in hog]
    return res

#for future training implementation
def flatten(img):
    return list(img.flatten())


if __name__ == '__main__':
    #create a list of names of images
    #img list, append names as %03d x for x in (0, )

    img = skimage.io.imread("%s/GEI.bmp" % Project.test_data_path)
    #for img in glob.glob("D:/gait-energy-image-recognition-gei/data/test/*.bmp"):
    #    img = cv2.imread(img)
    hog_image_1d, hog_image_2d = get_1d_2d_hog(img)
    #call hog function as is created
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image_2d, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()