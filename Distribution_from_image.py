#! /bin/python3
#script Distribution_from_Image.py
''' generate 2d random numbers
distributed according to brightness of an image
.. author:: Guenter Quast <g.quast@kit.edu>
.. author:: Alexander Becker <nabla.becker@mailbox.org>
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

#open an image
def read_image(filepath) -> Image:
    """ Read in the image from the given filepath and catch possible errors"""
    try:
        img = Image.open(filepath)
        return img
    except FileNotFoundError:
        print("The file with the path {} could not be found. Exiting".format(filepath))
        sys.exit()
    except UnidentifiedImageError:
        print("The image formate of {} could not be identified. Exiting".format(filepath))
        sys.exit()

def rand_from_image(sample_size, image, invert=False) -> (np.ndarray, np.ndarray):
    """ generates a set of random numbers from an image

    Generate a set of `sample_size` random numbers conforming to
    a distribution given by the brightness of the given image
    using the rejection method

    Keyword arguments:
    invert -- if invert is set then the number of accepted points falls with increasing brightness
    """
    # convert the image into grayscale
    image = image.convert("L")

    # convert the image into a numpy array to use as 2D distribution
    im_array_size = (image.size[1], image.size[0])
    im_array = np.array(image.getdata()).reshape(im_array_size)

    # get a rough estimate of how many random numbers to generate
    inverse_rel_brightness = (255*image.size[0]*image.size[1])/sum(im_array.flatten())

    # generate random numbers until we have sample_size accepted ones
    accepted_points = []
    while len(accepted_points) < sample_size:
        # how many events still to generate
        samples_left = sample_size - len(accepted_points)
        samples_to_generate = int(samples_left*inverse_rel_brightness)

        # generate random tuples as coordinates
        coordinates = [np.random.randint(0, im_array_size) for i in range(samples_to_generate)]

        # generate a tuple of coordinate and brightness
        points_with_brightness = list(zip(np.random.randint(0, 255, size=len(coordinates)), coordinates))
        print(points_with_brightness[0])
        # filter out the random numbers that lie outside the distribution
        # and add the rest them to the accepted points list
        if invert:
            filtered_points = filter(lambda x: x[0] > im_array[x[1][0],x[1][1]], points_with_brightness)
        else:
            filtered_points = filter(lambda x: x[0] < im_array[x[1][0],x[1][1]], points_with_brightness)

        for elem in filtered_points:
            accepted_points.append(elem[1])

    # return an array for the x and one for the y coordinate of the
    # accepted points and transform coodrinate system into right handed one
    # (pictures start at the top left with y increasing downwards -> left handed coordinate system)
    return [p[1] for p in accepted_points[:sample_size]], [image.size[1] - p[0] for p in accepted_points[:sample_size]]

# -----------------------------------------------------------------------
if __name__ == "__main__":
    # load image
    IMAGE = read_image(sys.argv[1])
    X, Y = rand_from_image(20000, IMAGE)

    # plot distributions
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    plt.hist(Y, 50) # histogram data
    ax1.set_title('y projection ')

    ax4 = fig.add_subplot(2, 2, 4) # x histogram and statistics
    plt.hist(X, 50) # histogram data
    ax4.set_title('x projection')

    ax2 = fig.add_subplot(2, 2, 2) # 2d historgram and statistics
    plt.hist2d(X, Y, 50, cmap='Blues')
    ax2.set_title('2d distribution from image')

    plt.show()
