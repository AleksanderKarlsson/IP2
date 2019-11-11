import utils
import skimage
import skimage.morphology
import numpy as np


def remove_noise(im: np.ndarray) -> np.ndarray:
    """
        A function that removes noise in the input image.
        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    # Shave values off the edge of the structuring element. Gives smoother result shape.
    B = np.zeros(shape=[15, 15])
    start, end = skimage.draw.circle(7, 7, 8)
    B[start, end] = 1

    # Perform opening, then closing.
    skimage.morphology.binary_opening(im, selem=B, out=im)
    skimage.morphology.binary_closing(im, selem=B, out=im)

    return im

    ### END YOUR CODE HERE ### 


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)

    assert im.shape == noise_free_image.shape, \
        "Expected image shape ({}) to be same as resulting image shape ({})".format(
            im.shape, noise_free_image.shape)
    assert noise_free_image.dtype == np.bool, \
        "Expected resulting image dtype to be np.bool. Was: {}".format(
            noise_free_image.dtype)

    noise_free_image = utils.to_uint8(noise_free_image)
    utils.save_im("noisy-filtered.png", noise_free_image)

