import skimage
import numpy as np
import utils


def MaxPool2d(im: np.array,
              kernel_size: int):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    stride = kernel_size
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    
    # Create image with correct shrinked output shape. 
    new_im = np.empty(shape=[im.shape[0] // stride, im.shape[1] // stride, 3])

    for row in range(0, new_im.shape[0]):
        for col in range(0, new_im.shape[1]):
            # Pool each channel separately.
            for channel in range(im.shape[2]):
                # Slice matrix and find maximum value in pool.
                new_im[row, col, channel] = np.max(im[
                    stride * row : stride * (row + 1),
                    stride * col : stride * (col + 1),
                    channel,
                ])

    return new_im

    ### END YOUR CODE HERE ### 


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)