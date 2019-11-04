import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array,
                fft_kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)

    # Frequency domain representation of the original image.
    frequency_image = np.fft.fft2(im)
    # Visualize the frequency domain image.
    absolute_frequency_image = np.log(1 + np.abs(np.fft.fftshift(frequency_image)))
    
    # Convoluted image in the frequency domain.
    convoluted_frequency_image = frequency_image * fft_kernel
    # Visualize the frequency domain image convolution.
    absolute_convoluted_frequency_image = np.log(1 + np.abs(np.fft.fftshift(convoluted_frequency_image)))

    # Show the final convolved image.
    conv_result = np.real(np.fft.ifft2(convoluted_frequency_image))

    if verbose:
        fix, ax = plt.subplots(1, 4, figsize=(20, 4))

        ax[0].imshow(im, cmap='gray')
        ax[0].set_title('Original')

        ax[1].imshow(absolute_frequency_image, cmap='gray')
        ax[1].set_title('Frequency Domain')

        ax[2].imshow(absolute_convoluted_frequency_image, cmap='gray')
        ax[2].set_title('Frequency Domain Convolution')

        ax[3].imshow(conv_result, cmap='gray')
        ax[3].set_title('Convolved Image')

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)

    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
