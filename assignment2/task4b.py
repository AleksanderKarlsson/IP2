import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils




def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution.
        
        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    
    # Create empty matrix which will become the padded matrix.
    padded_kernel = np.zeros(shape=im.shape)
    
    # Insert kernel into the zero-padded kernel.
    padded_kernel[
        im.shape[0] // 2 - kernel.shape[0] // 2 : im.shape[0] // 2 - kernel.shape[0] // 2 + kernel.shape[0],
        im.shape[1] // 2 - kernel.shape[1] // 2 : im.shape[1] // 2 - kernel.shape[1] // 2 + kernel.shape[1],
    ] = kernel

    # Shift the kernel. After shifting, the middle of the kernel will be at the origin in
    # the frequency domain.
    # TODO: Shift kernel, or not? Seems to be related to circular / linear convolutions, and spill-over from the
    # linear convolution being too long. Images with either type of padding look similar.
    padded_kernel = np.fft.ifftshift(padded_kernel)

    frequency_image = np.fft.fft2(im)
    absolute_frequency_image = np.log(1 + np.abs(np.fft.fftshift(frequency_image)))
    
    convoluted_frequency_image = np.fft.fft2(im) * np.fft.fft2(padded_kernel)
    absolute_convoluted_frequency_image = np.log(1 + np.abs(np.fft.fftshift(convoluted_frequency_image)))

    conv_result = np.real(np.fft.ifft2(np.fft.fft2(im) * np.fft.fft2(padded_kernel)))

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
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
