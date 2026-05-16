from cv2.dnn import Image2BlobParams
import numpy as np
from matplotlib import image, pyplot as plt
from PIL import Image
from scipy import signal
import cv2 as cv

def gnoise(np_array, mean, sigma):
    # Ensure math happens in float to avoid overflow
    noise = np.random.normal(mean, sigma, np_array.shape)
    noisy = np_array.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def mean_3x3_5x5(np_array, filter_kernel):
    # convolution works best on floats
    res = signal.convolve2d(np_array, filter_kernel, boundary="fill", mode="same")
    return np.clip(res, 0, 255).astype(np.uint8)

def median_3x3_5x5(np_array, size=3):
    # medianBlur MUST be uint8
    return cv.medianBlur(np_array.astype('uint8'), size)

def sobel(np_array):
    # 1. Calculate gradients in FLOAT64 (Important!)
    grad_x = cv.Sobel(np_array, cv.CV_64F, 1, 0, ksize=-1)
    grad_y = cv.Sobel(np_array, cv.CV_64F, 0, 1, ksize=-1)
    
    # 2. Calculate magnitude while still in FLOAT
    mag = cv.magnitude(grad_x, grad_y)
    
    # 3. Convert back to uint8 for display
    return cv.convertScaleAbs(mag)

def laplacian(np_array):
    # Use CV_64F to catch negative slopes
    lap = cv.Laplacian(np_array, ddepth=cv.CV_64F, ksize=3)
    return cv.convertScaleAbs(lap)

if __name__ == "__main__":
    # --- FIGURE 1: "Eight" Noise & Filtering ---
    plt.figure(figsize=(12, 8))
    
    img_8 = Image.open("eight.png").convert("L")
    img_8_arr = np.array(img_8)
    
    plt.subplot(2, 2, 1); plt.imshow(img_8_arr, cmap='gray'); plt.title("Original"); plt.axis("off")
    
    noisy_8 = gnoise(img_8_arr, 50, 50) # Lowered sigma so it's visible
    plt.subplot(2, 2, 2); plt.imshow(noisy_8, cmap='gray'); plt.title("G-Noise"); plt.axis("off")

    mean_filter = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16
    mean_img = mean_3x3_5x5(noisy_8, mean_filter)
    plt.subplot(2, 2, 3); plt.imshow(mean_img, cmap='gray'); plt.title("Mean Filter"); plt.axis("off")

    med_img = median_3x3_5x5(noisy_8, 3)
    plt.subplot(2, 2, 4); plt.imshow(med_img, cmap='gray'); plt.title("Median Filter"); plt.axis("off")

    # --- FIGURE 2: "Circuit" Edge Detection ---
    # We call plt.figure() again to start a new window
    plt.figure(figsize=(12, 4))
    
    # Check if file exists to avoid crash
    circuit = cv.imread('circuit.jpg', cv.IMREAD_GRAYSCALE)
    plt.subplot(1, 3, 1)
    plt.imshow(circuit, cmap='gray');
    plt.title("Circuit"); plt.axis('off')



    plt.subplot(1, 3, 2)
    #image1 = cv.addWeighted(circuit, 1.0, sobel(circuit), 0.5, 0)
    sharpen_kernel_x = np.array([ [ -1, -2,  -1],
                                [  0, 0 ,  0],
                                [  1,  2, 1]])

    sharpen_kernel_y = np.array([ [ -1, 0,  1],
                                [  -2, 0 ,  2],
                                [  -1,  0, 1]])

    image1_x = cv.filter2D(circuit, -1, sharpen_kernel_x)
    image1_y = cv.filter2D(circuit, -1, sharpen_kernel_y)
    image1= cv.addWeighted(circuit, 1 , cv.add(image1_x,image1_y), 1, 1)
    plt.imshow(image1, cmap='gray')
    plt.title("Sobel"); plt.axis('off')
    

    plt.subplot(1, 3, 3)
    # image2 = cv.add(circuit, laplacian(circuit))
    sharpen_kernel = np.array([[ -1, -1,  -1],
                                   [-1,  9, -1],
                                   [ -1, -1,  -1]])
        
    image2 = cv.filter2D(circuit, -1, sharpen_kernel)
    plt.imshow(image2, cmap='gray')
    plt.title("True Laplacian Sharpen"); plt.axis('off')
    plt.imshow(image2, cmap='gray')
    plt.title("Laplacian"); plt.axis('off')

    plt.show()
