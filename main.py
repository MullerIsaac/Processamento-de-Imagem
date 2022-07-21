import numpy as np
import cv2
from convolution import convolution
from filtro_gaussiano import gaussian_blur


def sobel_edge_detection(image, filter):
    sobel_x = convolution(image, filter)
    cv2.imwrite("calopsita_sobelx.jpg", sobel_x)
    
    sobel_y = convolution(image, np.flip(filter.T, axis=0))
    cv2.imwrite("calopsita_sobely.jpg", sobel_y)


    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    return gradient_magnitude


if __name__ == '__main__':
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    image = cv2.imread("calopsita.jpg")
    image = gaussian_blur(image, 9)
    cv2.imwrite("calopsitaBlur.jpg", image)
    gradiente = sobel_edge_detection(image, filter)
    cv2.imwrite("gradient.jpg", gradiente)

    thr = float(50)
    matrix_d = np.zeros(image.shape)
    print(gradiente)
    gd_row, gd_col = gradiente.shape

    for row in range(gd_row):
        for col in range(gd_col):
            if abs(gradiente[row, col]) >= thr : 
                matrix_d[row, col] = 0
            else:
                matrix_d[row, col] = 1


    cv2.imwrite('matrizD.jpg', matrix_d)