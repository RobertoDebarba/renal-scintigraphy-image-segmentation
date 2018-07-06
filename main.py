from functools import reduce

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_SRC = 'renal.png'


def main(img_src):
    """
    Processa a imagem informada.

    Sumário:

    * b = valor de menor intensidade. Representa a base do rim.
    * m = valor de maior intensidade. Representa o topo do rim.
    * med = valor da mediana entre b e m. Representa o valor mais próximo da borda do rim.
    """

    # Valor de maior intensidade
    m = 159

    src_gray = cv2.cvtColor(cv2.imread(img_src), cv2.COLOR_BGR2GRAY)
    cv2.imshow("A Original", src_gray)

    # Pré-processamento
    src_processed = pre_process(src_gray)

    # Encontra os dois maiores picos da imagem, que definem os dois rins
    b_contours = max_contours(src_processed, m)
    draw_contours_and_show(src_gray, b_contours, "F Maximos")

    # noinspection PyTypeChecker
    center_point_1 = center_point(b_contours[0])
    # noinspection PyTypeChecker
    center_point_2 = center_point(b_contours[1])
    division = (center_point_1[0] + center_point_2[0]) / 2

    # Divide a imagem, resultando em uma para cada rim
    # A partir desse ponto cada rim é processado separadamente
    src_processed_1 = src_processed[:, :int(division)]
    src_processed_2 = src_processed[:, int(division):]
    src_gray_1 = src_gray[:, :int(division)]
    src_gray_2 = src_gray[:, int(division):]

    # Busca a mediana entre o ponto de maior intensidade e o ponto de menor intesidade
    med1, b1 = med(src_processed_1, m)
    med2, b2 = med(src_processed_2, m)

    # Busca as bordas do rim pelo valor de b
    # Desenha os contornos e junta as imagens
    src_color_1 = cv2.cvtColor(src_gray_1, cv2.COLOR_GRAY2RGB)
    src_color_2 = cv2.cvtColor(src_gray_2, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(src_color_1, [(bigger_contour(src_processed_1, b1))], 0, (0, 0, 255))
    cv2.drawContours(src_color_2, [(bigger_contour(src_processed_2, b2))], 0, (0, 0, 255))
    full_img = np.concatenate((src_color_1, src_color_2), 1)
    cv2.imshow("G Thresh B", full_img)

    # Busca as bordas do rim pelo valor da mediana
    # Desenha os contornos e junta as imagens
    src_color_1 = cv2.cvtColor(src_gray_1, cv2.COLOR_GRAY2RGB)
    src_color_2 = cv2.cvtColor(src_gray_2, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(src_color_1, [(bigger_contour(src_processed_1, med1))], 0, (0, 0, 255))
    cv2.drawContours(src_color_2, [(bigger_contour(src_processed_2, med2))], 0, (0, 0, 255))
    full_img = np.concatenate((src_color_1, src_color_2), 1)
    cv2.imshow("H Med", full_img)

    # TODO Etapa final
    # mask = np.zeros_like(filtrada1)
    # cv2.drawContours(mask, contours, 0, 255, -1)
    # out = np.zeros_like(filtrada1)
    # out[mask == 255] = filtrada1[mask == 255]
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gradient = cv2.morphologyEx(out, cv2.MORPH_GRADIENT, kernel)
    #
    # cv2.imshow("Gradient", gradient)

    plt.show()

    key = cv2.waitKey(0)
    while key != 32:
        key = cv2.waitKey(0)


def pre_process(src_gray):
    """ Executa as etapas de pre-processamento:

            * Dilatação
            * Blur
            * Fechamento
            * Blur
    """

    dilated = cv2.dilate(src_gray, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cv2.imshow("B Dilated", dilated)
    blurred = cv2.GaussianBlur(dilated, (7, 7), 0)
    cv2.imshow("C Blurred", blurred)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cv2.imshow("D Closed", closed)
    blurred2 = cv2.GaussianBlur(closed, (7, 7), 0)
    cv2.imshow("E Blurred", blurred2)
    return blurred2


def draw_contours_and_show(img_gray, contours, text, thickness=-1):
    """ Desenha os contornos e exibe a imagem """

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cv2.drawContours(img_color, contours, i, (0, 0, 255), thickness)
    cv2.imshow(text, img_color)


def med(img, m):
    """ Busca a mediana (med) e o valor de menor intesidade (b) entre m e b """

    contours_area = []
    contours_b = []
    for b_value in range(m, 0, -1):
        area = cv2.contourArea(bigger_contour(img, b_value))
        contours_area.append(area)
        contours_b.append(b_value)

    # Plota a curva de crescimento das áreas em função dos níveis de cinza
    plt.plot(contours_b, contours_area)
    plt.ylabel('Área')
    plt.xlabel('Níveis de cinza')
    plt.draw()

    b_idx = first_peak(contours_area)
    b_value = contours_b[b_idx]

    mb = contours_b[b_value:b_idx]  # TODO rever esse b_value
    return mb[int(len(mb) / 2)], b_value


def first_peak(contours_area):
    """ Encontra o indice do primeiro pico de variação da area """

    for i in range(len(contours_area)):
        if i == 0:
            continue

        current_variation = contours_area[i] - contours_area[i - 1]
        if current_variation > average_variation(contours_area[:i + 1]) * 6:
            return i


def bigger_contour(img, thresh):
    """ Obtem os maior contorno obtido com o thresh informado """

    return max_contours(img, thresh)[0]


def max_contours(img, thresh):
    """ Obtem os dois maiores contornos obtidos com o thresh informado """

    _, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=len, reverse=True)[:2]


def center_point(contour):
    """ Calcula o ponto central do contorno """

    max_x = 0
    max_y = 0
    min_x = 999999999999999999
    min_y = 999999999999999999
    for k in contour:
        x = k[0][0]
        y = k[0][1]

        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    med_x = (max_x + min_x) / 2
    med_y = (max_y + min_y) / 2
    return [med_x, med_y]


def average_variation(values):
    """ Calcula a média de variação dos valores"""

    variation = []
    for i in range(len(values)):
        if i == 0:
            continue

        variation.append(values[i] - values[i - 1])
    return reduce(lambda x, y: x + y, variation, 1) / len(variation)


if __name__ == '__main__':
    main(IMG_SRC)
