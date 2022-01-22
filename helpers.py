import os
import numpy as np
import cv2 as cv


def load_images(folder_name):
    images = []
    for filename in sorted(os.listdir(folder_name)):
        img = cv.imread(os.path.join(folder_name, filename))
        if img is not None:
            images.append(img)
    return images


def reflection_detection(image, image_res):

    (grayscaled, masked, reflection, reflection_clean_smalls,
     cleaned_reflection_opening, opened_reflection_closing,
     res) = detect_reflection(image)

    show_steps(image, image_res, grayscaled, masked, reflection,
               reflection_clean_smalls, cleaned_reflection_opening,
               opened_reflection_closing, res)


def detect_reflection(image):
    grayscaled = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mask = np.zeros(grayscaled.shape[:2], dtype="uint8")
    cv.rectangle(mask, (650, 500), (1300, 1000), 255, -1)
    masked = cv.bitwise_and(grayscaled, grayscaled, mask=mask)

    _, outer_circle = cv.threshold(grayscaled, 15, 255, cv.THRESH_BINARY)
    _, reflection = cv.threshold(masked, 150, 255, cv.THRESH_BINARY_INV)

    kernel_small = np.ones((20, 20), np.uint8)
    kernel_big = np.ones((60, 60), np.uint8)

    opening_outer_circle = cv.morphologyEx(outer_circle, cv.MORPH_OPEN,
                                           kernel_small)

    reflection_clean_smalls = cv.morphologyEx(reflection, cv.MORPH_CLOSE,
                                              kernel_small)
    cleaned_reflection_opening = cv.morphologyEx(reflection_clean_smalls,
                                                 cv.MORPH_OPEN, kernel_big)
    opened_reflection_closing = cv.morphologyEx(cleaned_reflection_opening,
                                                cv.MORPH_CLOSE, kernel_big)

    xor = cv.bitwise_xor(opened_reflection_closing, opening_outer_circle)
    res = 255 - xor
    return (grayscaled, masked, reflection, reflection_clean_smalls,
            cleaned_reflection_opening, opened_reflection_closing, res)


def show_steps(image, image_res, grayscaled, masked, reflection,
               reflection_clean_smalls, cleaned_reflection_opening,
               opened_reflection_closing, res):
    cv.imshow('original', cv.resize(image, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("Mask Applied to Image", cv.resize(masked, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('gray', cv.resize(grayscaled, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('orifinal result', cv.resize(image_res, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('reflection', cv.resize(reflection, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('reflection_clean_smalls',
              cv.resize(reflection_clean_smalls, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('cleaned_reflection_opening',
              cv.resize(cleaned_reflection_opening, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('opened_reflection_closing',
              cv.resize(opened_reflection_closing, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('calculated result', cv.resize(res, (700, 700)))
    cv.waitKey(0)
    cv.destroyAllWindows()
