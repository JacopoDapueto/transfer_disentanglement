import os
import math
import numpy as np
import cv2
import itertools
import pandas as pd
import shutil


COIL100_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", "../src/data/datasets"), "coil-100",
    "coil-100")

COIL100BINARY_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", "../src/data/datasets"), "coil-100",
    "coil-100-binary")


COIL100AUGMENTED_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", "../src/data/datasets"), "coil-100",
    "coil-100-augmented")


COIL100AUGMENTEDBINARY_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", "../src/data/datasets"), "coil-100",
    "coil-100-augmented-binary")


def degree_to_radiants(degree):
    return degree * math.pi / 180


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderValue=[0, 0, 0])
    return result


def rotate_image_no_crop(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)
    return rotated_mat


def zoom_at(img, zoom=1, angle=0, coord=None):
    if len(img.shape) > 2:
        shape = img.shape[:-1]
    else:
        shape = img.shape
    cy, cx = [i / 2 for i in shape] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST, borderValue=[0, 0, 0])

    return result


def binarize_img(img_name):
    img = cv2.imread(os.path.join(COIL100_PATH, img_name), cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_to_bin = cv2.medianBlur(gray_img, 5)
    ret, th3 = cv2.threshold(img_to_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)

    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

    closing = cv2.resize(closing, (64, 64), interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4
    ret, closing = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return closing


def binarize_dataset(path = COIL100BINARY_PATH):

    img_names = [image_name for image_name in os.listdir(COIL100_PATH) if image_name[-3:] == "png"]

    for name in img_names:
        binary = binarize_img(name)

        cv2.imwrite(os.path.join(path, name), binary)


def create_csv(path = COIL100BINARY_PATH, save_classes = True, save_factors = True):
    # factors list
    factors = []
    obj_range = range(1, 101)
    pose_range = range(0, 360, 5)

    # classes list
    classes = []
    obj_classes = range(0, 100)
    pose_classes = range(0, 72)


    for factor, clas in zip(itertools.product(obj_range, pose_range), itertools.product(obj_classes, pose_classes)):
        obj, pose = factor
        obj_class, pose_class = clas

        augmented_name = "obj{}__{}.png".format(obj, pose)
        classes.append([augmented_name, obj_class, pose_class])

    pd.DataFrame(factors, columns=["image", "object", "pose"]).to_csv(os.path.join(COIL100_PATH, "factors.csv"),
                                                                      index=False)
    pd.DataFrame(classes, columns=["image", "object", "pose"]).to_csv(os.path.join(COIL100_PATH, "classes.csv"),
                                                                      index=False)

    # copy classes
    if save_classes:
        shutil.copyfile(os.path.join(COIL100_PATH, "classes.csv"), os.path.join(path, "classes.csv"))

    # copy factors
    if save_factors:
        shutil.copyfile(os.path.join(COIL100_PATH, "factors.csv"), os.path.join(path, "factors.csv"))


def create_augmented_csv(augmented_path = COIL100AUGMENTEDBINARY_PATH):

    # factors list
    factors = []
    obj_range = range(1, 101)
    pose_range = range(0, 360, 5)
    rotation_range = range(0, 360, 20)
    scale_range = np.linspace(1, 0.5, num=9).round(2)

    # classes list
    classes = []
    obj_classes = range(0, 100)
    pose_classes = range(0, 72)
    rotation_classes = range(0, 18)
    scale_classes = range(0, 9)

    for factor, clas in zip(itertools.product(obj_range, pose_range, rotation_range, scale_range),
                            itertools.product(obj_classes, pose_classes, rotation_classes, scale_classes)):
        obj, pose, angle, scale = factor
        obj_class, pose_class, angle_class, scale_class = clas

        augmented_name = "obj{}__{}__{}__{}.png".format(obj, pose, angle, scale)
        classes.append([augmented_name, obj_class, pose_class, angle_class, scale_class])

    pd.DataFrame(factors, columns=["image", "object", "pose", "rotation", "scale"]).to_csv(
        os.path.join(augmented_path, "factors.csv"), index=False)
    pd.DataFrame(classes, columns=["image", "object", "pose", "rotation", "scale"]).to_csv(
        os.path.join(augmented_path, "classes.csv"), index=False)


def augment_data(original_path = COIL100BINARY_PATH, augmented_path = COIL100AUGMENTEDBINARY_PATH):
    factors = []
    obj_range = range(1, 101)
    pose_range = range(0, 360, 5)
    rotation_range = range(0, 360, 20)
    scale_range = np.linspace(1, 0.5, num=9).round(2)

    for obj, pose, angle, scale in itertools.product(obj_range, pose_range, rotation_range, scale_range):
        img_name = "obj{}__{}.png".format(obj, pose)
        img_path = os.path.join(original_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # add rotation and scale
        img_augmented = rotate_image(img, int(angle))
        img_augmented = zoom_at(img_augmented, zoom=scale)

        augmented_name = "obj{}__{}__{}__{}.png".format(obj, pose, angle, scale)
        cv2.imwrite(os.path.join(augmented_path, augmented_name), img_augmented)
        factors.append([augmented_name, obj, pose, angle, scale])



if __name__ == "__main__":

    # assuming original COIL100 is in folder COIL100_PATH

    # create folders
    if not os.path.exists(COIL100AUGMENTED_PATH):
        os.makedirs(COIL100AUGMENTED_PATH)

    if not os.path.exists(COIL100AUGMENTEDBINARY_PATH):
        os.makedirs(COIL100AUGMENTEDBINARY_PATH)


    # augment RGB dataset
    create_csv(COIL100_PATH, save_classes=False, save_factors=False)
    create_csv(COIL100AUGMENTED_PATH)
    augment_data(COIL100_PATH, COIL100AUGMENTED_PATH)


    # binarize original
    binarize_dataset()

    # augment binary dataset
    create_csv(COIL100BINARY_PATH)
    create_csv(COIL100AUGMENTEDBINARY_PATH)
    augment_data(COIL100BINARY_PATH, COIL100AUGMENTEDBINARY_PATH)







