import cv2
import pandas as pd
import numpy as np


def read_and_crop(img_path, bb_path):
    '''
    args:
        img_path: string, /path/to/image/file
        bb_path: string, /path/to/bounding/box/file
    return:
        patches: a list of cropped pataches, each is a cv2 image in BGR format
        img: complete image
        corners: a list of numpy array, each of shape (4, 1, 2), represents a
                 bounding box
    '''
    img = cv2.imread(img_path)
    bb_df = pd.read_csv(bb_path, sep=' ', header=None, index_col=False)
    bb_mat = bb_df.to_numpy()
    patches = []
    corners = []
    for row in bb_mat:
        x_l = row[0]
        x_r = row[0] + row[2]
        y_u = row[1]
        y_d = row[1] + row[3]
        patches.append(img[y_u:y_d, x_l:x_r])
        corners.append(np.int32([[x_l, y_u], 
                                 [x_l, y_d],
                                 [x_r, y_d],
                                 [x_r, y_u]
                                ]
                               ).reshape(-1, 1, 2)
                      )
    return patches, img, corners

def get_img_with_bb(img, bb_corners):
    pass