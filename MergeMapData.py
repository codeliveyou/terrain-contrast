import os
import cv2
import numpy as np
import math

print(os.path.abspath(os.curdir))

CORNER_POINTS_FILE_PATH = "../dataset_prep/corner_points/corner_points.txt"

HMAP_MERGED_TXT_PATH    = "../dataset_prep/hmap_merged_txt/1.txt"
HAMP_NPY_PATH           = "../dataset_prep/hmap_npy/"

SAT_IMG_PATH            = "../dataset_prep/sat_imgs/"

SPLITED_IMAGE_PATH      = "../dataset_prep/sat_image_splited/"
SPLITED_LABEL_PATH      = "../dataset_prep/sat_label_splited/"

PIXEL_STEP = 32
H_SHAPE = [1700, 1700]

COORDINATES = [
    [39.1862, 125.62931388888889, 39.267200081000084, 125.73381708395551],
]

def find_height(grid, N, E):
    st, ed = 0, len(grid)
    while st + 1 < ed:
        md = (st + ed) // 2
        if grid[md][0][0] < N:
            st = md
        else:
            ed = md
    id = md
    eps = 5e-5
    while id < len(grid):
        if grid[id][0][0] < N - eps:
            id += 1
            continue
        elif grid[id][0][0] > N + eps:
            break
        else:
            st, ed = 0, len(grid[id])
            while st + 1 < ed:
                md = (st + ed) // 2
                if grid[id][md][1] < E:
                    st = md
                else:
                    ed = md
            if abs(grid[id][md][1] - E) < eps:
                return grid[id][md][2]
            if md + 1 < len(grid[id]) and abs(grid[id][md + 1][1] - E) < eps:
                return grid[id][md + 1][2]
            id += 1
    print("Error 3")
    return 0


def txt2hmap(coordinates, txt_path):
    with open(txt_path, "r") as file:
        cells = [[float(x) for x in line.split(' ')] for line in file.readlines()]
        file.close()
    cells = sorted(cells, key = lambda x : -x[0])
    grid = []
    st, ed = 0, 0
    while st < len(cells):
        ed = st
        while ed < len(cells):
            ed += 1
        grid.append(sorted(cells[st : ed], key = lambda x : x[1]))
        st = ed
    cur_N, end_N = coordinates[0], coordinates[2]
    earth_R = 6378000
    delta_N = 5 * 180 / earth_R / math.pi
    delta_E = 5 * 180 / (math.cos(cur_N / 180 * math.pi) * earth_R) / math.pi
    height_map = []
    while cur_N < end_N:
        cur_E = coordinates[1]
        end_E = coordinates[3]
        height_row = []
        while cur_E < end_E:
            height_row.append(find_height(grid, cur_N, cur_E))
            cur_E += delta_E
        height_map.append(height_row.copy())
        cur_N += delta_N
    return height_map