import os
import cv2
import numpy as np
import math

print(os.path.abspath(os.curdir))

CORNER_POINTS_FILE_PATH = "../dataset_prep/corner_points/corner_points.txt"

HMAP_MERGED_TXT_PATH    = "Map Data/1.txt"
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
        if grid[md][0][0] > N:
            st = md
        else:
            ed = md
    id = st
    eps = 5e-5
    while id < len(grid):
        if grid[id][0][0] > N + eps:
            id += 1
            continue
        elif grid[id][0][0] < N - eps:
            break
        else:
            st, ed = 0, len(grid[id])
            while st + 1 < ed:
                md = (st + ed) // 2
                if grid[id][md][1] < E:
                    st = md
                else:
                    ed = md
            # print(grid[id][st][1], E, abs(grid[id][st][1] - E))
            if abs(grid[id][st][1] - E) < eps:
                return grid[id][st][2]
            if st + 1 < len(grid[id]) and abs(grid[id][st + 1][1] - E) < eps:
                return grid[id][st + 1][2]
            id += 1
    return 0


def txt2hmap(coordinates, txt_path):
    with open(txt_path, "r") as file:
        cells = [[float(x) for x in line.split('\t')] for line in file.read().split('\n')]

    cells = sorted(cells, key = lambda x : -x[0])
    grid = []
    st, ed = 0, 0
    min_N, max_N, min_E, max_E = 0, 0, 0, 0
    Es = []
    while st < len(cells):
        ed = st
        while ed < len(cells) and cells[st][0] == cells[ed][0]:
            min_N = min(cells[ed][0], min_N)
            max_N = max(cells[ed][0], max_N)
            min_E = min(cells[ed][1], min_E)
            max_E = max(cells[ed][1], max_E)
            Es.append(cells[ed][1])
            ed += 1
        grid.append(sorted(cells[st : ed], key = lambda x : x[1]))
        st = ed
    Es = sorted(Es)
    print((Es[-1] - Es[0]))
    print(Es[0], Es[-1])
    exit(0)
    # mx_E = 0
    # for i in range(1, len(Es)):
    #     mx_E = max(mx_E, Es[i] - Es[i - 1])
    # print(mx_E)
    # exit(0)
    cur_N, end_N = coordinates[0], coordinates[2]
    earth_R = 6378000
    delta_N = 5 * 180 / earth_R / math.pi
    delta_E = 5 * 180 / (math.cos(cur_N / 180 * math.pi) * earth_R) / math.pi
    # print((max_N - min_N) / delta_N, (max_E - min_E) / delta_E)
    # exit(0)
    # print(delta_N, delta_E)
    height_map = []
    # mn = 10
    # for a in cells:
    #     mn = min(mn, abs(a[0] - coordinates[0]) + abs(a[1] - coordinates[1]))
    # print(mn)
    # exit(0)
    while cur_N < end_N:
        cur_E = coordinates[1]
        end_E = coordinates[3]
        height_row = []
        while cur_E < end_E:
            height_row.append(find_height(grid, cur_N, cur_E))
            if(height_row[-1] == 0):
                print(cur_N, cur_E)
                exit(0)
            cur_E += delta_E
        height_map.append(height_row.copy())
        cur_N += delta_N
    return height_map

    
        

def hmap2npy(hmap_path, npy_path):

    return

def parse_coordinates(coord_str):
    print('coord_str', coord_str)
    coords = coord_str.strip().split(',')
    print('coords', coords)
    return [float(coords[0]), float(coords[1])]

def get_corner_points(corner_pt_path):
    corner_pts = []
    try:
        with open(corner_pt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                pair_str = line.strip()
                print('pair_str', pair_str)
                pair = [parse_coordinates(pair_str[0]), parse_coordinates(pair_str[1])]
                corner_pts.append(pair)
    except FileNotFoundError:
        print(f"Error: File '{corner_pt_path}' not found.")

    print('corner_pts', corner_pts)
    return corner_pts

def split_lg_imgs(sat_img_path, 
                     dest_img_path, 
                     dest_label_path, 
                     dest_size=[256, 256],
                     step=32):
    [d_w, d_h] = dest_size

    sat_img_names = os.listdir(sat_img_path)
    for sat_img_name in sat_img_names:
        sub_dir_name = sat_img_name.split('.png')[0]

        sub_dir_split_img = os.path.join(dest_img_path, sub_dir_name)
        sub_dir_split_label= os.path.join(dest_label_path, sub_dir_name)

        if not os.path.exists(sub_dir_split_img):
            os.mkdir(sub_dir_split_img)
        if not os.path.exists(sub_dir_split_label):
            os.mkdir(sub_dir_split_label)

        sat_img = cv2.imread(os.path.join(sat_img_path, sat_img_name), cv2.IMREAD_UNCHANGED)
        (h, w) = sat_img.shape[: 2]

        h_len = h // step - d_w // step

        for i in range(h // step - d_h // step):
            direction = i % 2
            print(direction)
            for j in range(h // step - d_w // step):
                if direction == 0:
                    cropped_image = sat_img[i * step: i * step + d_h, j * step: j * step + d_w]
                    coords = [i * step + d_h // 2, j * step + d_h // 2]
                else:
                    cropped_image = sat_img[i * step: i * step + d_h, w - j * step - d_w: w - j  * step]
                    coords = [i * step + d_h // 2, w - j * step + d_h // 2]

                cv2.imwrite(os.path.join(sub_dir_split_img, "{:04d}_{:04d}.png".format(i, j)), cropped_image)
                with open(os.path.join(sub_dir_split_label, "{:04d}_{:04d}.txt".format(i, j)), 'w', encoding='utf-8') as f:
                    f.write(str(coords[0]) + ' ' + str(coords[1]))
                    f.close()

                # cv2.imshow('croped', cropped_image)
                # cv2.waitKey(100)
        



    return


def _main():

    # get_corner_points(CORNER_POINTS_FILE_PATH)
    # split_lg_imgs(SAT_IMG_PATH, SPLITED_IMAGE_PATH, SPLITED_LABEL_PATH, [256, 256], PIXEL_STEP)
    hmap = txt2hmap(COORDINATES[0], HMAP_MERGED_TXT_PATH)
    # print(hmap)
    
    show_result = hmap - np.min(hmap)
    show_result = show_result / np.max(show_result) * 255
    show_result = np.array(show_result, np.uint8) 
    cv2.imshow('2222showresult1', cv2.resize(show_result, [1000, 1000]))
    cv2.waitKey(0)

    print(np.array(hmap, dtype=float).shape)

    return

if __name__ == '__main__':
    _main()