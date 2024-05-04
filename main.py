from PIL import Image

from get_mesh import *


def convert_data_into_grayimage(total_data, height, width):

    data = []
    for x in total_data.split("\n"):
        if len(x) < 1:
            continue
        temp_array = []
        for y in x.split('	'):
            temp_array.append(float(y))
        data.append(temp_array.copy())

    data = sorted(data, key = lambda x : -x[0])

    final_data = []
    for i in range(height):
        temp_array = sorted(data[i * width : (i + 1) * width], key = lambda x : x[1])
        final_data.append([y[2] for y in temp_array])

    return final_data

def value_to_color(value, min_height, max_height):
    return (max_height - value) / (max_height - min_height)

def convert_3d_cluster_into_height_map():

    pass

def main():
    f = open('data.txt')
    total_data = f.read()
    height = 1000
    width = 1600

    map_data = convert_data_into_grayimage(total_data, height, width)
    gray_image = map_data.copy()

    min_height = np.min(map_data)
    max_height = np.max(map_data)

    for i in range(height):
        for j in range(width):
            gray_image[i][j] = value_to_color(map_data[i][j], min_height, max_height)

    data = np.array(gray_image)
    scaled_data = (255 * data).astype(np.uint8)
    img = Image.fromarray(scaled_data)

    img.show()
    img.save("gray_image.jpg")

    test_mesh = []
    for i in range(height):
        if i % 5 == 0:
            for j in range(width):
                if j % 5 == 0:
                    test_mesh.append([i * 10, j * 10, gray_image[i][j] * 2000])
    get_mesh1(test_mesh)

    


main()