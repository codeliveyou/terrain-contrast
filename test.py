import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time

start_time = time.time()

arr = [[random.randint(0, 100000), random.randint(0, 100000)] for _ in range(100000)]

def distance(x, y, target_x, target_y):
    return abs(target_x - x) + abs(target_y - y)

target_x = random.randint(0, 100000)
target_y = random.randint(0, 100000)

def find_nearest_position(arr, target_x, target_y):
    min_distance = -1
    ans_x, ans_y = 0, 0
    for cell in arr:
        cur_distance = distance(cell[0], cell[1], target_x, target_y)
        if min_distance == -1 or cur_distance < min_distance:
            min_distance = cur_distance
            ans_x, ans_y = cell
    return [ans_x, ans_y, min_distance]

print(find_nearest_position(arr, target_x, target_y))
print(time.time() - start_time)

exit(0)


def draw_path(
    a,
    b,
    height=1920,
    width=1080,
    img_path="./image.png",
    min_y=39.1025444444444,
    max_y=39.1479055555556,
    min_x=125.818866666667,
    max_x=125.894291666667,
):
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    img = mpimg.imread(img_path)
    plt.imshow(img, extent=[min_x, max_x, min_y, max_y])

    def plot(dots, color):
        for index, dot in enumerate(dots):
            plt.plot(dot[1], dot[0], color)
            if index % 20 == 0:
                plt.text(dot[1], dot[0], str(index), color="yellow")

    plot(a, "bo")
    plot(b, "ro")
    plt.show()


a_file_path = "./output.txt"
a = []
with open(a_file_path) as a_file:
    lines = a_file.readlines()
for line in lines:
    a.append([float(item) for item in line.split(" ")[0:2]])
# print(a)
draw_path(a=a, b=[])