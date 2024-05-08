import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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