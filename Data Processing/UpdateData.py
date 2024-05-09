import os

def dot_to_degree(dot):
    degree = float(dot.split('°')[0]) + float(dot.split('°')[1].split("'")[0]) / 60 + float(dot.split('°')[1].split("'")[1].split('"')[0]) / 60 / 60
    return round(degree, 9)

def degree_to_dot(degree):
    d = float(degree)
    x = int(d)
    d = (d - x) * 60
    y = int(d)
    d = (d - y) * 60
    z = round(d, 2)
    return str(x) + "°" + str(y) + "'" + str(z) + '"'



current_directory = os.getcwd()

files = os.listdir(current_directory)

change_list = []

for file in files:
    if file.endswith('.txt'):
        with open(file, 'r') as f:
            min_N, max_N, min_E, max_E = 360, 0, 360, 0
            for line in f.read().split('\n'):
                x, y = float(line.split('\t')[0]), float(line.split('\t')[1])
                if min_N > x:
                    min_N = x
                if max_N < x:
                    max_N = x
                if min_E > y:
                    min_E = y
                if max_E < y:
                    max_E = y
            min_N_dot = degree_to_dot(str(min_N))
            max_N_dot = degree_to_dot(str(max_N))
            min_E_dot = degree_to_dot(str(min_E))
            max_E_dot = degree_to_dot(str(max_E))
            excepted_name = max_N_dot + 'N ' + min_E_dot + 'E' + min_N_dot + 'N ' + max_E_dot + 'N.txt'
            if file != excepted_name:
                print(excepted_name)
                change_list.append((file, excepted_name))

for old_name, new_name in change_list:
    os.rename(old_name, new_name)
    old_path = os.path.join(current_directory, old_name)
    new_path = os.path.join(current_directory, new_name)
    print(f"Renamed '{old_name}' to '{new_name}'")
