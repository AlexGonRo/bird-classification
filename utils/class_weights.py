import os


def class_weights(path_data):
    class_weight = {}
    class_count = []
    total_count = 0
    classes = os.listdir(path_data)
    for fol in classes:
        imgfiles = os.listdir(path_data + '/' + fol);
        class_count.append(len(imgfiles))
        total_count += len(imgfiles)
    for i, c in enumerate(class_count):
        max_count = max(class_count)
        class_weight[i] = float(max_count) / c

    return class_weight