import os


#  Enter Image Path:  Detection layer: 139 - type = 28
#  Detection layer: 150 - type = 28
#  Detection layer: 161 - type = 28
def count_different(path):
    file = open(path, "r")
    textlist = file.readlines()
    file_result = []
    object_result = []
    filename = ""
    for value in textlist:
        a = value[-15:-1]
        if path == "E:/data/result.docx" and value[-15:-1] == "milli-seconds.":
            list = value.split("/")
            filename = list[2][0:12]
            continue
        elif value[-15:-1] == "milli-seconds.":
            list = value.split("/")
            filename = list[3][0:12]
            continue
        if str(value) == "Enter Image Path:  Detection layer: 139 - type = 28 \n":
            continue

        if str(value) == " Detection layer: 161 - type = 28 \n":
            pit_dict = {filename: object_result}
            file_result.append(pit_dict)
            object_result = []
            continue
        if str(value) == " Detection layer: 150 - type = 28 \n":
            continue
        object_ = value.split(":")[0]
        object_result.append(object_)
    return file_result


if __name__ == "__main__":
    path = "E:/data/"
    original_path = "E:/data/result.docx"
    file = os.listdir(path)
    original_file = count_different(original_path)
    mr_file = []
    # for f in file:
    filepath = os.path.join(path, "result_repeat.txt")

    # `   filepath = os.path.join(path, f)
    # if os.path.isdir(filepath):
    #     continue
    mr_file = count_different(filepath)
    for i in range(len(mr_file)):
        for k, v in mr_file[i].items():
            v_o = list(original_file[i].values())[0]
            # repeat
            if len(v_o) == 1 and len(v) == 2 and v[0] == v[1]:
                continue
            if len(v) == 0 and len(v_o) == 0:
                continue
            # 单目标
            # if len(v_o) == len(v) and v[0] == v_o[0]:
            #     continue
            # diff_path = "E:/data/diff/" + f.split(".")[0]

            diff_path="E:/data/diff/repeat"
            if not os.path.exists(diff_path):
                os.makedirs(diff_path)
            f_diff = open(diff_path + "/difflist.txt", "a")
            f_diff.write(k + "\n")
