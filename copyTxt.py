import os
import shutil

if __name__ == "__main__":
    origin_path = "E:/data/origin_txt/"
    repeat_path = "E:/data/repeat_txt/"
    goal_path = "E:/data/diff/"
    goal_dir = os.listdir(goal_path)
    for file in goal_dir:
        if file == "repeat":
            list = os.listdir(goal_path + file)
            for f in list:
                if "train" not in f:
                    shutil.copy(repeat_path + f[0:12] + ".txt", goal_path + file + "/" + f[0:12] + ".txt")
        elif os.path.isdir(os.path.join(goal_path,file)):
            list = os.listdir(goal_path + file)
            for f in list:
                if "train" not in f:
                    shutil.copy(origin_path + f[0:12] + ".txt", goal_path + file + "/" + f[0:12] + ".txt")
