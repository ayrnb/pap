import os;
import random


def listname(path, trainpath):
    filelist = os.listdir(path);
    for f in filelist:
        f1=os.path.join(path, f)
        if os.path.isfile(f1):
            continue
        filelists = os.listdir(f1)
        train_file = f1 + "/train.txt"

        train = open(train_file, "w")
        for files in filelists:
            Olddir = os.path.join(f1+"/", files);
            if os.path.isdir(Olddir):
                continue;
            if "txt" not in str(files):
                # 2表示一个比例，可以修改
                train.write(Olddir);
                train.write('\n');
        train.close();


# f2.close();


savepath = "E:/data/diff/"  # 修改为自己的
trainpath = savepath + "/train.txt"
# testpath = savepath + "/test.txt"
listname(savepath, trainpath)
print("Txt have been created!")
