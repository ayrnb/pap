import os;
import random


def listname(path, trainpath):
    filelist = os.listdir(path);
    filelist.sort()
    f1 = open(trainpath, 'w');
   # f2 = open(testpath, 'w');
    for files in filelist:
        Olddir = os.path.join(path, files);
        if os.path.isdir(Olddir):
            continue;
        if "xml" not in str(files):
             # 2表示一个比例，可以修改
            f1.write("data/obj/" + files);
            f1.write('\n');

    f1.close();
   # f2.close();


savepath = "D:/darknet-master/darknet-master/build/darknet/x64/data"  # 修改为自己的
trainpath = savepath + "/train.txt"
#testpath = savepath + "/test.txt"
listname(savepath + "/obj", trainpath)
print("Txt have been created!")
