import os

path = "E:/data/"
savepath = "E:/data/"
trainpath = "e:/"


def creat_path(listpath,files):
    filelist = os.listdir(listpath)

    diff_save=os.path.join(savepath, files + ".txt")

    f1 = open(diff_save, 'w');
    for f in filelist:
        Olddir = os.path.join(listpath+"/", f);
        if os.path.isdir(Olddir):
            continue;
        f1.write(Olddir)
        f1.write('\n')
    f1.close();


if __name__ == "__main__":
    f = os.listdir(path)
    for files in f:
        if os.path.isfile(os.path.join(path,files)):
            continue
        listpath = os.path.join(path, files)
        creat_path(listpath,files)
