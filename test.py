import os

d1={"k":1,"o":2,"w":3}
d2={"k":{"1":0,"2":3},"o":2}

if __name__=="__main__":
    print(d1.keys())
    print(d2.keys())
    print(d1.keys()>d2.keys())
    assert d1.keys()>d2.keys()