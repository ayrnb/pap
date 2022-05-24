# -*- coding:utf-8 -*-

import json
import csv

json_file = r"F:/data/annotations_trainval2017/annotations/instances_train2017.json"
data = json.load(open(json_file, 'r'))

data_2 = {
    'info': data['info'],
    'categories': data['categories'],
    'licenses': data['licenses'],
    'images': [data['images'][0]]
}

annotation = []
d={}
# imgID = data_2['images'][0]['id']
for ann in data['annotations']:
    num = ann['image_id']
    if num not in d:
        d2 = {num: 1}
        d.update(d2)
    else:
        d[num]+=1

keyList = d.keys()
valueList = d.values()
rows = zip(keyList, valueList)
with open('test.csv', 'a', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

# data_2['annotations'] = annotation

# json.dump(data_2, open(r'path\to\single_person_kp.json', 'w'), indent=4)
