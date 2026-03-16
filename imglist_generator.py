import os
import ipdb
import torch
import ftfy
import html
import re
# from src.clip import clip as clip
# from openood.networks.clip import clip
# import openood.networks.clip as clip
# from PIL import Image
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

'''
path="./data/images_classic/cinic/valid"
save_path="./data/benchmark_imglist/cifar10/val_cinic10.txt"
prefix="cinic/valid/"
category=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
with open(save_path,'a') as f:
    for name in category:
        label=category.index(name)
        sub_path=path+'/'+name
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+name+'/'+file+' '+str(label)+'\n'
            f.write(line)
    f.close()       
'''
############################################################ prepare ood dataset for SUN, Place, and dtd (full textures)s
path="./data/images_largescale/dtd/images/"
save_path="./data/benchmark_imglist/imagenet/test_dtd.txt"
prefix="dtd/images/"
classes=os.listdir(path)
valid_extensions = {'.jpg', '.jpeg', '.png'}
with open(save_path,'w') as f:
    for cla in classes:
        used_prefix = prefix + cla 
        cla_path = path + cla
        files = os.listdir(cla_path)
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                line=used_prefix+ '/' +file+" -1\n"
                f.write(line)
            else:
                line=used_prefix+ '/' +file+" -1\n" 
                print(line)
    f.close()  


'''
path="./data/images_largescale/imagenet_v2"
save_path="./data/benchmark_imglist/imagenet/test_imagenetv2.txt"
prefix="imagenet_v2/"
with open(save_path,'a') as f:
    for i in range(0,1000):
        label=str(i)
        sub_path=path+'/'+label
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+label+'/'+file+' '+label+'\n'
            f.write(line)
    f.close() 
'''