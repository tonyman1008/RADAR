from glob import glob
import os
import shutil

root = './results/'
target ="TestResults_20220613_full"
for filename in os.listdir(root):
    if filename.startswith(target):
        src = os.path.join(root,filename)
        print("src",src)
        src_folder = os.path.join(src,'animations_2')
        dst_folder = os.path.join(src,'animations')
        print("src_folder",src_folder)
        print("dst_folder",dst_folder)
        if os.path.isdir(src_folder) :
        # newFileName = filename.replace('20220607','20220613')
        # dst = os.path.join(root,newFileName)
            shutil.copytree(src_folder, dst_folder)