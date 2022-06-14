import os
from glob import glob
import shutil

def copyImageAndMaskData(inPath,outPath):

    os.makedirs(outPath, exist_ok=True)

    images_out_dir = os.path.join(outPath,'images')
    masks_out_dir = os.path.join(outPath,'masks')

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)
    
    image_path_list = glob(os.path.join(inPath, '**/*im_rendered.png'), recursive=True)
    for image_path in image_path_list:
        shutil.copy(image_path,images_out_dir)

    mask_path_list = glob(os.path.join(inPath, '**/*mask_rendered.png'), recursive=True)
    for mask_path in mask_path_list:
        shutil.copy(mask_path,masks_out_dir)

if __name__ == '__main__':
    in_dir = '../syn_curv_sgl5_tex_straight_20220220/rendering'
    out_dir = '../syn_curv_sgl5_tex_straight_20220220/image_mask'

    for split in ['train','test','val']:
        copyImageAndMaskData(os.path.join(in_dir,split),os.path.join(out_dir,split))
