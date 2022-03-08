import neural_renderer as nr
import os
from skimage.io import imsave
import torch
import math
from derender import utils, rendering
import numpy as np
import sys



current_dir = os.path.dirname(os.path.realpath(__file__))

## same camera parameter with RADAR
image_size = 256
fov = 10
ori_z = 5
world_ori=[0,0,2.5*ori_z] ## make sure the camera pose is the same
renderer_max_depth = 30
renderer_min_depth = 0.1

## sample image world_ori set to 2 times of ori_z, -5=>5 = 10
## TODO: check the camera pose
renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size, fov=fov, fill_back=True)

vertices, faces, textures = nr.load_obj(
    os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep/bend.obj'),normalization=False, load_texture=True, texture_size=16)
vertices,faces,textures = utils.parse3SweepObjData(vertices,faces,textures)

images = renderer.render_rgb(vertices.reshape(1,-1,3), faces[None, :, :], textures[None, :, :, :, :, :])
utils.save_images(os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep'), images.detach().cpu().numpy(), suffix='TestSampleObj', sep_folder=True)