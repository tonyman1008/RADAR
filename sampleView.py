import neural_renderer as nr
import os
from skimage.io import imsave
import torch
import math
from derender import utils, rendering
import numpy as np
import sys

print("====Start Sample View====")

## user input
objFolder = 'TestData_20220321_antiques_1'
objIndex = '3'
objName = objIndex + '.obj'
indexingOffset = -3

## same camera parameter with RADAR
image_size = 256
fov = 10
ori_z = 12.5
world_ori=[0,0,ori_z] ## make sure the camera pose is the same
sor_circum = 24

device = 'cuda:0'
current_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = current_dir + '/3SweepData/' + objFolder
save_dir = current_dir + '/3SweepData/' + objFolder +'/' + objIndex

renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size,fov=fov, fill_back=True)

vertices, faces, textures = nr.load_obj(
    os.path.join(input_dir, objName),normalization=False, load_texture=True, texture_size=8)

radcol_height = vertices.shape[0] // sor_circum
vertices, faces, textures = utils.parse3SweepObjData(radcol_height,sor_circum,vertices,faces,textures,indexOffset=indexingOffset)

# save parse object
utils.save_obj(save_dir,vertices,faces,suffix='obj_parsed',sep_folder=True)

## normalize
vertices = utils.normalizeObjVertices(vertices)

# # save parsed and normalize object
# utils.save_obj(save_dir,vertices,faces,suffix='obj_parsed_normalize',sep_folder=True)

# sample front view with straight axis object
sor_curve = rendering.get_straight_sor_curve(radcol_height,device)
canon_sor_vtx = rendering.get_sor_vtx(sor_curve, sor_circum) # BxHxTx3

images = renderer.render_rgb(canon_sor_vtx.reshape(1,-1,3), faces[None, :, :], textures[None, :, :, :, :, :])
utils.save_images(save_dir, images.detach().cpu().numpy(), suffix='sample_view', sep_folder=True)

print("====Sample view complete！====")