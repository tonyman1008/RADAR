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
ori_z = 12.5
world_ori=[0,0,ori_z] ## make sure the camera pose is the same
sor_circum = 24
radcol_height = 0
inputFolder = ''
device = 'cuda:0'
inputFolder = 'TestData_20220321_spout_vase'
objName = '1.obj'


## TODO: check the camera pose
renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size,fov=fov, fill_back=True)

vertices, faces, textures = nr.load_obj(
    os.path.join(current_dir, '3SweepData/'+inputFolder+'/'+objName),normalization=False, load_texture=True, texture_size=8)

radcol_height = vertices.shape[0] // sor_circum
vertices, faces, textures = utils.parse3SweepObjData(radcol_height,sor_circum,vertices,faces,textures)

##TODO: save origin obj
os.makedirs(os.path.join(current_dir, '3SweepData/'+inputFolder+'/obj'),exist_ok=True)
nr.save_obj(os.path.join(current_dir, '3SweepData/'+inputFolder+'/obj/'+objName),vertices,faces)

## normalize
vertices = utils.normalizeObjVertices(vertices)

sor_curve = rendering.get_straight_sor_curve(radcol_height,device)
canon_sor_vtx = rendering.get_sor_vtx(sor_curve, sor_circum) # BxHxTx3

images = renderer.render_rgb(vertices.reshape(1,-1,3), faces[None, :, :], textures[None, :, :, :, :, :])
utils.save_images(os.path.join(current_dir, '3SweepData/'+inputFolder), images.detach().cpu().numpy(), suffix='TestSampleView', sep_folder=True)

os.makedirs(os.path.join(current_dir, '3SweepData/'+inputFolder+'/obj_normalize'),exist_ok=True)
nr.save_obj(os.path.join(current_dir, '3SweepData/'+inputFolder+'/obj_normalize/'+objName),vertices,faces)