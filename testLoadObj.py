import neural_renderer as nr
import os
from skimage.io import imsave
import torch
import math
from derender import utils, rendering
import numpy as np
import sys



current_dir = os.path.dirname(os.path.realpath(__file__))

## use custom renderer 
# ori_z = 5
# world_ori=[0,0,ori_z*2.5]
# image_size = 256
# fov = 10
# device = 'cuda:0'
# ## renderer for visualization
# R = [[[1.,0.,0.],
#         [0.,1.,0.],
#         [0.,0.,1.]]]
# R = torch.FloatTensor(R).to(device)
# t = torch.FloatTensor(world_ori).to(device)
# fx = image_size /2 /(math.tan(fov/2 *math.pi/180))
# fy = image_size /2 /(math.tan(fov/2 *math.pi/180))
# cx = image_size /2
# cy = image_size /2
# # print("fx",fx)
# # print("fy",fx)
# # print("cx",cx)
# # print("cy",cy)
# K = [[fx, 0., cx],
#         [0., -fy, cy],
#         [0., 0., 1.]]
# # print("K",K)
# K = torch.FloatTensor(K).to(device)
# inv_K = torch.inverse(K).unsqueeze(0)
# K = K.unsqueeze(0)
# renderer = nr.Renderer(camera_mode='projection',
#                         K=K, R=R, t=t,
#                         image_size=image_size, orig_size=image_size,
#                         light_intensity_ambient=1.0,
#                         light_intensity_directional=0.,
#                         background_color=[1.,1.,1.]
#                         )

## same camera parameter with RADAR
image_size = 256
fov = 10
ori_z = 5
world_ori=[0,0,2.5*ori_z] ## make sure the camera pose is the same
sor_circum = 24
radcol_height =0
inputFolder = ''
device = 'cuda:0'

## TODO: check the camera pose
renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size, fov=fov, fill_back=True)

vertices, faces, textures = nr.load_obj(
    os.path.join(current_dir, 'data/Test_Bending_20220314_3Sweep/vase.obj'),normalization=True, load_texture=True, texture_size=8)
radcol_height = vertices.shape[0]//sor_circum
print("vertices",vertices.shape)
print("faces",faces.shape)
print("textures",textures.shape)
# vertices, faces, textures = utils.parse3SweepObjData(radcol_height,sor_circum,vertices,faces,textures)
sor_curve = rendering.get_straight_sor_curve(radcol_height,device)
canon_sor_vtx = rendering.get_sor_vtx(sor_curve, sor_circum) # BxHxTx3

print("straight vertices",canon_sor_vtx.shape)
print("vertices",vertices[None, :, :].shape)
print("faces",faces[None, :, :].shape)
print("textures",textures[None, :, :, :, :, :].shape)
images = renderer.render_rgb(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :])
# images = renderer.render_rgb(canon_sor_vtx.reshape(1,-1,3), faces[None, :, :], textures[None, :, :, :, :, :])
utils.save_images(os.path.join(current_dir, 'data/Test_Bending_20220314_3Sweep'), images.detach().cpu().numpy(), suffix='TestSampleObj', sep_folder=True)

##TODO: save obj
os.makedirs(os.path.join(current_dir, 'data/Test_Bending_20220314_3Sweep/SaveObj'),exist_ok=True)
nr.save_obj(os.path.join(current_dir, 'data/Test_Bending_20220314_3Sweep/SaveObj/vase.obj'),vertices,faces)