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
radcol_height = 37
sor_circum = 24

## TODO: check the camera pose
renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size, fov=fov, fill_back=True)

vertices, faces = nr.load_obj(
    os.path.join(current_dir, 'results/TestResults_20220310_Obj_SampleView_37x24_ApplyOriginTransform/Obj/bend.obj'),normalization=True, load_texture=False, texture_size=8)
vertices_t, faces_t, textures_t = nr.load_obj(
    os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep/bend.obj'),normalization=True, load_texture=True, texture_size=8)
vertices_t, faces_t, textures_t = utils.parse3SweepObjData(radcol_height,sor_circum,vertices_t,faces_t,textures_t)

print("vertices",vertices[None, :, :].shape)
print("faces",faces[None, :, :].shape)
print("textures_t",textures_t[None, :, :, :, :, :].shape)
images = renderer.render_rgb(vertices[None, :, :], faces[None, :, :], textures_t[None, :, :, :, :, :])
utils.save_images(os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep'), images.detach().cpu().numpy(), suffix='TestSampleObj', sep_folder=True)

##TODO: save obj
os.makedirs(os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep/SaveObj'),exist_ok=True)
nr.save_obj(os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep/SaveObj/bend.obj'),vertices,faces)