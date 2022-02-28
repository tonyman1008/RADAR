import neural_renderer as nr
import os
from skimage.io import imsave
import torch
import math
from derender import utils, rendering
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

## same camera parameter with RADAR
image_size = 256
fov = 10
ori_z = 5
world_ori=[0,0,ori_z]
renderer_max_depth = 30
renderer_min_depth = 0.1

## sample image world_ori set to 2 times of ori_z, -5=>5 = 10
## TODO: check the camera pose
renderer = rendering.get_renderer(world_ori=[0, 0, 2.5*ori_z], image_size=image_size, fov=fov, fill_back=True)

vertices, faces, textures = nr.load_obj(
    os.path.join(current_dir, 'TestData_Obj_20220225/Test_Straight_3.obj'), load_texture=True, texture_size=16)
print("vertices",vertices.shape)
print("faces",faces.shape)
print("textures",textures.shape)

## rotation from 3-sweep ?
test_rxyz = torch.tensor([[0,np.pi/5,0]]).to(vertices.device)
rotmat = rendering.get_rotation_matrix(test_rxyz).to(vertices.device)
rotat_vertices = torch.stack([vertices],0)
print("rotat_vertices",rotat_vertices.shape)
rotat_vertices = rendering.rotate_pts(rotat_vertices, rotmat)  # BxNx3
vertices = rotat_vertices.view(-1,3)

# renderer.eye = nr.get_points_from_angles(-10, 0, 0)
# images, _, _ = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :])
images = renderer.render_rgb(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :])
print()
# images = images.permute(0,2,3,1).detach().cpu().numpy()
# imsave(os.path.join(current_dir, 'TestData_Obj_20220225/test.png'), images[0])
utils.save_images(os.path.join(current_dir, 'TestData_Obj_20220225'), images.detach().cpu().numpy(), suffix='TestSampleObj', sep_folder=True)