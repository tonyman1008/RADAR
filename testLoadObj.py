import neural_renderer as nr
import os
from skimage.io import imsave
import torch
import math
from derender import utils, rendering
import numpy as np
import sys



current_dir = os.path.dirname(os.path.realpath(__file__))

def parse3SweepObjData(vertices,faces,textures):
    ## because the 3-sweep index method need to parse the data to fit RADAR indexing method
    
    ## Parameter
    rowIndexOffset = 5
    rowVertices = 24
    verticesNum = vertices.shape[0]
    rad_col = verticesNum // rowVertices
    sor_circum = rowVertices
    print("rad_col",rad_col)

    ## Face
    lastRowMap = torch.arange(47,25,-1).to(faces.device)
    firstTwoElement = torch.arange(24,26,1).to(faces.device)
    lastRowMap = torch.cat([firstTwoElement,lastRowMap],0)
    offsetLastRowMap = torch.arange((verticesNum),(verticesNum+rowVertices),1).to(faces.device)
    faces = faces[44:]
    for i, (originValue,mapValue) in enumerate(zip(lastRowMap,offsetLastRowMap)):
        faces[faces==originValue] = mapValue.int()
    index = faces>(rowVertices-1)
    faces[index] -= rowVertices

    ## Texture (delete first 44 value in textures upper+lower circle) 
    textures = textures[44:]

    ## Vertices (replace new vertices instead)
    sor_curve = rendering.get_random_straight_sor_curve(rad_col).to(vertices.device)
    sor_vtx = rendering.get_sor_vtx(sor_curve,sor_circum).to(vertices.device)
    print("sor_curve",sor_curve.shape)
    print("sor_vtx",sor_vtx.shape)
    sor_vtx = torch.roll(sor_vtx,rowIndexOffset,2)

    return faces,sor_vtx,textures

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
    os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep/bend.obj'),normalization=False, load_texture=True, texture_size=16)
faces,vertices,textures = parse3SweepObjData(vertices,faces,textures)

images = renderer.render_rgb(vertices.reshape(1,-1,3), faces[None, :, :], textures[None, :, :, :, :, :])
utils.save_images(os.path.join(current_dir, 'data/Test_Bending_20220308_3Sweep'), images.detach().cpu().numpy(), suffix='TestSampleObj', sep_folder=True)