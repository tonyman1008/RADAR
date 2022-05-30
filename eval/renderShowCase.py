import neural_renderer as nr
import os
import torch
from derender import utils, rendering

def sample3SweepOriginalFullView(objFolder):
    print("====Start sample 3sweep full view  ====")

    ## same camera parameter with RADAR
    image_size = 256
    fov = 10
    ori_z = 12.5
    world_ori=[0,0,ori_z] # make sure the camera pose is the same
    tx_size = 16

    ## initial setting
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_dir = current_dir + '/3SweepData/' + objFolder
    test_dataSet_dir = current_dir + '/3SweepData/' + objFolder  # final test data folder

    renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size,fov=fov, fill_back=True)

    ## load obj with normalization
    vertices, faces, textures = nr.load_obj(
        os.path.join(input_dir, 'full.obj'),normalization=True, load_texture=True, texture_size=tx_size)

    vertices[:,1:]*=-1

    images = renderer.render_rgb(vertices.reshape(1,-1,3), faces[None, :, :], textures[None, :, :, :, :, :])
    utils.save_images(test_dataSet_dir, images.detach().cpu().numpy(), suffix='3Sweep_view', sep_folder=True)
    print("====Sample 3sweep full view finished ====")

if __name__ == '__main__':
    objFolder = 'TestData_20220523/L_2009_22_201'
    objNum = 3

    ## sample original view from 3sweep
    sample3SweepOriginalFullView(objFolder)

    print("====Sample View Complete !!! =====")