import neural_renderer as nr
import os
from skimage.io import imsave
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))

renderer = nr.Renderer(camera_mode='look_at')

vertices, faces, textures = nr.load_obj(
    os.path.join(current_dir, 'TestData_20220225/Test_Straight_3.obj'), load_texture=True)

print("vertices",vertices.shape)
print("faces",faces.shape)
print("textures",textures.shape)

# renderer.eye = nr.get_points_from_angles(2, 15, 30)
# images, _, _ = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :])
# images = images.permute(0,2,3,1).detach().cpu().numpy()
# imsave(os.path.join(current_dir, 'TestData_20220225/car.png'), images[0])

vertices, faces, textures = nr.load_obj(
    os.path.join(current_dir, 'TestData_20220225/Test_Straight_3.obj'), load_texture=True, texture_size=16)
renderer.eye = nr.get_points_from_angles(3, 0, 0)
images, _, _ = renderer.render(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :])
images = images.permute(0,2,3,1).detach().cpu().numpy()
imsave(os.path.join(current_dir, 'TestData_20220225/test.png'), images[0])