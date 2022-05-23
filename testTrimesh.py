import numpy as np
import trimesh
trimesh.util.attach_to_log()

mesh = trimesh.load_mesh('data/test_subdivision_2/1.obj')
## subdivide
sor_circum = 24
rad_height = len(mesh.vertices) // sor_circum

new_sor_cirum = sor_circum *2
new_rad_height = rad_height*2 -1

new_faces_num = len(mesh.faces)*4
new_vertices_num = new_sor_cirum*new_rad_height

print("faces shape",mesh.faces.shape)
print("vertices shape",mesh.vertices.shape)
# print("rad_height",rad_height)
# print("new_sor_cirum",new_sor_cirum)
# print("new_rad_height",new_rad_height)
# print("new_faces_num",new_faces_num)
# print("new_vertices_num",new_vertices_num)

print(mesh.metadata['vertex_texture'])

mesh.show()

# idx_map = mesh.vertices.reshape(rad_height,sor_circum,3)

# # print("idx_map",idx_map)
# print("idx_map shape",idx_map.shape)

# idx_map_new = np.empty((0,3))
# for i in range(rad_height):

#     # original row (append index)
#     row = np.empty((0,3))
#     for j in range(sor_circum):
#         next_point_index = 0 if j== sor_circum-1 else j+1

#         mid_point = np.array((idx_map[i,j,:] + idx_map[i,next_point_index,:]) * 0.5)
#         # 2 element for a set
#         set = np.vstack((idx_map[i,j,:],mid_point))
#         row = np.vstack((row,set))
#     idx_map_new = np.vstack((idx_map_new,row))
    
#     # middle row (between row and row)
#     if i != rad_height-1:
        
#         row_mid = np.empty((0,3))
#         for k in range(sor_circum):
#             left_point = (idx_map[i,k,:] + idx_map[i+1,k,:]) * 0.5
#             next_point_index = 0 if k == sor_circum-1 else k+1
#             right_point = (idx_map[i,next_point_index,:] + idx_map[i+1,next_point_index,:]) * 0.5
#             mid_point = (left_point + right_point) *0.5
#             # 3 element for a set
#             set = np.vstack((left_point,mid_point))
#             row_mid = np.vstack((row_mid,set))
#         idx_map_new = np.vstack((idx_map_new,row_mid))

# print("idx_map_new shape",idx_map_new.shape)

# ## faces
# h = new_rad_height
# w = new_sor_cirum
# faces_map = np.arange(h*w).reshape(h,w)  # HxW
# faces_map = np.concatenate([faces_map, faces_map[:,:1]], 1)  # Hx(W+1), connect last column to first

# # ##origin 2 triangles(faces) vertice index 
# faces1 = np.stack([faces_map[:h-1,:w], faces_map[1:,:w], faces_map[:h-1,1:w+1]], -1)  # (H-1)xWx3
# faces2 = np.stack([faces_map[1:,1:w+1], faces_map[:h-1,1:w+1], faces_map[1:,:w]], -1)  # (H-1)xWx3

# final_faces_map = np.stack([faces1, faces2], 0) # 2x(H-1)xWx3
# final_faces_map = final_faces_map.reshape(-1,3)
# print("final_faces_map",final_faces_map.shape)
# print(final_faces_map)   

# newMesh = trimesh.Trimesh(idx_map_new,final_faces_map)

# mesh = mesh.subdivide()
# mesh.show()
# newMesh.export('data/test_subdivision/subdivision.obj',file_type='obj')
# mesh.export('data/test_subdivision/subdivision_trimesh.obj',file_type='obj')

## smooth test
# mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.8, iterations=10, laplacian_operator=None)
