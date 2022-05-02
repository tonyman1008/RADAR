import numpy as np
import trimesh
trimesh.util.attach_to_log()

mesh = trimesh.load_mesh('data/obj_parsed/00002_obj_parsed.obj')

print("vertice",mesh.vertices)

mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.8, iterations=10, laplacian_operator=None)

mesh.show()

mesh.export('data/obj_parsed/smooth.obj',file_type='obj')