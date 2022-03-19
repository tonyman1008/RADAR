import numpy as np
import torch

faces1 = torch.zeros(3,2,3)+1
faces2 = torch.zeros(3,2,3)+2

faces3 = torch.zeros(4,2,3)+3
faces4 = torch.zeros(4,2,3)+4

full_face_obj1 = torch.stack([faces1, faces2], 0).int()
full_face_obj2 = torch.stack([faces3, faces4], 0).int()
full_face_allObjects = torch.cat([full_face_obj1,full_face_obj2],1)
print("full_face_obj1",full_face_obj1.shape)
print("full_face_obj1",full_face_obj1)
print("full_face_obj2",full_face_obj2.shape)
print("full_face_obj2",full_face_obj2)
print("full_face_allObjects",full_face_allObjects.shape)

# full_face_obj1 = torch.cat([faces1, faces3], 0).int()
# full_face_obj2 = torch.cat([faces2, faces4], 0).int()
# full_face_allObjects = torch.stack([full_face_obj1,full_face_obj2],0)
# print("full_face_obj1",full_face_obj1.shape)
# print("full_face_obj1",full_face_obj1)
# print("full_face_obj2",full_face_obj2.shape)
# print("full_face_obj2",full_face_obj2)
# print("full_face_allObjects",full_face_allObjects.shape)

full_face_allObjects = full_face_allObjects.reshape(1,-1,3)
print("full_face_allObjects",full_face_allObjects.shape)
print("full_face_allObjects",full_face_allObjects)