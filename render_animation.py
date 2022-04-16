from operator import truediv
import os
from glob import glob
from pickle import TRUE
import numpy as np
import cv2
import torch
from derender import utils, rendering
import neural_renderer as nr

def load_imgs(flist):
    return torch.stack([torch.FloatTensor(cv2.imread(f) /255.).flip(2) for f in flist], 0).permute(0,3,1,2)

def load_txts(flist):
    return torch.stack([torch.FloatTensor(np.loadtxt(f, delimiter=',')) for f in flist], 0)

def load_sor_curve_txt(flist):
    sor_curve_all = torch.tensor([])
    radcol_height_list = []
    for f in flist:
        ## sor_curve of each components
        sor_curve_component = torch.FloatTensor(np.loadtxt(f, delimiter=','))
        sor_curve_all = torch.cat([sor_curve_all,sor_curve_component],0)
        
        ## radcol height of each components
        radcol_height_component = sor_curve_component.size(0)
        radcol_height_list.append(radcol_height_component)
    return sor_curve_all, radcol_height_list

def load_obj(flist):
    vertices_all = torch.tensor([]).cuda()
    faces_all = torch.tensor([]).int().cuda()
    for f in flist:
        vertices,faces= nr.load_obj(f,normalization=False, load_texture=False, texture_size=8)
        vertices_all = torch.cat([vertices_all,vertices],0)
        faces_all = torch.cat([faces_all,faces],0)
    return vertices_all,faces_all

def render_views_multiObject(renderer, cam_loc, canon_sor_vtx, sor_faces, albedo_list, env_map, spec_alpha, spec_albedo,radcol_height_list, tx_size):
    b = canon_sor_vtx.size(0)
    print("====render_views====",)
    s = 80 # sample number
    rxs = torch.linspace(0, np.pi/3, s//2) # rotation x axis (roll)
    rxs = torch.cat([rxs, rxs.flip(0)], 0) # rotation x axis back to origin pose
    rys = torch.linspace(0, 2*np.pi, s) # rotation y axis (pitch)

    ims = []
    for i, (rx, ry) in enumerate(zip(rxs, rys)):

        ## rotate y-axis first then rotate x-axis
        rxyz = torch.stack([rx*0, ry, rx*0], 0).unsqueeze(0).to(canon_sor_vtx.device)
        sor_vtx = rendering.transform_pts(canon_sor_vtx, rxyz, None)
        rxyz = torch.stack([rx*0, ry*0, rx*0], 0).unsqueeze(0).to(canon_sor_vtx.device)

        ## rendering multiple objects test
        sor_vtx = rendering.transform_pts(sor_vtx, rxyz, None)

        tex_im_list = []
        for j in range(len(radcol_height_list)):
            h_start = sum(radcol_height_list[:j])
            h_end = sum(radcol_height_list[:j+1])
            sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx[:,h_start:h_end,:,:])  # Bx(H-1)xTx3
            normal_map = rendering.get_sor_quad_center_normal(sor_vtx[:,h_start:h_end,:,:])  # Bx(H-1)xTx3
            diffuse, specular = rendering.envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
            tex_im = rendering.compose_shading(albedo_list[j], diffuse, spec_albedo.view(b,1,1,1), specular).clamp(0,1)
            tex_im_list.append(tex_im)
            # print("sor_vtx_map",sor_vtx_map.shape)
            # print("normal_map",normal_map.shape)
            # print("diffuse",diffuse.shape)
            # print("specular",specular.shape)

        im_rendered = rendering.render_sor_multiObject(renderer, sor_vtx, sor_faces.repeat(b,1,1,1,1),radcol_height_list, tex_im_list, tx_size=tx_size, dim_inside=False).clamp(0, 1)
        ims += [im_rendered]
    ims = torch.stack(ims, 1)  # BxTxCxHxW
    return ims


def render_relight_multiObject(renderer, cam_loc, sor_vtx, sor_faces, albedo_list, spec_alpha_list, spec_albedo_list,radcol_height_list, tx_size):
    b = sor_vtx.size(0)
    lam = 20
    F = 0.15
    env_amb = 0.015
    n_sgls = 1
    sgl_lams = torch.FloatTensor([lam]).repeat(b, n_sgls).to(sor_vtx.device)
    sgl_Fs = torch.FloatTensor([F]).repeat(b, n_sgls).to(sor_vtx.device) *sgl_lams**0.5

    s = 80
    azims = torch.linspace(0, 4*np.pi, s)
    elevs = torch.linspace(0, np.pi/2, s//2)
    elevs = torch.cat([elevs, elevs.flip(0)], 0)

    ims = []
    for i, (azim, elev) in enumerate(zip(azims, elevs)):
        dy = -elev.sin()
        dx = elev.cos() * azim.sin()
        dz = -elev.cos() * azim.cos()
        sgl_dirs = torch.stack([dx, dy, dz], 0).repeat(b, n_sgls, 1).to(sor_vtx.device)
        sg_lights = torch.cat([sgl_dirs, sgl_lams.unsqueeze(2), sgl_Fs.unsqueeze(2)], 2).to(sor_vtx.device)

        env_map = rendering.sg_to_env_map(sg_lights, n_elev=16, n_azim=48)
        env_map_ambient = torch.FloatTensor([env_amb]).repeat(b).to(sor_vtx.device)
        env_map = env_map + env_map_ambient.view(b,1,1)

        # TODO: test use sample spec_alpha spec_albedo
        spec_alpha = spec_alpha_list[0]
        spec_albedo = spec_albedo_list[0]

        tex_im_list = []
        for j in range(len(radcol_height_list)):
            h_start = sum(radcol_height_list[:j])
            h_end = sum(radcol_height_list[:j+1])
            sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx[:,h_start:h_end,:,:])  # Bx(H-1)xTx3
            normal_map = rendering.get_sor_quad_center_normal(sor_vtx[:,h_start:h_end,:,:])  # Bx(H-1)xTx3
            diffuse, specular = rendering.envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
            tex_im = rendering.compose_shading(albedo_list[j], diffuse, spec_albedo.view(b,1,1,1), specular).clamp(0,1)
            tex_im_list.append(tex_im)

        im_rendered = rendering.render_sor_multiObject(renderer, sor_vtx, sor_faces.repeat(b,1,1,1,1),radcol_height_list, tex_im_list, tx_size=tx_size, dim_inside=False).clamp(0, 1)
        ims += [im_rendered]
    ims = torch.stack(ims, 1)  # BxTxCxHxW
    return ims


def main(in_dir, out_dir):
    device = 'cuda:0'

    # set sor_circum to 24 fit 3sweep object
    sor_circum = 24 

    image_size = 256
    tex_im_h = 256
    tex_im_w = 768 ## 256*3 => 3 times width of texture
    env_map_h = 16
    env_map_w = 48
    fov = 10  # in degrees
    ori_z = 12.5 # camera z-axis orientation
    world_ori = [0,0,ori_z]
    tx_size = 8
    cam_loc = torch.FloatTensor([0,0,-ori_z]).to(device) # camera position

    apply_origin_vertices = True
    batch_size = 1 # for multiObject fix 1

    renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size,fov=fov, fill_back=True, device='cuda:0')
    
    # data type: sor_curve_all => tensor, radcol_height_list => list
    sor_curve_all,radcol_height_list = load_sor_curve_txt(sorted(glob(os.path.join(in_dir, 'sor_curve/*_sor_curve.txt'), recursive=True)))
    material_all = load_txts(sorted(glob(os.path.join(in_dir, 'material/*_material.txt'), recursive=True)))
    pose_all = load_txts(sorted(glob(os.path.join(in_dir, 'pose/*_pose.txt'), recursive=True)))
    albedo_all = load_imgs(sorted(glob(os.path.join(in_dir, 'albedo_map/*_albedo_map.png'), recursive=True)))
    mask_gt_all = load_imgs(sorted(glob(os.path.join(in_dir, 'mask_gt/*_mask_gt.png'), recursive=True)))
    env_map_all = load_imgs(sorted(glob(os.path.join(in_dir, 'env_map/*_env_map.png'), recursive=True)))[:,0,:,:]

    ## list
    vertices_obj_all,faces_obj_all = load_obj(sorted(glob(os.path.join(in_dir, 'obj_parsed/*_obj_parsed.obj'), recursive=True)))

    component_num = len(radcol_height_list)
    print("component_num",component_num)

    vertices_size_list = []
    spec_alpha_list = []
    spec_albedo_list = []
    albedo_replicated_list = []

    canon_sor_vtx = torch.empty(0).to(device)
    canon_sor_vtx_obj = torch.empty(0).to(device)

    ##TODO: test only use the same env_map,mask,pose for full object
    env_map = env_map_all[:1].to(device)
    mask_gt = mask_gt_all[:1].to(device)
    pose = pose_all[:1].to(device) # => 0,0,0
    
    ## set multi-object data list
    for i in range(0, component_num):
        
        index_start = sum(radcol_height_list[:i])
        index_end = sum(radcol_height_list[:i+1])

        sor_curve = sor_curve_all[index_start:index_end].to(device)
        material = material_all[i:i+1].to(device)
        albedo = albedo_all[i:i+1].to(device)
        
        ## calculate paramter
        vertices_size = radcol_height_list[i]*sor_circum # for indexing
        vertices_size_list.append(vertices_size)

        print("sor_curve",sor_curve.shape)
        # set sor_vtx map
        canon_sor_vtx_component =  rendering.get_sor_vtx(sor_curve.repeat(batch_size,1,1), sor_circum)
        canon_sor_vtx = torch.cat([canon_sor_vtx,canon_sor_vtx_component],1)

        ## specular
        spec_alpha, spec_albedo = material.unbind(1)
        spec_alpha_list.append(spec_alpha)
        spec_albedo_list.append(spec_albedo)

        ## replicate albedo
        wcrop_ratio = 1/6
        wcrop_tex_im = int(wcrop_ratio * tex_im_w//2)
        albedo = rendering.gamma(albedo)
        p = 8
        front_albedo = torch.cat([albedo[:,:,:,p:2*p].flip(3), albedo[:,:,:,p:-p], albedo[:,:,:,-2*p:-p].flip(3)], 3)
        albedo_replicated = torch.cat([front_albedo[:,:,:,:wcrop_tex_im].flip(3), front_albedo, front_albedo.flip(3), front_albedo[:,:,:,:-wcrop_tex_im]], 3)
        albedo_replicated_list.append(albedo_replicated)
        utils.save_images(out_dir, albedo_replicated.cpu().numpy(), suffix='albedo_replicated', sep_folder=True)
        utils.save_images(out_dir, front_albedo.cpu().numpy(), suffix='front_albedo', sep_folder=True)

    ## get sor_faces with all component
    sor_faces = rendering.get_sor_full_face_idx_multiObject(radcol_height_list, sor_circum).to(device)  # 2x(H-1)xWx3
    print("sor_faces",sor_faces.shape)

    ## normalize from NR loadObj
    vertices_obj_all_normalized = utils.normalizeObjVertices(vertices_obj_all)
    print("vertices_obj_all_normalized",vertices_obj_all_normalized.shape)

    ## get sor vertices data
    for i in range(len(radcol_height_list)):
        ## re-assign the normalize vertices
        index_vertices_start = sum(vertices_size_list[:i])
        index_vertices_end = sum(vertices_size_list[:i+1])
        vertices_component_normalized = vertices_obj_all_normalized[index_vertices_start:index_vertices_end].reshape(1,radcol_height_list[i],sor_circum,3)

        ## concate to fit RADAR data dimension
        canon_sor_vtx_obj = torch.cat([canon_sor_vtx_obj,vertices_component_normalized],1)
    print("canon_sor_vtx_obj",canon_sor_vtx_obj.shape)

    ## test for relighting
    rxyz = pose[:,:3] / 180 * np.pi # 1x3
    txy = pose[:,3:] # 1x2
    tz = torch.zeros(len(txy), 1).to(txy.device) ## set z-transform to zero
    txyz = torch.cat([txy, tz], 1)
    
    if apply_origin_vertices == True:
        sor_vtx_relighting = rendering.transform_pts(canon_sor_vtx_obj, rxyz, txyz)
    else:
        sor_vtx_relighting = rendering.transform_pts(canon_sor_vtx, rxyz, txyz)

    # sor_vtx_map_relighting = rendering.get_sor_quad_center_vtx(sor_vtx_relighting)  # Bx(H-1)xTx3
    # normal_map_relighting = rendering.get_sor_quad_center_normal(sor_vtx_relighting)  # Bx(H-1)xTx3
    
    ##TODO: test only use the same env_map
    spec_alpha = spec_alpha_list[0]
    spec_albedo = spec_albedo_list[0]

    with torch.no_grad():
        if apply_origin_vertices == True :
            novel_views = render_views_multiObject(renderer, cam_loc, canon_sor_vtx_obj, sor_faces, albedo_replicated_list, env_map, spec_alpha, spec_albedo,radcol_height_list, tx_size)
        else:
            novel_views = render_views_multiObject(renderer, cam_loc, canon_sor_vtx, sor_faces, albedo_replicated_list, env_map, spec_alpha, spec_albedo,radcol_height_list, tx_size)
        relightings = render_relight_multiObject(renderer, cam_loc, sor_vtx_relighting, sor_faces, albedo_replicated_list, spec_alpha_list, spec_albedo_list,radcol_height_list, tx_size)
        [utils.save_images(out_dir, novel_views[:,i].cpu().numpy(), suffix='novel_views_%d'%i, sep_folder=True) for i in range(0, novel_views.size(1), novel_views.size(1)//10)]
        utils.save_videos(out_dir, novel_views.cpu().numpy(), suffix='novel_view_videos', sep_folder=True, fps=25)
        [utils.save_images(out_dir, relightings[:,i].cpu().numpy(), suffix='relight_%d'%i, sep_folder=True) for i in range(0, relightings.size(1), relightings.size(1)//10)]
        utils.save_videos(out_dir, relightings.cpu().numpy(), suffix='relight_videos', sep_folder=True, fps=25)

if __name__ == '__main__':
    in_dir = 'results/TestResults_20220412_spout_vase_biggerCurveX'
    out_dir = 'results/TestResults_20220412_spout_vase_biggerCurveX/animations'
    main(in_dir, out_dir)
