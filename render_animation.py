from operator import truediv
import os
from glob import glob
from pickle import TRUE
import numpy as np
import cv2
import torch
from derender import utils, rendering
import neural_renderer as nr

EPS = 1e-7


def load_imgs(flist):
    return torch.stack([torch.FloatTensor(cv2.imread(f) /255.).flip(2) for f in flist], 0).permute(0,3,1,2)


def load_txts(flist):
    return torch.stack([torch.FloatTensor(np.loadtxt(f, delimiter=',')) for f in flist], 0)


def render_views(renderer, cam_loc, canon_sor_vtx, sor_faces, albedo,albedo2, env_map, spec_alpha, spec_albedo,radcol_height,radcol_height2, tx_size):
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
        # rxyz = torch.stack([rx, ry*0, rx*0], 0).unsqueeze(0).to(canon_sor_vtx.device)
        rxyz = torch.stack([rx*0, ry*0, rx*0], 0).unsqueeze(0).to(canon_sor_vtx.device)

        ## translation test
        test_txyz = [[0,0,0]]
        txyz = torch.FloatTensor(test_txyz).to(canon_sor_vtx.device)

        ## rendering multiple objects test
        sor_vtx = rendering.transform_pts(sor_vtx, rxyz, txyz)
        # print("sor_vtx",sor_vtx.shape)

        ##TODO:multi-obj
        sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx[:,:radcol_height,:,:])  # Bx(H-1)xTx3
        normal_map = rendering.get_sor_quad_center_normal(sor_vtx[:,:radcol_height,:,:])  # Bx(H-1)xTx3

        # sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx)  # Bx(H-1)xTx3
        # normal_map = rendering.get_sor_quad_center_normal(sor_vtx)  # Bx(H-1)xTx3
        diffuse, specular = rendering.envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
        tex_im = rendering.compose_shading(albedo, diffuse, spec_albedo.view(b,1,1,1), specular).clamp(0,1)
        # print("sor_vtx_map",sor_vtx_map.shape)
        # print("normal_map",normal_map.shape)
        # print("diffuse",diffuse.shape)
        # print("specular",specular.shape)

        ##TODO:multi-obj
        ## test second object
        sor_vtx_map_2 = rendering.get_sor_quad_center_vtx(sor_vtx[:,radcol_height:,:,:])  # Bx(H-1)xTx3
        normal_map_2 = rendering.get_sor_quad_center_normal(sor_vtx[:,radcol_height:,:,:])  # Bx(H-1)xTx3
        diffuse_2, specular_2 = rendering.envmap_phong_shading(sor_vtx_map_2, normal_map_2, cam_loc, env_map, spec_alpha)
        tex_im_2 = rendering.compose_shading(albedo2, diffuse_2, spec_albedo.view(b,1,1,1), specular_2).clamp(0,1)
        # utils.save_images(out_dir, tex_im.cpu().numpy(), suffix='novel_views_texture1', sep_folder=True)
        # utils.save_images(out_dir, tex_im_2.cpu().numpy(), suffix='novel_views_texture2', sep_folder=True)

        im_rendered = rendering.render_sor_multiObjTest(renderer, sor_vtx, sor_faces.repeat(b,1,1,1,1),radcol_height,radcol_height2, tex_im,tex_im_2, tx_size=tx_size, dim_inside=False).clamp(0, 1)
        ims += [im_rendered]
    ims = torch.stack(ims, 1)  # BxTxCxHxW
    return ims


def render_relight(renderer, cam_loc, sor_vtx, sor_faces, albedo,albedo2, spec_alpha,spec_alpha2, spec_albedo,spec_albedo2,radcol_height,radcol_height2, tx_size):
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

        sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx[:,:radcol_height,:,:])  # Bx(H-1)xTx3
        normal_map = rendering.get_sor_quad_center_normal(sor_vtx[:,:radcol_height,:,:])  # Bx(H-1)xTx3

        ##TODO: multi-Object
        sor_vtx_map_2 = rendering.get_sor_quad_center_vtx(sor_vtx[:,radcol_height:,:,:])  # Bx(H-1)xTx3
        normal_map_2 = rendering.get_sor_quad_center_normal(sor_vtx[:,radcol_height:,:,:])  # Bx(H-1)xTx3

        diffuse, specular = rendering.envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
        tex_im = rendering.compose_shading(albedo, diffuse, spec_albedo.view(b,1,1,1), specular).clamp(0,1)

        ##TODO: multi-Object
        diffuse2, specular2 = rendering.envmap_phong_shading(sor_vtx_map_2, normal_map_2, cam_loc, env_map, spec_alpha)
        tex_im2 = rendering.compose_shading(albedo2, diffuse2, spec_albedo.view(b,1,1,1), specular2).clamp(0,1)

        im_rendered = rendering.render_sor_multiObjTest(renderer, sor_vtx, sor_faces.repeat(b,1,1,1,1),radcol_height,radcol_height2, tex_im,tex_im2, tx_size=tx_size, dim_inside=False).clamp(0, 1)
        ims += [im_rendered]
    ims = torch.stack(ims, 1)  # BxTxCxHxW
    return ims


def main(in_dir,in_dir2, out_dir):
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

    batch_size = 1
    renderer = rendering.get_renderer(world_ori=world_ori, image_size=image_size,fov=fov, fill_back=True, device='cuda:0')

    sor_curve_all = load_txts(sorted(glob(os.path.join(in_dir, 'sor_curve/*_sor_curve.txt'), recursive=True)))
    albedo_all = load_imgs(sorted(glob(os.path.join(in_dir, 'albedo_map/*_albedo_map.png'), recursive=True)))
    mask_gt_all = load_imgs(sorted(glob(os.path.join(in_dir, 'mask_gt/*_mask_gt.png'), recursive=True)))
    pose_all = load_txts(sorted(glob(os.path.join(in_dir, 'pose/*_pose.txt'), recursive=True)))
    material_all = load_txts(sorted(glob(os.path.join(in_dir, 'material/*_material.txt'), recursive=True)))
    env_map_all = load_imgs(sorted(glob(os.path.join(in_dir, 'env_map/*_env_map.png'), recursive=True)))[:,0,:,:]

    sor_curve_all2 = load_txts(sorted(glob(os.path.join(in_dir2, 'sor_curve/*_sor_curve.txt'), recursive=True)))
    albedo_all2 = load_imgs(sorted(glob(os.path.join(in_dir2, 'albedo_map/*_albedo_map.png'), recursive=True)))
    mask_gt_all2 = load_imgs(sorted(glob(os.path.join(in_dir2, 'mask_gt/*_mask_gt.png'), recursive=True)))
    pose_all2 = load_txts(sorted(glob(os.path.join(in_dir2, 'pose/*_pose.txt'), recursive=True)))
    material_all2 = load_txts(sorted(glob(os.path.join(in_dir2, 'material/*_material.txt'), recursive=True)))
    env_map_all2 = load_imgs(sorted(glob(os.path.join(in_dir2, 'env_map/*_env_map.png'), recursive=True)))[:,0,:,:]

    total_num = sor_curve_all.size(0)
    for b0 in range(0, total_num, batch_size):
        print("====main====")
        ## calculate the index
        b1 = min(total_num, b0+batch_size)
        b = b1 - b0
        print("Rendering %d-%d/%d" %(b0, b1, total_num))

        sor_curve = sor_curve_all[b0:b1].to(device)
        albedo = albedo_all[b0:b1].to(device)
        mask_gt = mask_gt_all[b0:b1].to(device)
        pose = pose_all[b0:b1].to(device)
        material = material_all[b0:b1].to(device)
        env_map = env_map_all[b0:b1].to(device)

        sor_curve2 = sor_curve_all2[b0:b1].to(device)
        albedo2 = albedo_all2[b0:b1].to(device)
        mask_gt2 = mask_gt_all2[b0:b1].to(device)
        pose2 = pose_all2[b0:b1].to(device)
        material2 = material_all2[b0:b1].to(device)
        env_map2 = env_map_all2[b0:b1].to(device)

        ## load origin obj( the obj is already parsed )
        vertices_from_obj, faces_from_obj = nr.load_obj(
        os.path.join(in_dir, 'Obj/1.obj'),normalization=False, load_texture=False, texture_size=8)
        vertices_from_obj2, faces_from_obj2 = nr.load_obj(
        os.path.join(in_dir2, 'Obj/2.obj'),normalization=False, load_texture=False, texture_size=8)
        vertices_from_obj[:,1:]*=-1
        vertices_from_obj2[:,1:]*=-1

        radcol_height = vertices_from_obj.shape[0] // sor_circum # radius column & height
        radcol_height2 = vertices_from_obj2.shape[0] // sor_circum # radius column & height
        obj1VerticesSize = radcol_height*sor_circum

        #TODO:multi-obj
        ## test radcol_height x 2
        sor_faces = rendering.get_sor_full_face_idx_multiObjectTest(radcol_height,radcol_height2, sor_circum).to(device)  # 2x(H-1)xWx3
        print("sor_faces",sor_faces.shape)

        #TODO:multi-obj
        # env_map = torch.cat([env_map,env_map2],1)

        ## load origin obj( the obj is already parsed )
        vertices_from_obj, faces_from_obj = nr.load_obj(
        os.path.join(in_dir, 'Obj/1.obj'),normalization=False, load_texture=False, texture_size=8)
        vertices_from_obj2, faces_from_obj2 = nr.load_obj(
        os.path.join(in_dir2, 'Obj/2.obj'),normalization=False, load_texture=False, texture_size=8)
        vertices_from_obj[:,1:]*=-1
        vertices_from_obj2[:,1:]*=-1

        # vertices_from_obj,_,_ = utils.parse3SweepObjData(radcol_height,sor_circum,vertices_from_obj,None,None)
        # vertices_from_obj2,_,_ = utils.parse3SweepObjData(radcol_height2,sor_circum,vertices_from_obj2,None,None)

        ## normalize from NR loadObj
        vertices_from_obj_normalize = torch.cat([vertices_from_obj,vertices_from_obj2],0)
        print("vertices_from_obj_normalize",vertices_from_obj_normalize.shape)
        print("vertices_from_obj",vertices_from_obj.shape)
        print("vertices_from_obj2",vertices_from_obj2.shape)
        vertices_from_obj_normalize = utils.normalizeObjVertices(vertices_from_obj_normalize)

        ## re-assign
        vertices_from_obj = vertices_from_obj_normalize[:obj1VerticesSize,:]
        vertices_from_obj2 = vertices_from_obj_normalize[obj1VerticesSize:,:]
        print("vertices_from_obj",vertices_from_obj.shape)
        print("vertices_from_obj2",vertices_from_obj2.shape)

        canon_sor_vtx_obj = vertices_from_obj.reshape(1,radcol_height,sor_circum,3)
        canon_sor_vtx_obj2 = vertices_from_obj2.reshape(1,radcol_height2,sor_circum,3)

        # print("vertices_from_obj shape",vertices_from_obj.shape)
        # print("faces_from_obj shape",faces_from_obj.shape)
        print("canon_sor_vtx_obj shape",canon_sor_vtx_obj.shape)
        print("canon_sor_vtx_obj2 shape",canon_sor_vtx_obj2.shape)

        ## get vertices from sor_curve
        canon_sor_vtx = rendering.get_sor_vtx(sor_curve, sor_circum)  # BxHxTx3
        canon_sor_vtx2 = rendering.get_sor_vtx(sor_curve2, sor_circum)  # BxHxTx3
        print("canon_sor_vtx shape",canon_sor_vtx.shape)
        print("canon_sor_vtx2 shape",canon_sor_vtx.shape)

        ##TODO:hard code tranlation
        test_txyz_obj2 = [[-0.6,0,0]]
        txyz_obj2 = torch.FloatTensor(test_txyz_obj2).to(canon_sor_vtx2.device)
        canon_sor_vtx2 = rendering.transform_pts(canon_sor_vtx2, None, txyz_obj2)

        #TODO:multi-obj
        # test concatenate at dimension 1
        canon_sor_vtx_obj = torch.cat([canon_sor_vtx_obj,canon_sor_vtx_obj2],1)
        canon_sor_vtx = torch.cat([canon_sor_vtx,canon_sor_vtx2],1)
        print("canon_sor_vtx_obj after cat",canon_sor_vtx_obj.shape)
        print("canon_sor_vtx after cat",canon_sor_vtx.shape)

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

        spec_alpha, spec_albedo = material.unbind(1)
        spec_alpha2, spec_albedo2 = material2.unbind(1)

        ## replicate albedo
        wcrop_ratio = 1/6
        wcrop_tex_im = int(wcrop_ratio * tex_im_w//2)
        albedo = rendering.gamma(albedo)
        p = 8
        front_albedo = torch.cat([albedo[:,:,:,p:2*p].flip(3), albedo[:,:,:,p:-p], albedo[:,:,:,-2*p:-p].flip(3)], 3)
        albedo_replicated = torch.cat([front_albedo[:,:,:,:wcrop_tex_im].flip(3), front_albedo, front_albedo.flip(3), front_albedo[:,:,:,:-wcrop_tex_im]], 3)
        # utils.save_images(out_dir, albedo_replicated.cpu().numpy(), suffix='albedo_replicated', sep_folder=True)
        # utils.save_images(out_dir, front_albedo.cpu().numpy(), suffix='front_albedo', sep_folder=True)
        
        ##TODO:multiobject
        albedo2 = rendering.gamma(albedo2)
        front_albedo2 = torch.cat([albedo2[:,:,:,p:2*p].flip(3), albedo2[:,:,:,p:-p], albedo2[:,:,:,-2*p:-p].flip(3)], 3)
        albedo_replicated2 = torch.cat([front_albedo2[:,:,:,:wcrop_tex_im].flip(3), front_albedo2, front_albedo2.flip(3), front_albedo2[:,:,:,:-wcrop_tex_im]], 3)
        # utils.save_images(out_dir, albedo_replicated2.cpu().numpy(), suffix='albedo_replicated2', sep_folder=True)
        # utils.save_images(out_dir, front_albedo2.cpu().numpy(), suffix='front_albedo2', sep_folder=True)
        
        with torch.no_grad():
            if apply_origin_vertices == True :
                novel_views = render_views(renderer, cam_loc, canon_sor_vtx_obj, sor_faces, albedo_replicated,albedo_replicated2, env_map, spec_alpha, spec_albedo,radcol_height,radcol_height2, tx_size)
            else:
                novel_views = render_views(renderer, cam_loc, canon_sor_vtx, sor_faces, albedo_replicated,albedo_replicated2, env_map, spec_alpha, spec_albedo,radcol_height,radcol_height2, tx_size)
            relightings = render_relight(renderer, cam_loc, sor_vtx_relighting, sor_faces, albedo_replicated,albedo_replicated2, spec_alpha,spec_alpha2, spec_albedo,spec_albedo2,radcol_height,radcol_height2, tx_size)
            [utils.save_images(out_dir, novel_views[:,i].cpu().numpy(), suffix='novel_views_%d'%i, sep_folder=True) for i in range(0, novel_views.size(1), novel_views.size(1)//10)]
            utils.save_videos(out_dir, novel_views.cpu().numpy(), suffix='novel_view_videos', sep_folder=True, fps=25)
            [utils.save_images(out_dir, relightings[:,i].cpu().numpy(), suffix='relight_%d'%i, sep_folder=True) for i in range(0, relightings.size(1), relightings.size(1)//10)]
            utils.save_videos(out_dir, relightings.cpu().numpy(), suffix='relight_videos', sep_folder=True, fps=25)

if __name__ == '__main__':
    in_dir = 'results/TestResults_20220321_teapot_1_part1'
    in_dir2 = 'results/TestResults_20220321_teapot_1_part2'
    out_dir = 'results/TestResults_20220324_teapot_1/animations'
    main(in_dir,in_dir2, out_dir)
