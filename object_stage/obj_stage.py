import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import argparse
import numpy as np
from pathlib import Path
import json
from utils.transform import *
from utils.io import *
from utils.loss import *
import shutil

import torch
import drjit
import psdr_jit as psdr
from drjit.cuda.ad import Float as FloatD, Array3f as Vector3fD, Matrix4f as Matrix4fD
from drjit.cuda import Float as FloatC, Array3f as Vector3fC, Matrix4f as Matrix4fC
from torch.optim import Adam
from time import time

import matplotlib.pyplot as plt


lr_trans = 0.01
lr_scale = 0.01
lr_rotate = 1.0
forward_npass = 1
backward_npass = 1
thold = 0.5
save_iter = 10
mesh_save_iter = 10


def get_parser():
    parser = argparse.ArgumentParser(description="scene initalization")
    parser.add_argument(
        "scene",
        default="AdobeStock_5889655",
        help="scene to run",
    )
    return parser


class RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, integrator, sc, mesh_id, param_list, *param):
        if len(param_list) != len(param):
            print(param_list)
            print(param)
            print("error: param size mismatch")
            exit()


        for pdict, pid in zip(param_list, param):
            if pdict['type'] == 'obj_toworld':
                sc.param_map[f"Mesh[{pdict['id']}]"].set_transform(pid.detach().cpu().numpy(), False)
            else:
                assert("invalid parameter")
            # print(pdict, pid)

        sc.configure()

        for ii in range(0, forward_npass):
            if ii == 0:
                psdr_image = integrator.renderC(sc, 0)#psdr_image.torch().to('cuda').to(torch.float32)
            else:
                psdr_image += integrator.renderC(sc, 0)
        image = psdr_image.torch()
        image /= forward_npass
        ctx.scene = sc
        ctx.integrator = integrator
        ctx.param_list = param_list
        ctx.mesh_id = mesh_id
        ctx.param = param
        del psdr_image
        return image.reshape((sc.opts.height, sc.opts.width, 3))

    @staticmethod
    def backward(ctx, grad_out):
        drjit_param = []
        # print(ctx.param_list)
        for pdict, pid in zip(ctx.param_list, ctx.param):
            if ctx.mesh_id == -1 or pdict['id'] == ctx.mesh_id: 
                sc.param_map[f"Mesh[{pdict['id']}]"].enable_edges = True
                if pdict['type'] == 'obj_toworld':
                    drjit_param.append(sc.param_map[f"Mesh[{pdict['id']}]"].to_world_right)
                drjit.enable_grad(drjit_param[-1])
            else:
                sc.param_map[f"Mesh[{pdict['id']}]"].enable_edges = False
                if pdict['type'] == 'obj_toworld':
                    drjit_param.append(sc.param_map[f"Mesh[{pdict['id']}]"].to_world_right)
                # drjit.enable_grad(drjit_param[-1])

        ctx.scene.configure([0])
        image_grad = Vector3fC(grad_out.reshape(-1, 3))

        param_grad = []
        grad_tmp = []
        for ii in range(0, backward_npass):
            imaged = ctx.integrator.renderD(ctx.scene, 0)
            tmp = drjit.dot(image_grad, imaged)
            drjit.backward(tmp)
            if ii == 0:
                for dpar in range(0, len(ctx.param)):
                    grad_tmp.append(drjit.grad(drjit_param[dpar]) / backward_npass)
            else:
                for dpar in range(0, len(ctx.param)):
                    grad_tmp[dpar] += drjit.grad(drjit_param[dpar]) / backward_npass

            del tmp, imaged, drjit_param

        for gtmp in grad_tmp:
            param_grad.append(torch.nan_to_num(gtmp.torch().cuda()))
        return tuple([None] * 4 + param_grad)


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, integrator, scene, mesh_id, param_list=[], params=[]):
        image = RenderFunction.apply(integrator, scene, mesh_id, param_list, *params)
        return image


if __name__ == "__main__":
    args = get_parser().parse_args()

    data_path = Path("input", args.scene)


    colormap = plt.get_cmap('inferno')

    obj_stage_init_path = data_path / "obj_stage_init"
    mask_path = data_path / "mask"

    obj_stage_path = Path("output", args.scene, "obj_stage")
    obj_stage_path.mkdir(parents=True, exist_ok=True)


    with open(str(obj_stage_init_path / "obj_stage_config.json"), "r") as f:
        scene_json = json.load(f)

    target_image = torch.tensor((read_png(data_path / "image_resize.png")), device='cuda', dtype=torch.float32)
    target_depth = torch.tensor((cv2.imread(str(data_path / "depth" / "depth_normalized.exr"))), device='cuda', dtype=torch.float32)
    img_shape = target_image.shape
    target_mask = []
    for obj_data in scene_json['obj_init']:
        mask = torch.tensor((read_png(mask_path / f"{obj_data['mask_name']}.png")), device='cuda', dtype=torch.float32)
        mask[mask>0.00001] = 1.0
        target_mask.append(mask)



    sc = psdr.Scene()
    sc.opts.spp = 4 # Interior Term
    sc.opts.sppe = 16 # Primary Edge
    sc.opts.sppse = 0 
    sc.opts.width = int(img_shape[1])
    sc.opts.height = int(img_shape[0])
    sc.opts.log_level = 0

    integrator = psdr.FieldExtractionIntegrator("depth") 


    sensor = psdr.PerspectiveCamera(scene_json["cameraFoV"], 0.000001, 10000000.)

    to_world = Matrix4fD(rotate3D(-scene_json["cameraPitch"], 0, scene_json["cameraRoll"], device='cpu').numpy().astype(float).tolist())


    sensor.to_world = to_world
    sc.add_Sensor(sensor)
    sc.add_BSDF(psdr.DiffuseBSDF([0.5, 0.5, 0.5]), "obj")

    # for debug_obj in glob.glob(str(data_path / "debug" / "inital_pc" / "*_unit_rev.obj")):
    #     sc.add_Mesh(debug_obj, Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "obj", None)

    scene_graph = {}

    to_world_raw = {}
    for oid, obj_data in enumerate(scene_json['obj_init']):
        if 'scene_graph' in scene_json.keys():
            if obj_data['mask_name'] in scene_json['scene_graph'].keys():
                parent_obj = scene_json['scene_graph'][obj_data['mask_name']]

                for oid2, obj_data2 in enumerate(scene_json['obj_init']):
                    if obj_data2['mask_name'] == parent_obj:
                        scene_graph[oid] = oid2
                        break
            



        to_world = torch.mm(translate(obj_data['center']), scale(obj_data['scale']))
        if obj_data['is_on_wall']:
            to_world = torch.mm(to_world, rotate3D(0, 180+scene_json['scene']['wall'][obj_data['wall_id']]+scene_json['scene']['room']['rotate'], 0, device='cpu'))

        else:
            to_world = torch.mm(to_world, rotate3D(0, 180+obj_data['rotate'], 0, device='cpu'))
        sc.add_Mesh(str(obj_stage_init_path / f"{obj_data['model']}.obj"), Matrix4fC(to_world.numpy().tolist()), "obj", None)
    # print(scene_graph)
    # exit()
    sc.configure()
    img = integrator.renderC(sc, 0)

    org_img = img.numpy().reshape((sc.opts.height, sc.opts.width, 3))
    output = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    # write_png(data_path / "debug" / "obj_init.png", output)

    

    opt_para = []
    trans_vec_list = []
    scale_vec_list = []
    rot_vec_list = []
    psdr_opt_dict = []

    for oid, obj_data in enumerate(scene_json['obj_init']):



        trans_vec = torch.tensor([0,0,0], device='cuda', dtype=torch.float32).requires_grad_()
        trans_vec_list.append(trans_vec)
        opt_para.append({'params': trans_vec, 'lr':  lr_trans})

        if 'square' in obj_data['model']:
            scale_vec = torch.tensor([1,1,1], device='cuda', dtype=torch.float32).requires_grad_()
        else:
            scale_vec = torch.tensor([1], device='cuda', dtype=torch.float32).requires_grad_()


        scale_vec_list.append(scale_vec)
        opt_para.append({'params': scale_vec, 'lr':  lr_scale})


        rot_vec = torch.tensor([0.0], device='cuda', dtype=torch.float32).requires_grad_()
        rot_vec_list.append(rot_vec)
        # print(obj_data['cat_name'])
        if obj_data['is_on_wall']:
            opt_para.append({'params': rot_vec, 'lr':  0})
        else:
            opt_para.append({'params': rot_vec, 'lr':  lr_rotate})

        psdr_opt_dict.append({'type':'obj_toworld', 'id':oid})

    # exit()
    # opt_floor_height = torch.tensor(scene_json['scene']['floor_height'], device='cuda', dtype=torch.float32)


    opt_floor_height = torch.tensor(scene_json['scene']['floor_height'], device='cuda', dtype=torch.float32).requires_grad_()
    opt_para.append({'params': opt_floor_height, 'lr': 0.0})
    optimizer = Adam(opt_para)



    resolution = sc.opts.height*sc.opts.width

    psdr_render = Renderer()

    
    # write_png(f"tar_0.png", target_mask[0])

    optimizer.zero_grad()
    num_obj = len(scene_json['obj_init'])

    debug_loss = 0.
    debug_loss_mask = 0.
    debug_depth_loss = 0.



    for subiter in (range(301*num_obj)):

        it = subiter // num_obj
        mid = subiter % num_obj

        if mid == 0:
            t0 = time()


        
        to_world_list = []
        # print(rot_vec_list)
        for oid, obj_data in enumerate(scene_json['obj_init']):
            to_world_tmp = torch.mm(translate(trans_vec_list[oid]), scale(scale_vec_list[oid]))
            to_world_tmp = torch.mm(to_world_tmp, rotate(torch.tensor([0,1,0], device='cuda', dtype=torch.float32), rot_vec_list[oid]))
            to_world_list.append(to_world_tmp)

        
        
        
        
        curr_depth = psdr_render(psdr.FieldExtractionIntegrator("depth"), sc, -1, psdr_opt_dict, to_world_list)

        curr_mask = psdr_render(psdr.FieldExtractionIntegrator(f"silhouette {mid}"), sc, mid, psdr_opt_dict,  to_world_list)

        # CHANGE TO 2CONV1D
        loss_mask = pyramid_loss(target_mask[mid], curr_mask, 8)*4


        depth_loss = torch.tensor(0.0, device='cuda',dtype=torch.float32)

        depth_mask_loss = (target_mask[mid]*(target_depth+1) - curr_mask*(curr_depth+1)).abs().mean() * 10
        
        if scene_json['obj_init'][mid]['is_on_floor']:
            depth_loss += (trans_vec_list[mid][0].abs() + trans_vec_list[mid][2].abs()) / 2
            if 'fix_pos' in scene_json['obj_init'][mid].keys():
                depth_loss *= 1000
        elif mid not in scene_graph.keys() and not scene_json['obj_init'][mid]['is_on_floor']:
            depth_loss += trans_vec_list[mid].abs().sum() / 3

        if 'fix' in scene_json['obj_init'][mid].keys():
            depth_loss += (rot_vec_list[mid]-float(scene_json['obj_init'][mid]['fix'])).abs().sum()

        if scene_json['obj_init'][mid]['is_on_floor']:
            depth_loss += (trans_vec_list[mid][1] + (scene_json['obj_init'][mid]['center'][1]-opt_floor_height) / scene_json['obj_init'][mid]['scale']).abs() / len(scene_json['obj_init']) * 10

        depth_loss += (opt_floor_height-scene_json['scene']['floor_height']).abs()


        loss = loss_mask + depth_loss + depth_mask_loss

        loss.backward(retain_graph=True)




        debug_loss+=loss.item()/num_obj
        debug_loss_mask+=loss_mask.item()/num_obj
        debug_depth_loss+=depth_loss.item()/num_obj

        with torch.no_grad():
            if mid in scene_graph.keys():
                parent_id = scene_graph[mid]

                gx = (sc.param_map[f"Mesh[{parent_id}]"].vertex_positions_T.x.numpy()).mean()
                gz = (sc.param_map[f"Mesh[{parent_id}]"].vertex_positions_T.z.numpy()).mean()

                tar_x = (((scene_json['obj_init'][mid]['center'][0]-gx) / (scene_json['obj_init'][mid]['scale'])))
                tar_z = (((scene_json['obj_init'][mid]['center'][2]-gz) / (scene_json['obj_init'][mid]['scale'])))

                if (trans_vec_list[mid][0] - tar_x) > thold:
                    trans_vec_list[mid][0] = tar_x + thold
                    trans_vec_list[mid].grad[0].zero_()
                elif (trans_vec_list[mid][0] - tar_x) < -thold:
                    trans_vec_list[mid][0] = tar_x - thold
                    trans_vec_list[mid].grad[0].zero_()

                if (trans_vec_list[mid][2] - tar_z) > thold:
                    trans_vec_list[mid][2] = tar_z + thold
                    trans_vec_list[mid].grad[2].zero_()
                elif (trans_vec_list[mid][2] - tar_z) < -thold:
                    trans_vec_list[mid][2] = tar_z - thold
                    trans_vec_list[mid].grad[2].zero_()

                graph_height = max(sc.param_map[f"Mesh[{parent_id}]"].vertex_positions_T.y.numpy())
                trans_vec_list[mid][1] = (-((scene_json['obj_init'][mid]['center'][1]-graph_height) / (scene_json['obj_init'][mid]['scale']))) + 0.001
                trans_vec_list[mid].grad[1].zero_()

            valid_mask_all = torch.logical_and(target_mask[mid]>=1, curr_mask>=1)
            if (valid_mask_all).sum() * 10 < (target_mask[mid]>=1).sum() or curr_mask.sum() * 3 < (target_mask[mid]>=1).sum():
                scale_vec_list[mid] += 0.05
                scale_vec_list[mid].grad.zero_()


            if curr_mask.sum() > (target_mask[mid]>=1).sum() * 1.5:
                scale_vec_list[mid] -= 0.05
                scale_vec_list[mid].grad.zero_()




        if mid == num_obj-1:


            optimizer.step()
            optimizer.zero_grad()




            
            ckp = {}
            ckp['floor_height'] = float(opt_floor_height.item())
            ckp['to_world'] = []
            ckp['to_world_right'] = []

            for oid in range(num_obj):
                ckp['to_world'].append(sc.param_map[f"Mesh[{oid}]"].to_world.numpy().tolist())
                ckp['to_world_right'].append(sc.param_map[f"Mesh[{oid}]"].to_world_right.numpy().tolist())


            if it % save_iter == 0:
                json_object = json.dumps(ckp, indent=4)
                with open(str(obj_stage_path / f"ckp_{it}.json"), "w") as outfile:
                    outfile.write(json_object)

                to_world_list = []
                for oid, obj_data in enumerate(scene_json['obj_init']):
                    to_world_tmp = torch.mm(translate(trans_vec_list[oid]), scale(scale_vec_list[oid]))
                    to_world_tmp = torch.mm(to_world_tmp, rotate(torch.tensor([0,1,0], device='cuda', dtype=torch.float32), rot_vec_list[oid]))
                    to_world_list.append(to_world_tmp)

                curr_depth = psdr_render(psdr.FieldExtractionIntegrator("depth"), sc, -1, psdr_opt_dict, to_world_list)

                curr_depth = curr_depth.detach().cpu().numpy()
                curr_depth /= curr_depth.max()
                heatmap = cv2.applyColorMap((curr_depth*255).astype(np.uint8), cv2.COLORMAP_HOT)

                cv2.imwrite(str(obj_stage_path / f"iter_{it}.jpg"), heatmap)


                if it % mesh_save_iter == 0 or it == 0:
                    dump_path = obj_stage_path / str(it)
                    dump_path.mkdir(parents=True, exist_ok=True)
                    for oid in range(num_obj):
                        sc.param_map[f"Mesh[{oid}]"].dump(str(dump_path/f"{oid}.obj"), True)


            t1 = time()

            print(f"Iter {it}", "Loss", round(debug_loss, 5), round(debug_loss_mask, 5), round(debug_depth_loss, 5), "opt_floor_height", round(opt_floor_height.item(), 5), "Time", round(t1 - t0, 5))
            debug_loss = 0.
            debug_loss_mask = 0.
            debug_depth_loss = 0.


        print(f"Iter {it}/{mid}", "Loss", round(loss.item(), 5), round(loss_mask.item(), 5), round(depth_loss.item(), 5), end='\r', flush=True)
        del loss, curr_mask
        # exit()
    scene_json['scene']['floor_height_opt'] = float(opt_floor_height.item())
    json_object = json.dumps(scene_json, indent=4)
    with open(str(obj_stage_init_path / "obj_stage_config.json"), "w") as outfile:
        outfile.write(json_object)




