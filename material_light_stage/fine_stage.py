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

from PIL import Image
from utils.loss import *

# from utils.clip import *
# from skimage.measure import regionprops
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import torch
import drjit
import psdr_jit as psdr
from drjit.cuda.ad import Float as FloatD, Array3f as Vector3fD, Matrix4f as Matrix4fD
from drjit.cuda import Float as FloatC, Array3f as Vector3fC, Matrix4f as Matrix4fC
from torch.optim import Adam
from time import time
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np 


from diffmat import MaterialGraphTranslator as MGT
from diffmat.optim import TextureDescriptor
import config

defaultconfig = config.FineStageConfig()

num_iter = defaultconfig.num_iter
mgt_res = defaultconfig.mgt_res


def get_parser():
    parser = argparse.ArgumentParser(description="scene initalization")
    parser.add_argument(
        "scene",
        default="AdobeStock_5889655",
        help="scene to run",
    )
    # parser.add_argument(
    #     "manual",
    #     default=0,
    #     help="scene to run",
    # )
    return parser

forward_npass = defaultconfig.forward_npass
backward_npass = defaultconfig.backward_npass

transformVGG = T.Resize(size=(defaultconfig.vggresizeres,defaultconfig.vggresizeres), interpolation=T.InterpolationMode.BILINEAR, max_size=None, antialias=None)


class RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, integrator, sc, param_list, *param):
        if len(param_list) != len(param):
            print(param_list)
            print(param)
            print("error: param size mismatch")
            exit()

        for pstr, pid in zip(param_list, param):
            # print("pstr", pstr)
            # print("pid", pid)
            if pstr['type'] == "BSDF":
                if pstr['BSDF_type'] == "Microfacet":
                    if pstr['param'] == "diffuseReflectance":
                        try:
                            sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.diffuseReflectance.data = Vector3fD(pid.cpu().numpy())
                        except:
                            sc.param_map[f"BSDF[id={pstr['name']}]"].diffuseReflectance.data = Vector3fD(pid.cpu().numpy())
                    elif pstr['param'] == "roughness":
                        try:
                            sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.roughness.data = FloatD(pid.cpu().numpy())
                        except:
                            sc.param_map[f"BSDF[id={pstr['name']}]"].roughness.data = FloatD(pid.cpu().numpy())
                    elif pstr['param'] == "NormalMap":
                        sc.param_map[f"BSDF[id={pstr['name']}]"].normal_map.data = Vector3fD(pid.cpu().numpy())
                    elif pstr['param'] == "rotate":
                        sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.diffuseReflectance.rotate = FloatD(pid)
                        sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.roughness.rotate = FloatD(pid)
                        sc.param_map[f"BSDF[id={pstr['name']}]"].normal_map.rotate = FloatD(pid)
                        # sc.param_map[f"BSDF[id={pstr['name']}]"].roughness.rotate = FloatD(pid)
                    elif pstr['param'] == "scale":
                        sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.diffuseReflectance.scale = FloatD(pid)
                        sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.roughness.scale = FloatD(pid)
                        sc.param_map[f"BSDF[id={pstr['name']}]"].normal_map.scale = FloatD(pid)

                        # sc.param_map[f"BSDF[id={pstr['name']}]"].roughness.rotate = FloatD(pid)

            elif pstr['type'] == "Emitter":
                if pstr['param'] == "radiance":
                    print(sc.param_map[f"Emitter[{pstr['id']}]"])
                    sc.param_map[f"Emitter[{pstr['id']}]"].radiance = Vector3fD([torch.exp(pid).cpu().numpy()])
        sc.configure()

        for ii in range(0, forward_npass):
            if ii == 0:
                psdr_image = integrator.renderC(sc, 0)#psdr_image.torch().to('cuda').to(torch.float32)
            else:
                psdr_image += integrator.renderC(sc, 0)
                drjit.eval(psdr_image)
        image = psdr_image.torch()
        image /= forward_npass
        ctx.scene = sc
        ctx.integrator = integrator
        ctx.param_list = param_list
        ctx.param = param
        del psdr_image
        return image.reshape((sc.opts.height, sc.opts.width, 3))

    @staticmethod
    def backward(ctx, grad_out):
        drjit_param = []
        for pstr, pid in zip(ctx.param_list, ctx.param):

            if pstr['type'] == "BSDF":
                if pstr['BSDF_type'] == "Microfacet":
                    if pstr['param'] == "diffuseReflectance":
                        try:
                            drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.diffuseReflectance.data)
                        except:
                            drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].diffuseReflectance.data)
                    elif pstr['param'] == "roughness":
                        try:
                            drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.roughness.data)
                        except:
                            drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].roughness.data)
                    elif pstr['param'] == "NormalMap":
                        drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].normal_map.data)
                    elif pstr['param'] == "rotate":
                        drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.diffuseReflectance.rotate)
                    elif pstr['param'] == "scale":
                        drjit_param.append(sc.param_map[f"BSDF[id={pstr['name']}]"].nested_bsdf.diffuseReflectance.scale)
            elif pstr['type'] == "Emitter":
                if pstr['param'] == "radiance":
                    drjit_param.append(sc.param_map[f"Emitter[{pstr['id']}]"].radiance)

            drjit.enable_grad(drjit_param[-1])
        ctx.scene.configure()
        image_grad = Vector3fC(grad_out.reshape(-1, 3))

        param_grad = []
        grad_tmp = []
        for ii in range(0, backward_npass):
            imaged = psdr.PathTracer(1).renderD(ctx.scene, 0)
            tmp = drjit.dot(image_grad, imaged)
            drjit.backward(tmp)
            drjit.eval(imaged, tmp)
            if ii == 0:
                for dpar in range(0, len(ctx.param)):
                    drjit.eval(drjit_param[dpar])
                    # print(drjit.grad(drjit_param[dpar]))
                    grad_tmp.append(drjit.grad(drjit_param[dpar]) / backward_npass)
            else:
                for dpar in range(0, len(ctx.param)):
                    drjit.eval(drjit_param[dpar])
                    grad_tmp[dpar] += drjit.grad(drjit_param[dpar]) / backward_npass

            del tmp, imaged
        del drjit_param

        for gtmp in grad_tmp:
            param_grad.append(torch.nan_to_num(gtmp.torch().cuda()))
        return tuple([None] * 3 + param_grad)


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, integrator, scene, param_list=[], params=[]):
        image = RenderFunction.apply(integrator, scene, param_list, *params)
        return image

if __name__ == "__main__":
    args = get_parser().parse_args()

    data_path = Path("input", args.scene)

    is_manual = 0#int(args.manual)

    tex_stage_init_path = data_path / "texture_stage_init"

    debug_path = data_path / "debug"

    # if is_manual==1:
    #     fine_stage_path = data_path / "fine_stage_manual"
    # else:
    fine_stage_path = Path("output", args.scene, "fine_stage")

    fine_stage_path.mkdir(parents=True, exist_ok=True)
    crop_path = tex_stage_init_path / "target_crop"

    with open(str(data_path / "scale.json"), "r") as f:
        scale_json = json.load(f)

    # # Specify SAT location and output folders
    toolkit_path = Path(defaultconfig.toolkit_path)
    external_input_path = Path('./external_input')
    sbs_source = Path('./sbs_material')
    sampler_path = Path('./sampler/')

    load_ckp = -1

    if len(glob.glob(str(fine_stage_path)+f"/iter_*.pt")) > 0 and load_ckp == -1:
        load_ckp = len(glob.glob(str(fine_stage_path)+f"/iter_*.pt"))-1
    print("Load ckp:", load_ckp)
    if load_ckp >= num_iter-1:
        exit()

    # if is_manual==1:
    #     with open(str(tex_stage_init_path / "text_stage_config_manual.json"), "r") as f:
    #         scene_json = json.load(f)
    # else:
    with open(str(tex_stage_init_path / "text_stage_config_auto.json"), "r") as f:
        scene_json = json.load(f)


    target_image = torch.tensor((read_png(data_path / "image_resize.png")), device='cuda', dtype=torch.float32)
    target_image = to_linear(target_image)
    target_image = torch.tensor(target_image, device='cuda', dtype=torch.float32)


    img_shape = target_image.shape

    transform = T.Resize(size=(img_shape[1] // 8,img_shape[0] // 8), interpolation=T.InterpolationMode.BILINEAR, max_size=None, antialias=None)

    sc = psdr.Scene()
    sc.opts.spp = defaultconfig.spp # Interior Term
    sc.opts.sppe = 0 # Primary Edge
    sc.opts.sppse = 0 # Secondary Edge

    sc.opts.height = img_shape[0]
    sc.opts.width = img_shape[1]
    sc.opts.log_level = 0

    integrator = psdr.PathTracer(defaultconfig.bounce) 

    sensor = psdr.PerspectiveCamera(scene_json["cameraFoV"], 0.000001, 10000000.)
    to_world = Matrix4fD(rotate3D(-scene_json["cameraPitch"], 0, scene_json["cameraRoll"], device='cpu').numpy().astype(float).tolist())
    sensor.to_world = to_world
    sc.add_Sensor(sensor)


    sc.add_BSDF(psdr.DiffuseBSDF(0.0), "light")


    emi_opt = []
    emi_para = []

    for emi_id in scene_json['id_to_emitter'].keys():
        emi_obj = scene_json['id_to_emitter'][str(emi_id)]
        emi_opt.append(torch.tensor(scene_json['emitter'][emi_obj],device='cuda', dtype=torch.float32).requires_grad_())
        sc.add_Mesh(str(tex_stage_init_path / emi_obj), Matrix4fC(np.eye(4)), "light", psdr.AreaLight(scene_json['emitter'][emi_obj]))
        emi_para.append({"id":int(emi_id), "type":"Emitter", "param":"radiance"})


    target_crop_dict = {}

    mat_graph = []
    mat_graph_filter_id = []

    mat_norm_opt = []


    num_mesh = len(scene_json['id_to_mesh'].keys())
    for mesh_id in range(num_mesh):
        obj_name = scene_json['id_to_mesh'][str(mesh_id)]
        obj_dict = scene_json['mesh'][obj_name]
        obj_color = obj_dict['color']


        obj_roughness = obj_dict['roughness']
        obj_specular = obj_dict['specular']

        toSided = True
        auto = True

        if 'manual_crop_data' in obj_dict.keys() and 'material' in obj_dict.keys():
            if obj_dict['manual_crop_data']:
                key_list = list(obj_dict['manual_crop_data'].keys())

                for kid, keykey in enumerate(key_list):
                    if kid == 0:
                        crop_im_tar = cv2.imread(str(tex_stage_init_path / "target_crop" / (key_list[0])), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
                        crop_im_tar = cv2.cvtColor(crop_im_tar, cv2.COLOR_RGB2BGR)
                        crop_im_tar = torch.tensor(crop_im_tar, device='cuda', dtype=torch.float32)
                        target_crop_dict[str(mesh_id)] = [[crop_im_tar, obj_dict['manual_crop_data'][key_list[0]]]]
                    else:
                        crop_im_tar = cv2.imread(str(tex_stage_init_path / "target_crop" / (keykey)), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
                        crop_im_tar = cv2.cvtColor(crop_im_tar, cv2.COLOR_RGB2BGR)
                        crop_im_tar = torch.tensor(crop_im_tar, device='cuda', dtype=torch.float32)
                        target_crop_dict[str(mesh_id)].append([crop_im_tar, obj_dict['manual_crop_data'][keykey]])


        if "window" in obj_dict['cat_name'] or"tv" in obj_dict['cat_name'] or"wall" in obj_dict['cat_name'] or  "floor" in obj_dict['cat_name'] or "ceil" in obj_dict['cat_name']:
            toSided = False
        if 'emitter' in obj_dict.keys():
            toSided = False
        if "material" in obj_dict.keys() and  'emitter' not in obj_dict.keys():
            print(str(obj_dict["material"]))
            sbs_file_path = sbs_source / str(obj_dict["material"])
            # print(os.path.exists(sbs_file_path))
            if not os.path.exists(sbs_file_path):
                sbs_file_path = sbs_source / ("matchv1_"+str(obj_dict["material"]))

            translator = MGT(sbs_file_path, res=mgt_res, toolkit_path=toolkit_path)
            graph = translator.translate(seed=0, external_input_folder=external_input_path, device='cuda')

            graph.compile()
            graph.train()

            print(str(sampler_path / obj_dict["material"][0:-4] / "**" / "**" / "param" / f"params_{obj_dict['material_var']}.pth"))


            find_ckp = glob.glob(str(sampler_path / obj_dict["material"][0:-4] / "**" / "**" / "param" / f"params_{obj_dict['material_var']}.pth"))
            # print(find_ckp)
            if len(find_ckp) == 0:
                find_ckp = glob.glob(str(sampler_path / ("matchv1_"+obj_dict["material"][0:-4]) / "**" / "**" / "param" / f"params_{obj_dict['material_var']}.pth"))
            print(find_ckp)

            find_ckp = find_ckp[0]
            ckp_data = torch.load(find_ckp)

            graph.set_parameters_from_tensor(ckp_data['param'])

            eval_graph = graph.evaluate_maps()

            normalmap = torch.nn.functional.normalize(eval_graph[1][:,0:3,:,:], dim=1).permute(0,2,3,1)[0].reshape(-1, 3)
            sc.add_normalmap_BSDF(psdr.NormalMapBSDF([0.4999,0.4999,0.70724817426]), psdr.MicrofacetBSDF(obj_specular, obj_color, obj_roughness), obj_name, toSided)


            sc.param_map[f"BSDF[id={obj_name}]"].nested_bsdf.diffuseReflectance.resolution = [2**mgt_res,2**mgt_res]
            sc.param_map[f"BSDF[id={obj_name}]"].nested_bsdf.roughness.resolution = [2**mgt_res,2**mgt_res]
            sc.param_map[f"BSDF[id={obj_name}]"].normal_map.resolution = [2**mgt_res,2**mgt_res]


            drjit.make_opaque(sc.param_map[f"BSDF[id={obj_name}]"].nested_bsdf.diffuseReflectance.data)
            drjit.make_opaque(sc.param_map[f"BSDF[id={obj_name}]"].nested_bsdf.diffuseReflectance.scale)
            drjit.make_opaque(sc.param_map[f"BSDF[id={obj_name}]"].nested_bsdf.diffuseReflectance.rotate)


            if "filter" in obj_dict.keys():
                mat_graph_filter_id.append(len(mat_graph))
            mat_graph.append(graph)


            mat_norm_opt.append(torch.tensor([0.,0.,0.], device='cuda', dtype=torch.float32).requires_grad_())
        else:
            sc.add_BSDF(psdr.MicrofacetBSDF(obj_specular, obj_color, obj_roughness), obj_name, toSided)


        if 'emitter' in obj_dict.keys():
            sc.add_Mesh(str(tex_stage_init_path / obj_name), Matrix4fC(np.eye(4)), obj_name, psdr.AreaLight(obj_dict['emitter']))

            emi_para.append({"id":len(emi_opt), "type":"Emitter", "param":"radiance"})
            emi_opt.append(torch.tensor(obj_dict['emitter'],device='cuda', dtype=torch.float32).requires_grad_())
        else:
            scale_val = 1.0
            if scale_json != {}:
                if obj_name[0] in scale_json.keys():
                    scale_val = scale_json[obj_name[0]]

            toworld = np.eye(4)
            toworld[:3, :3]*=scale_val

            sc.add_Mesh(str(tex_stage_init_path / obj_name), Matrix4fC(toworld), obj_name, None)
        sc.param_map[f"Mesh[{sc.num_meshes-1}]"].use_face_normal = True

    def all_params():
        for mid, in_graph in enumerate(mat_graph):
            if mid in mat_graph_filter_id:
                yield from in_graph.parameters(filter_generator=1)
            else:
                yield from in_graph.parameters(filter_generator=0)

    optimizer = None
    if mat_graph:
        optimizer = Adam(all_params(), lr=0.002)
    else:
        print("WARN: no material")

    color_opt = []

    diffuse_para = []
    roughness_para = []
    normalmap_para = []

    scale_para=[]
    rotate_para=[]
    scale_opt=[]
    rotate_opt=[]

    for mesh_id in range(num_mesh):

        obj_name = scene_json['id_to_mesh'][str(mesh_id)]
        obj_dict = scene_json['mesh'][obj_name]
        obj_color = obj_dict['color']

        if obj_dict['visible'] == True and 'emitter' not in obj_dict.keys():
            if 'material' not in obj_dict.keys() and 'emitter' not in obj_dict.keys():
                obj_color = torch.tensor(obj_color,device='cuda', dtype=torch.float32)
                obj_color = obj_color.clamp(0.0001, 0.9999)
                obj_color = torch.log(obj_color/(1-obj_color))
                color_opt.append(obj_color.requires_grad_())

            

            if 'material' in obj_dict.keys() :
                scale_opt.append(torch.tensor([obj_dict['scale']],device='cuda', dtype=torch.float32).requires_grad_())
                scale_para.append({"name": obj_name, "id":mesh_id, "type":"BSDF", "BSDF_type": "Microfacet", "param":"scale"})
                rotate_opt.append(torch.tensor([np.deg2rad(obj_dict['rotate'])],device='cuda', dtype=torch.float32).requires_grad_())
                rotate_para.append({"name": obj_name, "id":mesh_id, "type":"BSDF", "BSDF_type": "Microfacet", "param":"rotate"})
                roughness_para.append({"name": obj_name, "id":mesh_id, "type":"BSDF", "BSDF_type": "Microfacet", "param":"roughness"})
                normalmap_para.append({"name": obj_name, "id":mesh_id, "type":"BSDF", "BSDF_type": "Microfacet", "param":"NormalMap"})

            diffuse_para.append({"name": obj_name, "id":mesh_id, "type":"BSDF", "BSDF_type": "Microfacet", "param":"diffuseReflectance"}) 
    psdr_render = Renderer()
    write_jpg(fine_stage_path / f"target.jpg", target_image)

    per_obj_mask = []

    for mesh_id in range(num_mesh):
        im = cv2.imread(str(tex_stage_init_path/"per_obj_mask"/f"{mesh_id}.jpg"), cv2.IMREAD_UNCHANGED)
        # im /= 255
        im[im < 250] = 0
        im[im > 0] = 1
        im = torch.tensor(im, device='cuda',dtype=torch.float32)
        per_obj_mask.append(im)
        # print(im)
        # exit()
        # per_obj_mask.append(str(tex_stage_init_path/"per_obj_mask"/f"{mesh_id}.jpg"))


    emitter_mask = torch.tensor(np.zeros(per_obj_mask[0].shape)+1, device='cuda', dtype=torch.float32)

    for mesh_id in range(num_mesh):
        obj_name = scene_json['id_to_mesh'][str(mesh_id)]
        obj_dict = scene_json['mesh'][obj_name]

        if 'emitter' in obj_dict.keys():
            emitter_mask[per_obj_mask[mesh_id]>=1]=0


    log_data = []


    if load_ckp > 0:
        ckp_data = torch.load(str(fine_stage_path)+f"/iter_{load_ckp}.pt")

        gcount = 0
        for mat_g in mat_graph:
            mat_g.set_parameters_from_tensor(ckp_data["mat_ckp"][gcount].to("cuda"))
            gcount += 1
        color_opt = ckp_data["color_opt"]
        emi_opt = ckp_data["emi_opt"]
        scale_opt = ckp_data["scale_opt"]
        rotate_opt = ckp_data["rotate_opt"]
        mat_norm_opt = ckp_data["mat_norm_opt"]
        log_data = ckp_data["log_data"]


        
    if load_ckp < 0:
        load_ckp = 0

    metric_func = TextureDescriptor(device='cuda').evaluate

    for color in color_opt:
        if not optimizer:
            optimizer = Adam([{'params': color, "lr": defaultconfig.lr_color}])
        else:
            optimizer.add_param_group({'params': color, "lr": defaultconfig.lr_color})
    
    for eid, emi in enumerate(emi_opt):
        optimizer.add_param_group({'params': emi, "lr": defaultconfig.lr_emi})
    for scal in scale_opt:
        optimizer.add_param_group({'params': scal, "lr": defaultconfig.lr_scale})
    for rot in rotate_opt:
        optimizer.add_param_group({'params': rot, "lr": defaultconfig.lr_rot})
    for mnorm in mat_norm_opt:
        optimizer.add_param_group({'params': mnorm, "lr": defaultconfig.mat_norm_opt})

    if load_ckp > 0:
        optimizer.load_state_dict(ckp_data["optimizer"])



    for it in range(load_ckp, num_iter):
        t0 = time()
        try:

            mat_ckp = []
            for mat_g in  mat_graph:
                # print(mat_g.get_parameters_as_tensor().cpu())
                mat_ckp.append(mat_g.get_parameters_as_tensor().cpu())

            state = {
                'iter': it,
                "mat_ckp" : mat_ckp,
                "color_opt" : color_opt,
                "emi_opt" : emi_opt,
                "scale_opt" : scale_opt,
                "rotate_opt" : rotate_opt,
                "mat_norm_opt" : mat_norm_opt,
                "log_data" : log_data,
                "optimizer" : optimizer.state_dict()
            }

            torch.save(state, str(fine_stage_path)+f"/iter_{it}.pt")


            optimizer.zero_grad()

            diffuse_map = []
            roughness_map = []
            emitter_rad = []
            normal_map = []

            material_count = 0
            color_count = 0

            fix_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda')

            for mesh_id in range(num_mesh):

                obj_name = scene_json['id_to_mesh'][str(mesh_id)]
                obj_dict = scene_json['mesh'][obj_name]
                obj_color = obj_dict['color']
                if obj_dict['visible'] == True and 'emitter' not in obj_dict.keys():
                    if 'material' in obj_dict.keys():
                        eval_graph = mat_graph[material_count].evaluate_maps()
                        diffuse = (((eval_graph[0][:,0:3,:,:].permute(0,2,3,1)+0.055)/1.055) ** 2.4)[0].reshape(-1, 3)
                        diffuse = diffuse * torch.sigmoid(mat_norm_opt[material_count]) * 2.0
                        diffuse_map.append(diffuse)
                        normal = torch.nn.functional.normalize(eval_graph[1][:,0:3,:,:], dim=1).permute(0,2,3,1)[0].reshape(-1, 3)
                        normal_map.append(normal)

                        roughness = eval_graph[2][:,0:1,:,:][0][0].reshape(-1)
                        if "fix_roughness" in obj_dict.keys():
                            fix_loss += (roughness.mean() - obj_dict["fix_roughness"]).abs() * 0.1

                        if "fix_scale" in obj_dict.keys():
                            # print(scale_opt[material_count], torch.tensor([obj_dict["fix_scale"]], device='cuda', dtype=torch.float32))
                            fix_loss += (scale_opt[material_count][0] - obj_dict["fix_scale"]).abs() * 0.1
                            print("scel fix", fix_loss)

                        roughness_map.append(roughness)
                        material_count += 1
                    else:
                        diffuse_map.append(torch.sigmoid(color_opt[color_count]))
                        color_count += 1
            render = psdr_render(integrator, sc, diffuse_para+roughness_para+normalmap_para+emi_para+scale_para+rotate_para, diffuse_map+roughness_map+normal_map+emi_opt+scale_opt+rotate_opt)

            img_loss = (transform((target_image*emitter_mask).permute(2,1,0)) - transform((render*emitter_mask).permute(2,1,0))).abs().mean()
            vgg_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda')
            color_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda')
            for mesh_id in range(num_mesh):
                obj_name = scene_json['id_to_mesh'][str(mesh_id)]
                obj_dict = scene_json['mesh'][obj_name]
                obj_color = obj_dict['color']

                if str(mesh_id) in target_crop_dict.keys():
                    for ii_crop in range(0, len(target_crop_dict[str(mesh_id)])):
                        crop_pos = target_crop_dict[str(mesh_id)][ii_crop][1]
                        target_crop_buf = target_crop_dict[str(mesh_id)][ii_crop][0]
                        opt_crop_buf = render[crop_pos[0]:crop_pos[1], crop_pos[2]:crop_pos[3]]

                        if 'material' in obj_dict.keys():
                            target_crop = transformVGG((target_crop_buf).permute(2,1,0)).unsqueeze(0)
                            opt_crop = transformVGG((opt_crop_buf).permute(2,1,0)).unsqueeze(0)
                            vgg_loss += (metric_func(target_crop)-metric_func(opt_crop)).abs().mean()*0.1

                if obj_dict['visible'] == True:
                    if 'emitter' not in obj_dict.keys():
                        color_loss += (render.reshape(-1, 3)[per_obj_mask[mesh_id].reshape(-1, 3)>=1].reshape(-1, 3).mean(axis=0) - torch.tensor(obj_color, device='cuda', dtype=torch.float32)).abs().mean() * 10.

            vgg_loss /= num_mesh
            color_loss /= num_mesh


            loss = img_loss + vgg_loss + color_loss + fix_loss

            loss.backward()
            optimizer.step()


            t1 = time()

            print(f"[iter {it}/{num_iter}] loss: {loss.item()} img_loss: {img_loss.item()} vgg_loss: {vgg_loss.item()} color_loss: {color_loss.item()}  time: {t1 - t0}")

            log_data.append(f"[iter {it}/{num_iter}] loss: {loss.item()} img_loss: {img_loss.item()} vgg_loss: {vgg_loss.item()} color_loss: {color_loss.item()}  time: {t1 - t0}")

            write_jpg(fine_stage_path / f"iter_{it}.jpg", render)

            del loss, render
        except:
            print("error, the system can crash sometimes, you can simply rerun using fine_stage.py xx to load pervious checkpoint")
            exit()
 


