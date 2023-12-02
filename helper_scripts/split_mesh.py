import os
import glob
import igl
from utils.io import *
import argparse
from pathlib import Path
def get_parser():
    parser = argparse.ArgumentParser(description="scene initalization")
    parser.add_argument(
        "scene",
        default="AdobeStock_5889655",
        help="scene to run",
    )
    return parser

def split_obj_mesh(input_obj_file, output_obj_path, obj_name, raw_obj_path, scene_name):
    with open(input_obj_file, "r") as obj_file:
        obj_content = obj_file.read().splitlines()

    current_material_group = None
    material_groups = {}
    vertices = []
    texture_coords = []
    normals = []

    for line in obj_content:
        if line.startswith("o") or line.startswith("g"):
            if current_material_group:
                material_groups[current_material_group] = current_content

            current_material_group = line.strip().split(maxsplit=1)[1]
            current_content = []

        elif line.startswith("v "):
            vertices.append(line)
        elif line.startswith("vt"):
            texture_coords.append(line)
        elif line.startswith("vn"):
            normals.append(line)
        else:
            if current_material_group:
                current_content.append(line)

    if current_material_group:
        material_groups[current_material_group] = current_content

    group_count = 0
    # print(obj_name)

    v_model, _, _, f_model, _, _ = igl.read_obj(output_obj_path+f"/{obj_name}.obj")
    n_mesh_min = v_model[:,1].min()

    for group_name, group_content in material_groups.items():
        output_filename = output_obj_path+f"/{obj_name}_{group_count}.obj"
        with open(output_filename, "w") as output_file:
            output_file.write("o {}\n".format(group_name))
            output_file.write("\n".join(vertices) + "\n")
            output_file.write("\n".join(texture_coords) + "\n")
            output_file.write("\n".join(normals) + "\n")
            output_file.write("\n".join(group_content) + "\n")

        v_model, _, _, f_model, _, _ = igl.read_obj(input_obj_file)
        v_modelrr, _, _, f_modelrr, _, _ = igl.read_obj(raw_obj_path+f"/../../obj_stage_init/{obj_name}.obj")
        vv_min = v_modelrr[:,1].min()

        # input_obj_file = raw_obj_path + f"/../obj_stage_init/{obj_name}.obj"

        v_group, tc, _, f_group, ftc, _ = igl.read_obj(output_filename)

        cx = (v_model[:,0].min() + v_model[:,0].max()) / 2
        cy = (v_model[:,1].min() + v_model[:,1].max()) / 2
        cz = (v_model[:,2].min() + v_model[:,2].max()) / 2
        v_group[:,0] -= cx
        v_group[:,1] -= v_model[:,1].min()
        # v_group[:,1] += n_mesh_min
        v_group[:,2] -= cz
        dsize =  max(v_model[:,0].max()-v_model[:,0].min(), 
                     v_model[:,1].max()-v_model[:,1].min(), 
                     v_model[:,2].max()-v_model[:,2].min())

        v_group /= dsize

        v_group[:,1] += vv_min

        out_path = Path("split_mesh_output", scene_name)
        out_path.mkdir(parents=True, exist_ok=True)

        write_obj(f"./split_mesh_output/{scene_name}/{obj_name}_{group_count}.obj", v_group, f_group, tc=tc, ftc=ftc)

        group_count += 1

if __name__ == "__main__":
    args = get_parser().parse_args()

    scene_name = args.scene

    for raw_obj_path in glob.glob(f"./split_mesh_example/{scene_name}/obj_raw/**"):

        raw_obj_path = raw_obj_path.replace('\\','/')
        input_obj_file = raw_obj_path + "/normalized_model.obj"
        output_obj_path = f"./split_mesh_example/{scene_name}/obj_stage_init/"
        obj_name = raw_obj_path.split("/")[-1].split('.')[0]

        # print(obj_name)
        # exit()
        split_obj_mesh(input_obj_file, output_obj_path, obj_name, raw_obj_path, scene_name)
