import torch 

def normalize(vector):
    return vector / torch.norm(vector)

def lookat(origin, target, up):
    device = origin.device
    dtype = origin.dtype 

    origin = origin
    dir = normalize(target - origin)
    left = normalize(torch.cross(up, dir))
    new_up = normalize(torch.cross(dir, left))

    to_world = torch.eye(4).to(device).to(dtype)
    to_world[:3, 0] = left
    to_world[:3, 1] = new_up
    to_world[:3, 2] = dir
    to_world[:3, 3] = origin

    return to_world


def translate(t_vec):
    if not torch.is_tensor(t_vec):
        t_vec = torch.tensor(t_vec, dtype=torch.float)
    device = t_vec.device
    dtype = t_vec.dtype 

    to_world = torch.eye(4).to(device).to(dtype)
    to_world[:3, 3] = t_vec

    return to_world

def rotate(axis, angle, use_degree=True):
    if not torch.is_tensor(axis):
        axis = torch.tensor(axis, dtype=torch.float)
    device = axis.device
    dtype = axis.dtype 

    to_world = torch.eye(4).to(device).to(dtype)
    axis = normalize(axis).reshape(3, 1)
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle).to(device).to(dtype)
    if use_degree:
        angle = torch.deg2rad(angle)

    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    cpm = torch.zeros((3, 3)).to(device).to(dtype)
    cpm[0, 1] = -axis[2]
    cpm[0, 2] =  axis[1]
    cpm[1, 0] =  axis[2]
    cpm[1, 2] = -axis[0]
    cpm[2, 0] = -axis[1]
    cpm[2, 1] =  axis[0]

    R = cos_theta * torch.eye(3).to(device).to(dtype)
    R += sin_theta * cpm
    R += (1 - cos_theta) * (axis @ axis.T)

    to_world[:3, :3] = R

    return to_world

def rotate3D(anglex, angley, anglez, device='cuda'):
    rot_axisx = torch.tensor([1,0,0], device=device, dtype=torch.float32)
    rot_axisy = torch.tensor([0,1,0], device=device, dtype=torch.float32)
    rot_axisz = torch.tensor([0,0,1], device=device, dtype=torch.float32)

    rotmatx = rotate(rot_axisx, anglex)
    rotmaty = rotate(rot_axisy, angley)
    rotmatz = rotate(rot_axisz, anglez)

    rotmat_3d = torch.mm(rotmatx, rotmaty)
    rotmat_3d = torch.mm(rotmat_3d, rotmatz)
    return rotmat_3d


def scale(size):
    if not torch.is_tensor(size):
        size = torch.tensor(size, dtype=torch.float)
    device = size.device
    dtype = size.dtype 

    to_world = torch.eye(4).to(device).to(dtype)

    if size.size() == () or size.size(dim=0) == 1:
        to_world[:3, :3] = torch.eye(3).to(device).to(dtype) * size
    elif size.size(dim=0) == 3:
        to_world[:3, :3] = torch.diag(size).to(device).to(dtype)
    else:
        print("error transform.py for scale")
        exit()

    return to_world

# texture map transform (2d)
def translate2D(t_vec):
    device = t_vec.device
    dtype = t_vec.dtype 

    to_world = torch.eye(3).to(device).to(dtype)
    to_world[:2, 2] = t_vec

    return to_world

def rotate2D(angle, use_degree=True):
    device = angle.device
    dtype = angle.dtype 

    to_world = torch.eye(3).to(device).to(dtype)
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle).to(device).to(dtype)
    if use_degree:
        angle = torch.deg2rad(angle)

    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    R = cos_theta * torch.eye(2).to(device).to(dtype)

    R[0 ,1] = -sin_theta
    R[1 ,0] = sin_theta

    to_world[:2, :2] = R

    return to_world

def scale2D(size):
    device = size.device
    dtype = size.dtype 

    to_world = torch.eye(3).to(device).to(dtype)

    if size.size(dim=0) == 1:
        to_world[:2, :2] = torch.diag(size).to(device).to(dtype) * torch.eye(2).to(device).to(dtype)
    elif size.size(dim=0) == 2:
        to_world[:2, :2] = torch.diag(size).to(device).to(dtype)
    else:
        print("error transform.py for scale")
        exit()

    return to_world