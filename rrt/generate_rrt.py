import numpy as np
import copy 
import torch
import scipy.spatial.transform
import math
import matplotlib.pyplot as plt

from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.utils import to_torch, to_numpy
from se3dif.samplers import Grasp_AnnealedLD


def get_approximated_grasp_diffusion_field(p, model_, device='cpu'):
    batch = 10
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_ # grasp_dif_mugs 
    }
    # {'device': 'cpu', 'pretrained_model': 'grasp_dif_mugs', 'NetworkArch': 'PointcloudGraspDiffusion', 'NetworkSpecs': {'feature_encoder': {'enc_dim': 132, 'in_dim': 3, 'out_dim': 7, 'dims': [512, 512, 512, 512, 512, 512, 512, 512], 'dropout': [0, 1, 2, 3, 4, 5, 6, 7], 'dropout_prob': 0.2, 'norm_layers': [0, 1, 2, 3, 4, 5, 6, 7], 'latent_in': [4], 'xyz_in_all': False, 'use_tanh': False, 'latent_dropout': False, 'weight_norm': True}, 'encoder': {'latent_size': 132, 'hidden_dim': 512}, 'points': {'n_points': 30, 'loc': [0.0, 0.0, 0.5], 'scale': [0.7, 0.5, 0.7]}, 'decoder': {'hidden_dim': 512}}}
    model = load_model(model_args)
    

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model

def sample_pointcloud(obj_id=0, obj_class='Mug'):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class) # grasp
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()  # mesh 

    P = mesh.sample(1000)  # (1000,3) sample points on surface

    sampled_rot = scipy.spatial.transform.Rotation.random()
    rot = sampled_rot.as_matrix()  # (3,3)
    rot_quat = sampled_rot.as_quat()  # (4,1)

    P = np.einsum('mn,bn->bm', rot, P) #(1000,3) # apply random rotation 
    P *= 8.  
    P_mean = np.mean(P, 0)  # (1,3)
    P += -P_mean  # adjust scale, location 

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    translational_shift = copy.deepcopy(H)

    return P, mesh, translational_shift, rot_quat 
        # (1000,3), (verticles:(1167,3),faces:(2334,3)) (4,4) (4,)
        # rotated, tranlated, scaled mesh, and P // H only translation


