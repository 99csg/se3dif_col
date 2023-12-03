# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']
import copy
import configargparse
import scipy.spatial.transform
from scipy.spatial.transform import Rotation 
import numpy as np
import sys
sys.path.append("../../")
from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
from se3dif.rrt.pybullet_vis import PandaVisualization
from se3dif.rrt.ik_ex import KukaVisualization
from se3dif.rrt.q_based_grasp_samplers import Q_based_Grasp_AnnealedLD
from se3dif.rrt.joint_based_grasp_samplers import Joint_based_Grasp_AnnealedLD
from se3dif.rrt.rrt_based_grasp_samplers import rrt_Grasp_AnnealedLD

import torch
import pybullet as p
from se3dif.visualization import grasp_visualization
import matplotlib.pyplot as plt


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=0)
    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--obj_class', type=str, default='Mug')
    p.add_argument('--device', type=str, default='cpu') # cuda:0
    p.add_argument('--eval_sim', type=bool, default=False)
    p.add_argument('--model', type=str, default='grasp_dif_mugs') # grasp_dif_multi 


    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, args, device='cpu'):
    model_params = args.model  # grasp_dif_mugs 
    batch = 10
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    # {'device': 'cpu', 'pretrained_model': 'grasp_dif_mugs', 'NetworkArch': 'PointcloudGraspDiffusion', 'NetworkSpecs': {'feature_encoder': {'enc_dim': 132, 'in_dim': 3, 'out_dim': 7, 'dims': [512, 512, 512, 512, 512, 512, 512, 512], 'dropout': [0, 1, 2, 3, 4, 5, 6, 7], 'dropout_prob': 0.2, 'norm_layers': [0, 1, 2, 3, 4, 5, 6, 7], 'latent_in': [4], 'xyz_in_all': False, 'use_tanh': False, 'latent_dropout': False, 'weight_norm': True}, 'encoder': {'latent_size': 132, 'hidden_dim': 512}, 'points': {'n_points': 30, 'loc': [0.0, 0.0, 0.5], 'scale': [0.7, 0.5, 0.7]}, 'decoder': {'hidden_dim': 512}}}
    model = load_model(model_args)
    

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    #generator = Joint_based_Grasp_AnnealedLD(model, batch=batch, T=25, T_fit=5, k_steps=2, device=device)
    generator = Q_based_Grasp_AnnealedLD(model, batch=batch, T=50, T_fit=5, k_steps=2, device=device)
    #generator = rrt_Grasp_AnnealedLD(model, batch=batch, T=5, T_fit=5, k_steps=2, device=device)
    
    
    #generator = Grasp_AnnealedLD(model, batch=batch, T=20, T_fit=20, k_steps=2, device=device)

    return generator, model


def sample_pointcloud(obj_id=0, obj_class='Mug'):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class) # grasp
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()  # mesh 
    P = mesh.sample(200)  # (1000,3) sample points on surface


    #sampled_rot = scipy.spatial.transform.Rotation.random()

    x_rot = np.random.uniform(-np.pi/2,np.pi/2)
    y_rot = np.random.uniform(-np.pi/2,np.pi/2)
    z_rot = np.random.uniform(-np.pi/2,np.pi/2)
    sampled_rot = Rotation.from_euler('xyz',[0, 0,0])

    rot = sampled_rot.as_matrix()  # (3,3)
    rot_quat = sampled_rot.as_quat()  # (4,1)

    P = np.einsum('mn,bn->bm', rot, P) #(1000,3) # apply random rotation on P
    P *= 8.  # scale x8 
    P_mean = np.mean(P, 0)  # (1,3)
    # print("sample")
    # print(P_mean)
    P += -P_mean  # adjust scale, location on P - normalize 

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H) # random rotation
    mesh.apply_scale(8.) # x8
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H) # normalize
    translational_shift = copy.deepcopy(H) # only translation

    return P, mesh, translational_shift, rot_quat 
        # (1000,3), (verticles:(1167,3),faces:(2334,3)) (4,4) (4,)
        # rotated, tranlated, scaled mesh, and P // H only translation



if __name__ == '__main__':


    args = parse_args()
    EVAL_SIMULATION = args.eval_sim
    if (EVAL_SIMULATION):
        from isaac_evaluation.grasp_quality_evaluation import GraspSuccessEvaluator


    print('##########################################################')
    print('Object Class: {}'.format(args.obj_class))
    print(args.obj_id)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = 30
    device = args.device

    ## Set Model and Sample Generator ##
    P, mesh, trans, rot_quad = sample_pointcloud(obj_id, obj_class) 
    generator, model = get_approximated_grasp_diffusion_field(P, args, device)
    quat, trj_quat = generator.sample(save_path=True, P=P, mesh=mesh) # (b_s,4,4),(121,b_s,4,4)

    exit()
    
    vis_trj_H = trj_H[-20:,0,:,:] # update for visualize # (121,4,4)
    vis_trj_H[:,:3,-1] *= 1/8.
    #vis_trj_H[:, :3, -1] = (vis_trj_H[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()

    #H_grasp = copy.deepcopy(H) # (10,4,4)

    # counteract the translational shift of the pointcloud (as the spawned model in simulation will still have it)
    #H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
    H[..., :3, -1] *=1/8.
    #H_grasp[..., :3, -1] *=1/8.

    # return to normal position - scale 1/8, denormalize 
    vis_H = H.squeeze()
    P *=1/8 # (1000,3)
    mesh = mesh.apply_scale(1/8)

    # shift additional
    new_trans=np.array([0.4,0.3,0.2])
    vis_trj_H[:, :3, -1] = (vis_trj_H[:, :3, -1] + torch.as_tensor(new_trans,device=device)).float()
    P += new_trans
    H = np.eye(4)
    H[:3,-1] = new_trans
    mesh.apply_transform(H) 



    grasp_visualization.visualize_grasps(to_numpy(vis_trj_H),mesh=mesh)


    
    # # visualize for pybullet 
    # env = PandaVisualization()

    # # if mesh=None -> pc visualize // elif mesh=mesh -> mesh visualize 
    # env.process(waypoints=waypoints,mesh=None,pc=P,trj_H=vis_trj_H)

    # # visualize for pybullet 
    # env = KukaVisualization()

    # # if mesh=None -> pc visualize // elif mesh=mesh -> mesh visualize 
    # env.process(mesh=mesh,pc=P,trj_H=vis_trj_H)



    if (EVAL_SIMULATION):
        ## Evaluate Grasps in Simulation##
        num_eval_envs = 10
        evaluator = GraspSuccessEvaluator(obj_class, n_envs=num_eval_envs, idxs=[args.obj_id] * num_eval_envs, viewer=True, device=device, \
                                          rotations=[rot_quad]*num_eval_envs, enable_rel_trafo=False)
        succes_rate = evaluator.eval_set_of_grasps(H_grasp)
        print('Success cases : {}'.format(succes_rate))
