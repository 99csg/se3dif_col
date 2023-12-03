import numpy as np
import torch
import os, os.path as osp
import math
import matplotlib.pyplot as plt
import time

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
from se3dif.rrt.ik_ex import KukaVisualization
from se3dif.rrt.ik_ex_panda import pandaVisualization
from se3dif.rrt.rrt_star import RobotArm, RRTStar
from se3dif.rrt.rrt_before_grasp_samplers import Grasp_AnnealedLD

class rrt_Grasp_AnnealedLD():
    def __init__(self, model, device='cpu', batch=10, dim =3, k_steps=1,
                 T=200, T_fit=5, deterministic=False):

        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.k_steps = k_steps
        self.deterministic = deterministic

        ## planning related params
        self.trans = np.array([0.4,0.4,-0.05])
        self.w_ob = 5.
        self.w_e = 1. 
        self.w_sm = 5.
        self.w_tb = 5.

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

   

    def sample(self, save_path=False, batch=None, P=None, mesh=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)

        H0, mesh, P = self.downscale_shift(H0=H0,mesh=mesh,P=P)

        env = KukaVisualization()
        ob_list=env.init_trg_obs(mesh=mesh, pc=P)

        # get start, final state 
        generator = Grasp_AnnealedLD(self.model, batch=batch, T=20, T_fit=20, k_steps=2, device='cpu')
        H0, Hf = generator.sample(save_path=False) # (b_s,4,4),(121,b_s,4,4)
        print("finished sampling")
        H0[..., :3, -1] *=1/8.
        Hf[..., :3, -1] *=1/8.
        H0[:,:3,-1] = (H0[:,:3,-1] + torch.as_tensor(self.trans, device=self.device)).float()
        Hf[:,:3,-1] = (Hf[:,:3,-1] + torch.as_tensor(self.trans, device=self.device)).float()
        
        # H->joint q
        q0=env.H_to_q(H0)
        qf=env.H_to_q(Hf)

        # test H
        env.test_H(H0)
        env.test_H(Hf)

        show_animation = True
        ## rrt input 
        seven_joint_arm = RobotArm([[0., -math.pi/2., 0., .34],
                                    [0., math.pi/2., 0., 0.],
                                    [0., math.pi/2., 0., 0.4],
                                    [0., -math.pi/2., 0., 0.],
                                    [0., -math.pi/2., 0., 0.4],
                                    [0., math.pi/2., 0., 0.],
                                    [0., 0.,         0., 0.126]])
        
        obstacle_list = [
            (ob_list[0][1][0], ob_list[0][1][1], ob_list[0][1][2], env.object_edge_radius),
            (ob_list[1][1][0], ob_list[1][1][1], ob_list[1][1][2], env.object_edge_radius),
            (ob_list[2][1][0], ob_list[2][1][1], ob_list[2][1][2], env.object_edge_radius),
        ]

        # start = [math.pi/2, math.pi/4, 0, -math.pi/2, 0, math.pi/4, 0]
        # end = [0.2648, 1.6627, -1.8099, 1.42068, 1.9223, 2.1205, -3.04268]
        start = q0.detach().numpy()
        end = qf.detach().numpy()
        
        rrt_star = RRTStar(start=start[0], goal=end[0], rand_area=[0,2], max_iter=200, robot=seven_joint_arm, obstacle_list=obstacle_list)

        path = rrt_star.planning(animation=show_animation, search_until_max_iter=False)
        


        if path is None:
            print("Cannot find path")
        else:
            print("found path!!")

        # Draw final path
        if show_animation:
            ax = rrt_star.draw_graph()

            # Plot final configuration
            x_points, y_points, z_points = seven_joint_arm.get_points(path[-1])
            ax.plot([x for x in x_points],
                    [y for y in y_points],
                    [z for z in z_points],
                    "o-", color="red", ms=5, mew=0.5)

            for i, q in enumerate(path):
                x_points, y_points, z_points = seven_joint_arm.get_points(q)
                ax.plot([x for x in x_points],
                        [y for y in y_points],
                        [z for z in z_points],
                        "o-", color="grey",  ms=4, mew=0.5)
                plt.pause(0.01)

            plt.show()
        
        # show in pybullet 
        if path is not None:
            for i in range(len(path)-1,-1,-1):
                print(f"{i}th step")
                env.joint_based_move(path[i])
                
            time.sleep(10)

        else:
            print("no path")


    def downscale_shift(self,H0,mesh,P):
        # down scale
        H0[...,:3,-1]*=1/8. #(10,4,4)
        P*=1/8.
        mesh=mesh.apply_scale(1/8)

        # shift + 
        
        new_trans=self.trans
        H0[:,:3,-1] = (H0[:,:3,-1] + torch.as_tensor(new_trans, device=self.device)).float()
        P += new_trans
        H=np.eye(4)
        H[:3,-1] = new_trans
        mesh.apply_transform(H)
        return H0, mesh, P


    def upscale_deshift(self,H0,mesh,P):
        # shift - 
        new_trans=-self.trans
        H0[:,:3,-1] = (H0[:,:3,-1] + torch.as_tensor(new_trans, device=self.device)).float()
        P += new_trans
        H=np.eye(4)
        H[:3,-1] = new_trans
        mesh.apply_transform(H)

        # upscale 
        H0[...,:3,-1]*=8.
        P*=8.
        mesh=mesh.apply_scale(8)  


        return H0, mesh, P
    
    def mesh_P_only_downscale_shift(self,mesh,P):
        # down scale
        P*=1/8.
        mesh=mesh.apply_scale(1/8)

        # shift + 
        
        new_trans=self.trans
        P += new_trans
        H=np.eye(4)
        H[:3,-1] = new_trans
        mesh.apply_transform(H)
        return mesh, P
        
    def mesh_P_only_upscale_deshift(self,mesh,P):
        # shift - 
        new_trans=-self.trans
        P += new_trans
        H=np.eye(4)
        H[:3,-1] = new_trans
        mesh.apply_transform(H)

        # upscale 
        
        P*=8.
        mesh=mesh.apply_scale(8)  


        return mesh, P


if __name__ == '__main__':
    import torch.nn as nn

    class model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, H, k):
            H_th = SO3_R3(R=H[:, :3, :3], t=H[:, :3, -1])
            x = H_th.log_map()
            return x.pow(2).sum(-1)

    ## 1. Approximated Grasp_AnnealedLD
    generator = ApproximatedGrasp_AnnealedLD(model(), T=100, T_fit=500)
    H = generator.sample()
    print(H.shape) # (10,4,4)

    ## 2. Grasp_AnnealedLD
    generator = Grasp_AnnealedLD(model(), T=100, T_fit=500, k_steps=1)
    H = generator.sample()
    print(H.shape) # (10,4,4)




