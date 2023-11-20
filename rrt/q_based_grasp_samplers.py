import numpy as np
import torch
import os, os.path as osp

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
from se3dif.rrt.ik_ex import KukaVisualization
from se3dif.rrt.ik_ex_panda import pandaVisualization




class Q_based_Grasp_AnnealedLD():
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

        self.trans = np.array([0.4,0.3,0.2])

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    
    def _step(self, env, q_list, t, noise_off=True, obj_cost=None):

        ## Phase
        noise_std = .5
        eps = 1e-3
        phase = ((self.T - t) / (self.T)) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i ** 2 / sigma_T ** 2
        c_lr = alpha * ratio
        if noise_off:
            c_lr = 0.003

        for _ in range(self.k_steps):
            b_s = q_list.shape[0]
            T = torch.zeros((b_s,4,4))
            for i in range(b_s):
                q1 = q_list[i]
                tf, dH_dq = env.compute_fk(q1) # fk process 
                T[i]=torch.tensor(tf)

            H0 = SO3_R3(R=T[:,:3,:3],t=T[:,:3,-1]) # (10,4,4)
            H0_in = H0.to_matrix().detach().requires_grad_(True) # (10,4,4)
            h0 = H0.log_map() # (10,6)
            h0_in = h0.detach().requires_grad_(True) # (10,6)
            H_in = SO3_R3().exp_map(h0_in).to_matrix() # (10,4,4)
            t_in = phase*torch.ones_like(H_in[:,0,0]) # (10,1,1)
            e = self.model(H_in, t_in) # (10,1)

            q_grad = torch.autograd.grad(e.sum(), q_list)[0] # 10 energy sum 
            print(q_grad)

            ## 3. Compute noise vector ##
            if noise_off:
                noise = torch.zeros_like(h0_in)
            else:
                noise = torch.randn_like(h0_in)*noise_std
            
            q1 = q1-0.5*c_lr*q_grad+np.sqrt(c_lr)*noise

        return q1

    def sample(self, save_path=False, batch=None, P=None, mesh=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)

        H0, mesh, P = self.downscale_shift(H0=H0,mesh=mesh,P=P)

        env = KukaVisualization()
        #env = pandaVisualization()
        ob_list=env.init_trg_obs(mesh=mesh, pc=P)

        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if save_path:
            trj_H = Ht[None,...]

        for t in range(self.T):
            Ht, mesh, P = self.upscale_deshift(Ht,mesh,P)
            q_list = env.H_to_q(Ht)

            # check collision
            robot_xyz_state = env.get_xyz_state() # get x,y,z from robot
            obj_cost = env.check_obj_collision(ob_list=ob_list,robot_xyz_state=robot_xyz_state)
            coll_grad=torch.autograd.grad(obj_cost, robot_xyz_state)[0]
            print(coll_grad)
            Ht = self._step(env, q_list, t, noise_off=self.deterministic,obj_cost=obj_cost)
            
            Ht, mesh, P = self.downscale_shift(Ht,mesh,P)
            env.move(Ht)
            print(f"{t}-th Langevin Dynamics completed")
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        for t in range(self.T_fit):
            Ht, mesh, P = self.upscale_deshift(Ht,mesh,P)
            Ht = self._step(Ht, self.T, noise_off=True)
            Ht, mesh, P = self.downscale_shift(Ht,mesh,P)
            env.move(Ht)
            print(f"{t}-th deterministic sampling completed")

            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        if save_path:
            return Ht, trj_H
        else:
            return Ht
        
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
        




if __name__ == '__main__':
    import torch.nn as nn

    class model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, H, k):
            H_th = SO3_R3(R=H[:, :3, :3], t=H[:, :3, -1])
            x = H_th.log_map()
            return x.pow(2).sum(-1)

    ## 2. Grasp_AnnealedLD
    generator = Grasp_AnnealedLD(model(), T=100, T_fit=500, k_steps=1)
    H = generator.sample()
    print(H.shape) # (10,4,4)




