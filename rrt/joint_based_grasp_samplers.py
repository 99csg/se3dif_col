import numpy as np
import torch
import os, os.path as osp
import math
import time
import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
from se3dif.rrt.ik_ex import KukaVisualization
from se3dif.rrt.ik_ex_panda import pandaVisualization
from pytorch3d import transforms




class Joint_based_Grasp_AnnealedLD():
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
        self.trans = np.array([0.4,0.3,-0.1])
        self.w_ob = 5.
        self.w_e = 1. 

        ## Langevin MCMC related 
        self.q_indices = torch.tensor(torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long))
        self.q_factor = torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]])
        self.dt = 1e-1 # 1e-1
        self.std_theta = 0.5
        self.std_X = 0.5
        self.rot_trans = 1.


    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    
    def _step(self, env, angles=None, pre_joint=None, t=0, noise_off=False, obj_cost=None, tab_cost=None):
        
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

        # mapping : angles -> H -> e 
        jac_t, jac_r = env.get_jacobian(angles) # (3,7), (3,7)
        a,b = torch.tensor(jac_t[-1]), torch.tensor(jac_r[-1])
        jac_combined = torch.cat((a,b), axis=0) # (6,7)
        U,S,V = np.linalg.svd(jac_combined)
        pseudo_determinant = np.prod(S) # element multiplication - sigma 6 
        print(S)
        print(pseudo_determinant)

        _,ee_state = env.get_xyz_state() # (7,1)
        pos,quat = ee_state[:3],ee_state[3:]
        renew_quat = torch.cat((quat, pos),dim=0)
        ten_quat = renew_quat.unsqueeze(0).repeat(10,1)
        H=self.quat_to_SE3(quaternions=ten_quat) # (10,4,4)

        H0 = SO3_R3(R=H[:,:3,:3],t=H[:,:3,-1]) # (10,4,4)
        h0 = H0.log_map() # (10,6)
        h0_in = h0.requires_grad_(True) #h0_in = h0.detach().requires_grad_(True) # (10,6)
        H_in = SO3_R3().exp_map(h0_in).to_matrix() # (10,4,4)
        t_in = phase*torch.ones_like(H_in[:,0,0]) # (10,1,1)
        e = self.model(H_in, t_in) # (10,1)
        dE_dh = torch.autograd.grad(e.sum(), h0_in)[0] # (10,6)

        energy_angle_gradient = torch.mm(dE_dh,jac_combined)
        
        if noise_off:
            noise = torch.zeros_like(torch.tensor(angles))
        else:
            noise = torch.randn_like(torch.tensor(angles))*noise_std
        
        #print(f"before MCMC q grad:{q_grad[0]} ")
        new_angle = torch.tensor(angles) + 0.5*c_lr*energy_angle_gradient+0.01*np.sqrt(c_lr)*noise


        print(angles[0],"\n",0.5*c_lr*energy_angle_gradient[0],"\n",0.01*np.sqrt(c_lr)*noise[0])

        return new_angle

    def sample(self, save_path=False, batch=None, P=None, mesh=None):

        if batch is None:
            batch = self.batch
        #H0 = SO3_R3().sample(batch).to(self.device, torch.float32)
        #q_list=env.H_to_q(H0)

        mesh, P = self.mesh_P_only_downscale_shift(mesh=mesh,P=P)

        # test env init
        env = KukaVisualization() 
        _, ee_quat=env.go_to_init_state()
        env.test_pos_orn_init(quat=ee_quat) # visualize - blue 
        ob_list=env.init_trg_obs(mesh=mesh, pc=P) # visualize - object 
        
        # step 0 
        init_joint_states = [math.pi/2, math.pi/4, 0, -math.pi/2, 0, math.pi/4,0]
        

        mesh,P=self.mesh_P_only_upscale_deshift(mesh, P)
        obj_cost = env.angle_based_check_obj_collision(ob_list=None,angles=init_joint_states,index=0)
        tab_cost = torch.tensor([0.0])
        new_joint = self._step(env, angles=init_joint_states, pre_joint=None, t=0, noise_off=False, obj_cost=obj_cost, tab_cost=tab_cost)
        mesh, P = self.mesh_P_only_downscale_shift(mesh=mesh,P=P)
        env.joint_based_move(new_joint[0])

        # if save_path:
        #     trj_q = quat[None,...]

        for t in range(0,self.T):
            mesh, P = self.mesh_P_only_upscale_deshift(mesh,P)
            
            obj_cost = env.angle_based_check_obj_collision(ob_list=None,angles=None,index=0)
            self.w_ob = self.w_ob_scheduling(t)
            new_joint = self._step(env, angles=new_joint, pre_joint=None, t=0, noise_off=False, obj_cost=obj_cost, tab_cost=tab_cost)
        
            mesh, P = self.mesh_P_only_downscale_shift(mesh,P)
            env.joint_based_move(new_joint[0])

            
            print(f"{t}-th Langevin Dynamics completed\n")
            # if save_path:
            #     trj_q = torch.cat((trj_q, down_quat[None,:]), 0)


           


        for t in range(0,self.T_fit):
            mesh, P = self.mesh_P_only_upscale_deshift(mesh,P)
            
            obj_cost = env.angle_based_check_obj_collision(ob_list=None,angles=None,index=0)
            self.w_ob = self.w_ob_scheduling(t)
            new_joint = self._step(env, angles=new_joint, pre_joint=None, t=0, noise_off=True, obj_cost=obj_cost, tab_cost=tab_cost)
        
            mesh, P = self.mesh_P_only_downscale_shift(mesh,P)
            env.joint_based_move(new_joint[0])

            print(f"{t}-th deterministic sampling completed\n")


        # horizon = self.T+self.T_fit
        
        # # postprocessing for smooth trajectory 
        # smooth_path = self.generate_smooth_trajectory(trj_q,horizon) # (60,7)

        # # visualize trajectory 
        # for i in range(len(smooth_path)):
        #     env.quat_based_move(smooth_path, i)
        # trj_q = smooth_path 
        if save_path:
            return quat, trj_q
        else:
            return quat
        
    def downscale_shift(self,quat,mesh,P):
        # down scale
        #H0[...,:3,-1]*=1/8. #(10,4,4)
        quat[...,4:7]=quat[...,4:7]*1/8. # (10,7)
        P*=1/8.
        mesh=mesh.apply_scale(1/8)

        # shift + 
        
        new_trans=self.trans
        #H0[:,:3,-1] = (H0[:,:3,-1] + torch.as_tensor(new_trans, device=self.device)).float()
        new_quat = quat.clone()
        quat[...,4:7] = (new_quat[...,4:7] + torch.as_tensor(new_trans, device=self.device)).float()
        P += new_trans
        H=np.eye(4)
        H[:3,-1] = new_trans
        mesh.apply_transform(H)
        return quat, mesh, P

    def upscale_deshift(self,quat,mesh,P):
        # shift - 
        new_trans=-self.trans
        #H0[:,:3,-1] = (H0[:,:3,-1] + torch.as_tensor(new_trans, device=self.device)).float()
        new_quat = quat.clone()
        quat[...,4:7] = (new_quat[...,4:7] + torch.as_tensor(new_trans, device=self.device)).float()
        P += new_trans
        H=np.eye(4)
        H[:3,-1] = new_trans
        mesh.apply_transform(H)

        # upscale 
        #H0[...,:3,-1]*=8.
        quat[...,4:7]=quat[...,4:7]*8.
        P*=8.
        mesh=mesh.apply_scale(8)  


        return quat, mesh, P
    
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
    
    def SE3_to_quaternion(self, rotation_matrix):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

        quaternion = torch.empty(4)

        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]

        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2
            quaternion[0] = 0.25 * s
            quaternion[1] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            quaternion[2] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            quaternion[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = torch.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            quaternion[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            quaternion[1] = 0.25 * s
            quaternion[2] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            quaternion[3] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = torch.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            quaternion[0] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            quaternion[1] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            quaternion[2] = 0.25 * s
            quaternion[3] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            quaternion[0] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            quaternion[1] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            quaternion[2] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            quaternion[3] = 0.25 * s

        return quaternion

    def generate_rn_quat(self, batch_size):
        quaternions = torch.randn(batch_size,7,requires_grad=True)
        
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

        return quaternions
    
    def calculate_angle_with_neg_z(self,quats):
        neg_z = torch.tensor([0,0,-1], dtype=torch.float32)
        angles = []
        for quat in quats:
            w, x, y, z = quat[3], quat[0], quat[1], quat[2]
            rot_matrix = torch.tensor([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
            ])
            rotated_vec = torch.matmul(rot_matrix, neg_z)
            dot_product = torch.dot(rotated_vec, neg_z)
            angle = torch.acos(dot_product)*180/math.pi
            angles.append(angle)
        return torch.tensor(angles)

    def quat_to_SE3(self, quaternions): # (qw, qx, qy, qz, x, y, z)
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # Computing the elements of the rotation matrix
        xx, xy, xz = x * x, x * y, x * z
        yy, yz, zz = y * y, y * z, z * z
        wx, wy, wz = w * x, w * y, w * z

        mat = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), quaternions[:,4]], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), quaternions[:,5]], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), quaternions[:,6]], dim=-1),
        torch.tensor([0, 0, 0, 1], dtype=quaternions.dtype, device=quaternions.device).repeat(quaternions.size(0), 1)
                        ], dim=1)

        return mat.requires_grad_(True)
    
    def normalize_quaternion(self,q):
        return transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))

    def w_ob_scheduling(self, t):
        w_ob = max(5-((5-0.1)*t/(0.8*self.w_ob)),0.1)
        return w_ob

    
if __name__ == '__main__':
    pass
    # import torch.nn as nn

    # class model(nn.Module):
    #     def __init__(self):
    #         super().__init__()

    #     def forward(self, H, k):
    #         H_th = SO3_R3(R=H[:, :3, :3], t=H[:, :3, -1])
    #         x = H_th.log_map()
    #         return x.pow(2).sum(-1)

    # ## 2. Grasp_AnnealedLD
    # generator = Grasp_AnnealedLD(model(), T=100, T_fit=500, k_steps=1)
    # H = generator.sample()
    # print(H.shape) # (10,4,4)




