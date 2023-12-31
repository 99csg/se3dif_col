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

        ## planning related params
        self.trans = np.array([0.4,0.4,-0.05])
        self.w_ob = 5.
        self.w_e = 1. 
        self.w_sm = 2.
        self.w_tb = 5.

        ## Langevin MCMC related 
        self.q_indices = torch.tensor(torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long))
        self.q_factor = torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]])
        self.dt = 1e-1 # 1e-1
        self.std_theta = 0.5
        self.std_X = 0.5
        self.rot_trans = 1.


    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    
    def _step(self, env, quat,pre_quat, t, noise_off=True, obj_cost=None,tab_cost=None):

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

        if pre_quat is not None:
            smooth_cost = env.check_smoothness(quat, pre_quat)
        else:
            smooth_cost = torch.zeros(10,1)

        mat=self.quat_to_SE3(quaternions=quat)
        H0 = SO3_R3(R=mat[:,:3,:3],t=mat[:,:3,-1]) # (10,4,4)
        h0 = H0.log_map() # (10,6)
        h0_in = h0.requires_grad_(True) #h0_in = h0.detach().requires_grad_(True) # (10,6)
        H_in = SO3_R3().exp_map(h0_in).to_matrix() # (10,4,4)
        t_in = phase*torch.ones_like(H_in[:,0,0]) # (10,1,1)
        e = self.model(H_in, t_in) # (10,1)

        total_cost = self.w_e*e.sum()+self.w_ob*obj_cost.sum()+self.w_sm*smooth_cost.sum()+self.w_tb*tab_cost.sum()
        print(e.sum(),obj_cost.sum(),smooth_cost.sum(),tab_cost.sum())
        q_grad = torch.autograd.grad(total_cost, quat)[0] # 10 energy sum 
        #print(f"before MCMC q grad:{q_grad[0]} ")
        q_new = self.propose(quat, q_grad, reject=False, var_dt=True, noise_off=noise_off)
        #print(f"new q:{q_new[0]} ")

        # q_new = quat-np.sqrt(c_lr)*q_grad    
        # print(f"new q:{q_new[0]} ")
        return q_new

    def sample(self, save_path=False, batch=None, P=None, mesh=None):

        if batch is None:
            batch = self.batch
        #H0 = SO3_R3().sample(batch).to(self.device, torch.float32)
        #random_quat = self.generate_rn_quat(batch_size=batch)

        mesh, P = self.mesh_P_only_downscale_shift(mesh=mesh,P=P)

        # test env init
        env = KukaVisualization()  #env = pandaVisualization()
        _ ,ee_quat = env.go_to_init_state() # robot init quat
        env.test_pos_orn_init(quat=ee_quat) # visualize - blue 
        ob_list=env.init_trg_obs(mesh=mesh, pc=P) # visualize - object 
        
        # step 0 
        quat = ee_quat.repeat(10,1).requires_grad_(True)
        mesh,P=self.mesh_P_only_upscale_deshift(mesh, P)
        obj_cost = env.check_obj_collision(ob_list=ob_list,quat=quat,index=0)
        tab_cost = env.check_table_collision(quat=quat)

        qquat = quat.clone()
        qquat[...,4:7]=quat[...,4:7]*5.
        new_quat = self._step(env, quat=qquat, pre_quat=None, t=0, noise_off=False, obj_cost=obj_cost, tab_cost=tab_cost)
        #env.test_pos_orn_init(quat=new_quat[0]) 
        quat_, mesh, P = self.downscale_shift(new_quat,mesh,P)
        env.quat_based_move(quat_, 0)

        # test
        env.test_pos_orn(quat=quat_, index=0)

        # Langevin MCMC 
        if save_path:
            trj_q = quat[None,...]
            trj_q = torch.cat((trj_q, quat[None,:]),0) # (2,10,7)

        for t in range(1,self.T+1):
            up_quat, mesh, P = self.upscale_deshift(quat_,mesh,P)
            #print(f"before model:{quat[0]}\n")
            
            obj_cost = env.check_obj_collision(ob_list=ob_list,quat=up_quat,index=0)
            total_coll=env.check_collision_info(ob_list=ob_list)
            print(f"{total_coll}-collision occurs")
            self.w_ob = self.w_ob_scheduling(t)
            tab_cost = env.check_table_collision(quat=quat)

            new_quat_ = self._step(env, quat=up_quat, pre_quat=trj_q[-2,:,:], t=t, noise_off=False, obj_cost=obj_cost, tab_cost=tab_cost)
            
            down_quat, mesh, P = self.downscale_shift(new_quat_,mesh,P)
            env.quat_based_move(down_quat, 0)

            # check pseudo determinant 
            jac_t, jac_r = env.get_jacobian(current_joint=quat)
            a,b = torch.tensor(jac_t[-1]), torch.tensor(jac_r[-1])
            jac_combined = torch.cat((a,b), axis=0) # (6,7)
            U,S,V = np.linalg.svd(jac_combined)
            pseudo_determinant = np.prod(S) # element multiplication - sigma 6 
            print(S)
            print(pseudo_determinant)

            # test
            env.test_pos_orn(quat=down_quat, index=0)
            
            print(f"{t}-th Langevin Dynamics completed\n")
            if save_path:
                trj_q = torch.cat((trj_q, down_quat[None,:]), 0) # (n, 10, 7)
        

            quat_ = down_quat


        for t in range(1,self.T_fit+1):
            up_quat, mesh, P = self.upscale_deshift(quat_,mesh,P)
            #print(f"before model:{quat[0]}\n")

            # collision check 
            obj_cost = env.check_obj_collision(ob_list=ob_list,quat=up_quat,index=0)
            self.w_ob = 0.01
            total_coll=env.check_collision_info(ob_list=ob_list)
            print(f"{total_coll}-collision occurs")
            tab_cost = env.check_table_collision(quat=quat)

            new_quat_ = self._step(env, quat=up_quat, pre_quat=trj_q[-2,:,:], t=t, noise_off=True, obj_cost=obj_cost, tab_cost=tab_cost)
            
            down_quat, mesh, P = self.downscale_shift(new_quat_,mesh,P)
            env.quat_based_move(down_quat, 0)

            # test
            env.test_pos_orn(quat=down_quat, index=0)

            print(f"{t}-th deterministic sampling completed\n")
            if save_path:
                trj_q = torch.cat((trj_q, down_quat[None,:]), 0)

            quat_ = down_quat
        time.sleep(10)
        horizon = self.T+self.T_fit
        
        # # move to init pos 
        # _ ,ee_quat = env.go_to_init_state() # robot init quat

        # # postprocessing for smooth trajectory 
        # smooth_path = self.generate_smooth_trajectory(trj_q,horizon) # (12,10,7)

        # # visualize trajectory 
        # for i in range(1,len(smooth_path)):
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

    def propose(self, T, grad, reject = False, var_dt = True, noise_off = True):
        #T, grad = T.detach(), grad.detach() # (Nt, 7), (Nt, 7)

        nat_grad = grad.clone() # (Nt, 7)
        L = T[...,self.q_indices] * self.q_factor # (Nt, 4, 3)  # LSO(3)
        if var_dt is True or reject is True:
            lie_grad = torch.empty_like(grad[...,:6]) # (Nt, 6) # (v, \omg)  # h
            lie_grad[..., :3] = transforms.quaternion_apply(transforms.quaternion_invert(T[...,:4]), point = grad[...,4:]) # (Nt, 3), Translation part
            lie_grad[..., 3:] = torch.einsum('...ia,...i->...a', L, grad[...,:4]) # (Nt, 3), Rotation part
        Ginv = (torch.eye(4, dtype=T.dtype, device=T.device) - torch.einsum('...i,...j->...ij', T[...,:4], T[...,:4]))/4  # (Nt, 4, 4)
        nat_grad[..., :4] = self.rot_trans*(torch.einsum('...ij,...j->...i', Ginv, grad[...,:4])) # (Nt, 7)
        if var_dt is True:
            dt = torch.min(3 / (lie_grad.abs().sum(dim=-1) + 1e-5), torch.tensor(1., device=grad.device, dtype=grad.dtype)).unsqueeze(-1) # (Nt,1)
            dt = dt*self.dt                                # (Nt,1)
        else:                                                                                          
            dt = self.dt
        std_R = torch.sqrt(2*dt) * self.std_theta      # (Nt,1) or (1,)
        std_X = torch.sqrt(2*dt) * self.std_X          # (Nt,1) or (1,)

        
        noise_R_lie = torch.randn_like(T[...,:3]) # (Nt, 3) # rotation
        noise_X_lie = torch.randn_like(noise_R_lie) # (Nt, 3) # translation
        noise_q = torch.einsum('...ij,...j', L, noise_R_lie) # (Nt, 4)
        # noise_X = transforms.quaternion_apply(quaternion = T[...,:4], point = noise_X_lie) # (Nt, 3) # Unnecessary due to rotation invariance of normal distribution
        noise_X = noise_X_lie  # (Nt, 3)

        if noise_off:
            dt = 0.01*dt
            dT = -(nat_grad*dt)
        else:
            dT = -(nat_grad*dt) + torch.cat([noise_q*std_R, noise_X*std_X], dim=-1) # (Nt, 7)
        #print(f"nat_grad:{nat_grad[0]}, \ndt:{dt[0]}, \nstd_R:{std_R[0]}, \nstd_X:{std_X[0]}\n")
        #print(f"gradient term:{nat_grad[0]*dt[0]},\nnoise term:{torch.cat([noise_q*std_R, noise_X*std_X], dim=-1)[0]}\n")
        T_prop = T + dT # (Nt, 7)

        #print(f"after MCMC q gradient:{dT[0]}")
        T_prop[...,:4] = self.normalize_quaternion(T_prop[...,:4])

        return T_prop

    def w_ob_scheduling(self, t):
        w_ob = max(5-((5-0.1)*t/(0.8*self.w_ob)),0.1)
        return w_ob

    def slerp(self, q0, q1, t):
        """Spherical linear interpolation."""
        dot = torch.dot(q0, q1)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot) * t
        q1_rel = q1 - q0 * dot
        q1_rel /= torch.norm(q1_rel)

        return torch.cos(theta) * q0 + torch.sin(theta) * q1_rel

    def generate_smooth_trajectory(self, waypoints, num_smooth_points):
        waypoints = torch.tensor(waypoints, dtype=torch.float32)
        smooth_trajectory = torch.zeros((num_smooth_points, 7))

        for i in range(1, num_smooth_points):
            t = i / (num_smooth_points - 1)
            q0 = waypoints[i,0, :4]  # (4)
            q1 = waypoints[i-1,0, :4] # (4)

            smooth_trajectory[i, :4] = self.slerp(q0, q1, t)
            smooth_trajectory[i, 4:] = waypoints[i,0, 4:]  * (1 - t) + waypoints[i-1,0, 4:]  * t

        return smooth_trajectory
    
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




