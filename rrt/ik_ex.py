import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import os
import torch 
from scipy.spatial.transform import Rotation
import random
import numpy as np
import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3

# print(pybullet_data.getDataPath()) # /home/csg/mambaforge/envs/se3dif_env/lib/python3.7/site-packages/pybullet_data

class KukaVisualization():
  
    def __init__(self) -> None:
        self.object_edge_radius = 0.07
        
        clid = p.connect(p.SHARED_MEMORY)
        
        if (clid < 0):
            p.connect(p.GUI)
            #p.connect(p.SHARED_MEMORY_GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # robot config - when change robot check #move - numJoints 
        kukaId=p.loadURDF("kuka_iiwa/model.urdf",[0,0,0],[0,0,0,1]) 
        #kukaId=p.loadURDF("franka_panda/panda.urdf",[0,0,0],[0,0,0,1],useFixedBase=True) # 
        
        p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
        self.kukaId = kukaId
        
        self.numJoints = p.getNumJoints(kukaId) 
        self.kukaEndEffectorIndex = self.numJoints-1
        self.ikSolver=0
        self.p = p
        self.jd = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]

        p.setGravity(0, 0, 0)

        #table config
        self.table_id = p.loadURDF("table/table.urdf")
        table_pos = [0,0.4,-0.75]
        self.table_z = table_pos[-1]
        self.table_r = 0.05
        table_orientation = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.table_id, table_pos,table_orientation)
        
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)

    def init_trg_obs(self, mesh=None, pc=None):
        
        # mesh or pc load 
        if mesh is not None:
            mesh_body_id,mug_center = self.mesh_to_pybullet(mesh,pc)
        if mesh is None:
            pc_id,mug_center = self.pc_to_pybullet(pc)

        # random object load
        num = 3
        rad = 0.3 # mug-object distance 
        self.ob_list = self.call_object(num,mug_center,rad) # [[3, [0.584874501884813, 0.46591183056858754, 0.5685968345463284]], [4, [0.27034844244277056, 0.5943483417653938, 0.6457106129726388]], [5, [0.43319756030566836, 0.6125678578395343, 0.6639166800981371]]]
        
        #ob_list = None
        return self.ob_list

    def H_based_move(self,Ht=None):
        
        tf = Ht[0,:,:] # (4,4)
        position = (tf[:3,3]).tolist()
           

        orientation = p.getQuaternionFromEuler(self.getEulerFromMatrix(tf[:3,:3])) # torch(T_matrix)->list(euler)->list(quaternion)

        joint_angles = p.calculateInverseKinematics(self.kukaId,
                                                    self.kukaEndEffectorIndex, 
                                                    position, 
                                                    orientation,
                                                    jointDamping=self.jd,
                                                    solver=self.ikSolver,
                                                    maxNumIterations=100,
                                                    residualThreshold=.01)

        # test 
        #print(position, orientation)
        # rod_length = 0.05
        # rod_radius = 0.001
        # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,radius=rod_radius, length=rod_length,rgbaColor=[1, 0, 0, 1])
        # collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=rod_radius, height=rod_length)
        # rod_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id)
        # p.resetBasePositionAndOrientation(rod_id,position,orientation)

        # move 
        for k in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,
                                    jointIndex=k,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_angles[k],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
    

        for _ in range(100): # time control 
            p.stepSimulation()
            time.sleep(0.01) # wait 


    def joint_based_move(self, joint_angles):

        for k in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,
                                    jointIndex=k,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_angles[k],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
        # for i in range(len(self.ob_list)):
        #     closestPoints=p.getClosestPoints(self.kukaId,self.ob_list[i],0.01) # does not requrire simulation step
        #     if(closestPoints):
        #         print("Objects are in collision")
        #     else:
        #         print("Not in collision")


        for _ in range(10): # time control 
            p.stepSimulation()
            time.sleep(0.01) # wait 
        
        
    def quat_based_move(self,quat=None,index=None):
        
        quat = quat[index,:] # (7)
        position = quat[4:] 
        orientation = quat[:4]

        joint_angles = p.calculateInverseKinematics(self.kukaId,
                                                    self.kukaEndEffectorIndex, 
                                                    position, 
                                                    orientation,
                                                    jointDamping=self.jd,
                                                    solver=self.ikSolver,
                                                    maxNumIterations=100,
                                                    residualThreshold=.01)
        
        # test 
        #print(position, orientation)
        # rod_length = 0.05
        # rod_radius = 0.001
        # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,radius=rod_radius, length=rod_length,rgbaColor=[1, 0, 0, 1])
        # collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=rod_radius, height=rod_length)
        # rod_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id)
        # p.resetBasePositionAndOrientation(rod_id,position,orientation)

        # move 
        for k in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,
                                    jointIndex=k,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_angles[k],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
    

        for _ in range(10): # time control 
            p.stepSimulation()
            time.sleep(0.01) # wait 
        #time.sleep(0.5)

    def process(self, mesh=None, pc=None, trj_H=None):
        # whole trajectory optimization 

        # mesh or pc load 
        if mesh is not None:
            mesh_body_id,mug_center = self.mesh_to_pybullet(mesh,pc)
        if mesh is None:
            pc_id,mug_center = self.pc_to_pybullet(pc)

        # random object load
        num = 3
        rad = 0.3
        ob_list = self.call_object(num,mug_center,rad)

        i=0
        jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        waypoints=[]
        # ik process
        for i in range((trj_H.shape)[0]):
            tf = trj_H[i,:,:]
            position = (tf[:3,3]).tolist()
            orientation = p.getQuaternionFromEuler(self.getEulerFromMatrix(tf[:3,:3])) # torch(T_matrix)->list(euler)->list(quaternion)

            joint_angles = p.calculateInverseKinematics(self.kukaId,
                                                        self.kukaEndEffectorIndex, 
                                                        position, 
                                                        orientation,
                                                        jointDamping=jd,
                                                        solver=0,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)
            
            waypoints.append(list(joint_angles)) 
        print(f"waypoints len:{len(waypoints)}") 

        # test 
        #print(position, orientation)
        rod_length = 0.05
        rod_radius = 0.001
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,radius=rod_radius, length=rod_length,rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=rod_radius, height=rod_length)
        rod_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id)
        p.resetBasePositionAndOrientation(rod_id,position,orientation)


        # move 
        ii=0
        for wp in waypoints:
            for k in range(self.numJoints):
                p.setJointMotorControl2(bodyIndex=self.kukaId,
                                        jointIndex=k,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=wp[k],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1)
            # # check collision
            # self.check_collision
            
            for _ in range(100): # time control 
                p.stepSimulation()
                time.sleep(0.01) # wait 
            time.sleep(0.5)

            ii+=1
            print(f"{ii}th-step visualize ")

    def mesh_to_pybullet(self,mesh,pc):
        # tf = trj_H[-1,:,:]
        # position = (tf[:3,3]).tolist()
        # visual_shape_id_last = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.01,rgbaColor=[0,1,0,1])
        # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.01,rgbaColor=[1.,0.,0.,1.])
        # p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id_last,basePosition=position)

        position = [0,0,0]
        orientation=[0,0,0,1]
        temp_mesh_file = "temp_mesh.obj"
        mesh.export(temp_mesh_file)
        
        mesh_id = p.createCollisionShape(shapeType=p.GEOM_MESH,fileName=temp_mesh_file)
        body_id = p.createMultiBody(baseCollisionShapeIndex=mesh_id,basePosition=position, baseOrientation=orientation)
        mug_center=np.mean(pc,0)
        p.setCollisionFilterPair(self.kukaId, mesh_id, -1, -1, enableCollision=0)
        os.remove(temp_mesh_file)

        return body_id,mug_center

    def pc_to_pybullet(self,pc):
        # tf = trj_H[-1,:,:]
        # position = (tf[:3,3]).tolist()
        # visual_shape_id_last = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.01,rgbaColor=[0,1,0,1])
        # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.01,rgbaColor=[1.,0.,0.,1.])
        # p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id_last,basePosition=position)

        for i in range(pc.shape[0]):
            point = (pc[i]).tolist()
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id,basePosition=point)

    def getEulerFromMatrix(self,matrix:torch)->list: # rotation matrix -> euler angle 
        matrix_ = matrix.detach().numpy()
        # euler_angles = [x,y,z]
        rot = Rotation.from_matrix(matrix_)
        euler_angles = rot.as_euler('xyz')

        return euler_angles.tolist()
    
    def call_object(self,num,mug_center,rad):
        urdf_paths = ["lego/lego.urdf","duck_vhacd.urdf","soccerball.urdf"]
        ob_list = []

        for i in range(num):
            
            while True:
                pos = [random.uniform(0.1,0.7),random.uniform(0.1,0.7),random.uniform(0.1,0.7)]
                distance = np.linalg.norm(np.array(pos)-np.array(mug_center))
                if distance>=rad:
                    break
            
            orn = np.random.normal(size=4)
            orn /= np.linalg.norm(orn)
            orn = orn.tolist()
            #print(pos,orn)
            
            urdf_path = random.choice(urdf_paths)
            urdf_paths.remove(urdf_path)

            object_id = p.loadURDF(urdf_path,pos,orn)

            p.changeDynamics(object_id, -1, mass=0)
            ob_list.append([object_id,pos])

            # # test radius
            # print(f"{object_id},{urdf_path}")
            # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=self.object_edge_radius,rgbaColor=[1, 0, 0, 1])
            # sphere_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id,basePosition=pos)
            
        
       
        
        return ob_list
    
    def H_based_check_table_collision(self,H):
        dis = max(0,-(H[0,2,-1]-self.table_z-self.table_r))
        cost = torch.tensor(dis)

        return cost
    
    def check_table_collision(self,quat):
        dis = (quat[0,-1]-self.table_z)**2
        cost = torch.exp(-0.5*dis)

        return cost
        
    def check_obj_collision(self, ob_list,quat,index):
        if ob_list is None or len(ob_list)==0:
            return torch.tensor(0.0)
        
        total_cost = torch.tensor(0.0)
        for ob_data in (ob_list):
            id, ob_xyz = ob_data[0],torch.tensor(ob_data[1])
            cost = torch.sum((quat[index,4:]-ob_xyz)**2,dim=0)
            #cost /= (self.object_edge_radius**2)
            total_cost += torch.exp(-0.5*cost).sum()
        
        return total_cost 
    
    def H_based_check_obj_collision(self, ob_list,H,index):
        if ob_list is None or len(ob_list)==0:
            return torch.tensor(0.0)
        
        total_cost = torch.tensor(0.0)
        for ob_data in (ob_list):
            id, ob_xyz = ob_data[0],torch.tensor(ob_data[1])
            cost = max(0, -(H[index,:3,-1]-ob_xyz).sum()-(self.object_edge_radius))
            total_cost += torch.tensor(cost)
        
        return total_cost 
    
    def check_collision_info(self, ob_list):
        robot_xyz_state, _ = self.get_xyz_state() # (7,7) xyzqw,,,
        total_coll = 0
        for ob_data in (ob_list):
            id, ob_xyz = ob_data[0],torch.tensor(ob_data[1])
            for i in range(len(robot_xyz_state)):
                dis = torch.sum((robot_xyz_state[i,:3]-ob_xyz)**2,dim=0)
                if dis<=self.object_edge_radius:
                    total_coll+=1
        return total_coll

    
    def check_smoothness(self, quat, pre_quat):
        diff = quat[0]-pre_quat[0]
        cost = torch.norm(diff)**2
        return cost 
    
    def angle_based_check_obj_collision(self, ob_list,angles,index):
        if ob_list is None or len(ob_list)==0:
            return torch.tensor(0.0)
        
        total_cost = torch.tensor(0.0)
        for ob_data in (ob_list):
            id, ob_xyz = ob_data[0],torch.tensor(ob_data[1])
            cost = torch.sum((quat[index,4:]-ob_xyz)**2,dim=0)
            #cost /= (self.object_edge_radius**2)
            total_cost += torch.exp(-0.5*cost).sum()
        
        return total_cost 
    
    def compute_fk(self,theta):
        theta = torch.tensor(theta, dtype=torch.float32, requires_grad=True)
        alpha = torch.tensor([-math.pi/2,math.pi/2, math.pi/2,
                              -math.pi/2, -math.pi/2, math.pi/2, 0])
        d = torch.tensor([0.34, 0, 0.4, 0, 0.4, 0, 0.126])
        a = torch.tensor([0,0,0,0, 0,0,0])
        T = torch.eye(4, requires_grad=True)
        
        for i in range(7):
            T0 = torch.tensor([[torch.cos(theta[i]), -torch.sin(theta[i])*torch.cos(alpha[i]), torch.sin(theta[i])*torch.sin(alpha[i]), a[i]*torch.cos(theta[i])],
                                [torch.sin(theta[i]), torch.cos(theta[i])*torch.cos(alpha[i]), -torch.cos(theta[i])*torch.sin(alpha[i]), a[i]*torch.sin(theta[i])],
                                [0,torch.sin(alpha[i]), torch.cos(alpha[i]), d[i]],
                                [0,0,0,1]], requires_grad=True)
    
            T = T @ T0

        #dH_dq = torch.zeros((7, 4, 4), requires_grad=True)

        # for i in range(7):
        #     grad = torch.autograd.grad(T[3, :], theta[i], retain_graph=True, allow_unused=True)[0]
        #     if grad is not None:
        #         dH_dq[i] = grad
        dH_dq = None

        return T, dH_dq

    def get_xyz_state(self):
        positions=[]
        for i in range(self.numJoints):
            link_state = p.getLinkState(bodyUniqueId=self.kukaId, linkIndex=i)
            link_pos = link_state[0]
            link_orn = link_state[1]

            axis_length = 0.1

            # p.addUserDebugLine(lineFromXYZ=link_pos,
            #            lineToXYZ=[link_pos[0] + axis_length, link_pos[1], link_pos[2]],
            #            lineColorRGB=[1, 0, 0])
            
            # p.addUserDebugLine(lineFromXYZ=link_pos,
            #            lineToXYZ=[link_pos[0], link_pos[1] + axis_length, link_pos[2]],
            #            lineColorRGB=[0, 1, 0])

            # p.addUserDebugLine(lineFromXYZ=link_pos,
            #                 lineToXYZ=[link_pos[0], link_pos[1], link_pos[2] + axis_length],
            #                 lineColorRGB=[0, 0, 1])


            link_position = torch.tensor(link_pos, dtype=torch.float32)  # Ensure it's a float tensor
            link_orientation = torch.tensor(link_orn, dtype=torch.float32)
            combined = torch.cat((link_position, link_orientation), dim=0) 
            positions.append(combined)

        robot_xyz_state = torch.stack(positions,dim=0).requires_grad_(True)
        ee_state = robot_xyz_state[-1]

        return robot_xyz_state, ee_state # (7,7)
    
    def get_joint_states(self): 
        joint_angles=[]
        for i in range(self.numJoints):
            (joint_position, _,_,_) = p.getJointStateMultiDof(bodyUniqueId=self.kukaId,jointIndex=i)
            joint_angles.append(joint_position)
        return joint_angles # (7,1)

    def H_to_q(self,H): # H(10,4,4)
        b_s = H.shape[0]
        q_list = torch.zeros((b_s,7))
        for i in range(b_s):
            tf=H[i,:,:]
            position = (tf[:3,3]).tolist()
            orientation = p.getQuaternionFromEuler(self.getEulerFromMatrix(tf[:3,:3])) # torch(T_matrix)->list(euler)->list(quaternion)
        
            qt = self.p.calculateInverseKinematics(self.kukaId,
                                                        self.kukaEndEffectorIndex, 
                                                        position, 
                                                        orientation,
                                                        jointDamping=self.jd,
                                                        solver=self.ikSolver,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)
            
            q_list[i]=torch.tensor(qt,requires_grad=True)

        return q_list # (10,7)

    def test_H(self, H):
        tf = H[0,:,:]
        position = (tf[:3,3]).tolist()
        orientation = p.getQuaternionFromEuler(self.getEulerFromMatrix(tf[:3,:3])) # torch(T_matrix)->list(euler)->list(quaternion)
        rod_length = 0.05
        rod_radius = 0.01
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,radius=rod_radius, length=rod_length,rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=rod_radius, height=rod_length)
        rod_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id)
        p.setCollisionFilterGroupMask(rod_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        p.resetBasePositionAndOrientation(rod_id,position,orientation)
        
    def test_pos_orn(self,quat,index):

        position = quat[index,4:]
        orientation = quat[index,:4]
        rod_length = 0.05
        rod_radius = 0.001
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,radius=rod_radius, length=rod_length,rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=rod_radius, height=rod_length)
        rod_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id)
        p.setCollisionFilterGroupMask(rod_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        p.resetBasePositionAndOrientation(rod_id,position,orientation)

    def test_pos_orn_init(self,quat):
        position = quat[4:]
        orientation = quat[:4]
        rod_length = 0.1
        rod_radius = 0.001
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,radius=rod_radius, length=rod_length,rgbaColor=[0, 0, 1, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=rod_radius, height=rod_length)
        rod_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id)
        p.setCollisionFilterGroupMask(rod_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        p.resetBasePositionAndOrientation(rod_id,position,orientation)

    def go_to_init_state(self):
        joint_states = np.array([math.pi/2, math.pi/4, 0, -math.pi/2, 0, math.pi/4, 0])
        noise_level = 0.05
        noisy_joint_states = joint_states+np.random.normal(-noise_level, noise_level)
        
        for k in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,
                                    jointIndex=k,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=noisy_joint_states[k],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
            
        for _ in range(100): # time control 
            p.stepSimulation()
            time.sleep(0.01) # wait 
        time.sleep(0.5)

        # get quat 
        robot_xyz_state = torch.zeros((self.numJoints,7))
        for i in range(self.numJoints):
            link_state = p.getLinkState(bodyUniqueId=self.kukaId, linkIndex=i)
            link_pos = link_state[0] # x,y,z
            link_orn = link_state[1] # qx,qy,qz,qw

            axis_length = 0.1

            # p.addUserDebugLine(lineFromXYZ=link_pos,
            #            lineToXYZ=[link_pos[0] + axis_length, link_pos[1], link_pos[2]],
            #            lineColorRGB=[1, 0, 0])
            
            # p.addUserDebugLine(lineFromXYZ=link_pos,
            #            lineToXYZ=[link_pos[0], link_pos[1] + axis_length, link_pos[2]],
            #            lineColorRGB=[0, 1, 0])

            # p.addUserDebugLine(lineFromXYZ=link_pos,
            #                 lineToXYZ=[link_pos[0], link_pos[1], link_pos[2] + axis_length],
            #                 lineColorRGB=[0, 0, 1])


            link_position = torch.tensor(link_pos, dtype=torch.float32)  # Ensure it's a float tensor
            link_orientation = torch.tensor(link_orn, dtype=torch.float32)
            robot_xyz_state[i,0]=torch.tensor([link_orn[-1]])
            robot_xyz_state[i,1:4] = torch.tensor([link_orn[0:3]])
            robot_xyz_state[i,4:] = torch.tensor([link_pos[:]])
            
            
            
        ee_state = robot_xyz_state[-1]
        return noisy_joint_states, ee_state # (7,1)
    
    def quat_to_tf(self, quat):
        qw, qx, qy, qz = quat[:4]
        x, y, z = quat[4:]

        # 쿼터니언으로부터 회전 행렬 계산
        R = torch.tensor([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        # 변환 행렬 구성
        T = torch.zeros((4, 4))
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([x, y, z])
        T[3, 3] = 1.0

        return T


    def getJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques


    def getMotorJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def multiplyJacobian(self, robot, jacobian, vector):
        result = [0.0, 0.0, 0.0]
        i = 0
        for c in range(len(vector)):
            if p.getJointInfo(robot, c)[3] > -1:
                for r in range(3):
                    result[r] += jacobian[r][i] * vector[c]
                i += 1
        return result
    
    def get_jacobian(self, current_joint):
        # Get the joint and link state directly from Bullet.
        pos, vel, torq = self.getJointStates(self.kukaId)
        mpos, mvel, mtorq = self.getMotorJointStates(self.kukaId)

        jac_t_list, jac_r_list = [],[]
        for i in range(self.numJoints):
            result = p.getLinkState(self.kukaId,
                                    i,
                                    computeLinkVelocity=1,
                                    computeForwardKinematics=1) # about ee 
            link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result # com_trn : localInertialFramePosition # link_vt : worldLinkLinearVelocity # link_vr : worldLinkAngularVelocity
            zero_vec = [0.0] * len(mpos)
            jac_t, jac_r = p.calculateJacobian(self.kukaId, 
                                            i, 
                                            com_trn, mpos, zero_vec, zero_vec)
            jac_t_list.append(jac_t)
            jac_r_list.append(jac_r)
    
        # print("Link linear velocity of CoM from getLinkState:")
        # print(link_vt)
        # print("Link linear velocity of CoM from linearJacobian * q_dot:")
        # print(self.multiplyJacobian(self.kukaId, jac_t, vel))
        # print("Link angular velocity of CoM from getLinkState:")
        # print(link_vr)
        # print("Link angular velocity of CoM from angularJacobian * q_dot:")
        # print(self.multiplyJacobian(self.kukaId, jac_r, vel))

        return jac_t_list, jac_r_list

        


    def skew_symmetric(self,v):
        return torch.tensor([[0, -v[2], v[1]], 
                         [v[2], 0, -v[0]], 
                         [-v[1], v[0], 0]], requires_grad=True)
    
    def exp_map(self, omega, theta):
        omega_skew = self.skew_symmetric(omega)
        omega_matrix = omega.view(3, 1)  # 벡터를 3x1 행렬로 변환
        return torch.eye(3, requires_grad=True) + torch.sin(theta) * omega_skew + (1 - torch.cos(theta)) * omega_skew.mm(omega_matrix)

    def poE_compute_tf(self, thetas):
        screw_axes = torch.tensor([
            [0,0,1,0,0,0.1575],
            [0,-1,0,0,0,0.2025],
            [0,0,1,0,0,0.2045],
            [0,-1,0,0,0,0.2155],
            [0,0,-1,0,0,0.1845],
            [0,-1,0,0,0,0.2155],
            [0,0,1,0,0,0.081]
        ])
        T = torch.eye(4,requires_grad=True)
        T0 = torch.tensor([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,1.261],
                           [0,0,0,1]])
        T = T.mm(T0)
        for i in range(6,-1,-1):
            print(i)
            omega = screw_axes[i][:3]
            q = screw_axes[i][3:]
            v = -torch.cross(omega,q)
            R = self.exp_map(omega,thetas[i])
            wxv = torch.cross(omega,v)
            p = torch.mm((torch.eye(3, requires_grad=True) - R),(wxv.view(-1,1)))

            T_i = torch.eye(4,requires_grad=True)
            T_i = T_i.clone()
            T_i[:3,:3]=R.clone()
            T_i[:3,3]=p.squeeze()
            T = torch.mm(T_i, T)
            print(T)
        return T   

    def POE_FK(self, theta):
        w = torch.tensor([[0,0,1],[0,0,1],[0,-1,0],[0,0,1],[0,-1,0],[0,0,-1],[0,-1,0],[0,0,1]], dtype=torch.float32)
        q = torch.tensor([[0,0,0],[0,0,0.1575],[0,0,0.2025],[0,0,0.2045],[0,0,0.2155],[0,0,0.1845],[0,0,0.2155],[0,0,0.081]], dtype=torch.float32)
        T0 = torch.tensor([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,1.261],
                            [0,0,0,1]], dtype=torch.float32)
        T = self.POE(theta, q, w, T0)
        Tf = torch.mm(T, T0)

        return Tf

    def POE(self, theta, q, w, T0):

        T = torch.eye(4, dtype=torch.float32)
        n = len(theta)

        for ii in range(n - 1, -1, -1):
            w_hat = torch.tensor([[0, -w[ii, 2], w[ii, 1]],
                                [w[ii, 2], 0, -w[ii, 0]],
                                [-w[ii, 1], w[ii, 0], 0]], dtype=torch.float32)
            e_w_hat = torch.eye(3, dtype=torch.float32) + w_hat * torch.sin(theta[ii]) + torch.mm(w_hat, w_hat) * (1 - torch.cos(theta[ii]))

            if ii > 0:
                v = -torch.cross(w[ii], q[ii]).reshape(3, 1).float()
            else:
                v = torch.tensor([[0], [0], [0]], dtype=torch.float32)

            t = (torch.eye(3, dtype=torch.float32) * theta[ii] + (1 - torch.cos(theta[ii])) * w_hat + (theta[ii] - torch.sin(theta[ii])) * torch.mm(w_hat, w_hat)).mm(v)
            e_zai = torch.cat([torch.cat([e_w_hat, t], 1), torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)])
            print(e_zai)
            T = torch.mm(e_zai, T)

        return T


if __name__=='__main__':

    env = KukaVisualization()
    
    theta = torch.tensor([0, math.pi/4, math.pi/4, math.pi/4, math.pi/4, math.pi/4, 0],requires_grad=True) 
        
    Tf = env.POE_FK(theta)
    print("tf:", Tf)
    


      
    




     
     