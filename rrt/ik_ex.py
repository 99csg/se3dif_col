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


        #table config
        # self.table_id = p.loadURDF("table/table.urdf")
        # table_pos = [0,0,-0.65]
        # table_orientation = p.getQuaternionFromEuler([0,0,0])
        # p.resetBasePositionAndOrientation(self.table_id, table_pos,table_orientation)

        p.setGravity(0, 0, 0)
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
        ob_list = self.call_object(num,mug_center,rad) # [[3, [0.584874501884813, 0.46591183056858754, 0.5685968345463284]], [4, [0.27034844244277056, 0.5943483417653938, 0.6457106129726388]], [5, [0.43319756030566836, 0.6125678578395343, 0.6639166800981371]]]
       
        #ob_list = None
        return ob_list

    def move(self,Ht=None):
        
        tf = Ht[0,:,:] # (4,4)
        position = (tf[:3,3]).tolist()
           

        orientation = p.getQuaternionFromEuler(self.getEulerFromMatrix(tf[:3,:3])) # torch(T_matrix)->list(euler)->list(quaternion)
        print(position, orientation)
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
        time.sleep(0.5)

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

        # euler_angles = [x,y,z]
        rot = Rotation.from_matrix(matrix)
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
    
    def check_table_collision(self):
        pass
    
    def check_obj_collision(self, ob_list,robot_xyz_state):
        if ob_list is None or len(ob_list)==0:
            return torch.tensor(0.0)
        
        total_cost = torch.tensor(0.0)
        for ob_data in (ob_list):
            id, ob_xyz = ob_data[0],torch.tensor(ob_data[1])
            cost = torch.sum((robot_xyz_state-ob_xyz)**2,dim=1)
            #cost /= (self.object_edge_radius**2)
            total_cost += torch.exp(-0.5*cost).sum()
        
        return total_cost 
    
    def compute_fk(self,theta):
        theta = torch.tensor(theta, requires_grad=True)
        alpha = torch.tensor([-math.pi/2, math.pi/2, math.pi/2,
                              -math.pi/2,-math.pi/2,math.pi/2, 0])
        d = torch.tensor([0.3105, 0, 0.4, 0, 0.39, 0, 0.078])
        a = torch.tensor([0,0,0,0,0,0,0])
        T = torch.eye(4, requires_grad=True)
        T_list,dT_list = [],[]
        for i in range(7):
            T0 = torch.tensor([[torch.cos(theta[i]), -torch.sin(theta[i])*torch.cos(alpha[i]), torch.sin(theta[i])*torch.sin(alpha[i]), a[i]*torch.cos(theta[i])],
                                [torch.sin(theta[i]), torch.cos(theta[i])*torch.cos(alpha[i]), -torch.cos(theta[i])*torch.sin(alpha[i]), a[i]*torch.sin(theta[i])],
                                [0,torch.sin(alpha[i]), torch.cos(alpha[i]), d[i]],
                                [0,0,0,1]], requires_grad=True)
            dT0 = torch.tensor([[-torch.sin(theta[i]), -torch.cos(theta[i])*torch.cos(alpha[i]), torch.cos(theta[i])*torch.sin(alpha[i]), -a[i]*torch.sin(theta[i])],
                                [torch.cos(theta[i]), -torch.sin(theta[i])*torch.cos(alpha[i]), torch.sin(theta[i])*torch.sin(alpha[i]), a[i]*torch.cos(theta[i])],
                                [0,0, 0, 0],
                                [0,0,0,0]], requires_grad=True)
            T_list.append(T0)
            dT_list.append(dT0)

        T = T_list[0]@T_list[1]@T_list[2]@T_list[3]@T_list[4]@T_list[5]@T_list[6]
        dH_dq = torch.zeros((7,4,4))
        
        # dH_dq[0] = torch.tensor([dT_list[0]@T_list[1]@T_list[2]@T_list[3]@T_list[4]@T_list[5]@T_list[6]])
        # dH_dq[1] = torch.tensor([T_list[0]@dT_list[1]@T_list[2]@T_list[3]@T_list[4]@T_list[5]@T_list[6]])
        # dH_dq[2] = torch.tensor([T_list[0]@T_list[1]@dT_list[2]@T_list[3]@T_list[4]@T_list[5]@T_list[6]])
        # dH_dq[3] = torch.tensor([T_list[0]@T_list[1]@T_list[2]@dT_list[3]@T_list[4]@T_list[5]@T_list[6]])
        # dH_dq[4] = torch.tensor([T_list[0]@T_list[1]@T_list[2]@T_list[3]@dT_list[4]@T_list[5]@T_list[6]])
        # dH_dq[5] = torch.tensor([T_list[0]@T_list[1]@T_list[2]@T_list[3]@T_list[4]@dT_list[5]@T_list[6]])
        # dH_dq[6] = torch.tensor([T_list[0]@T_list[1]@T_list[2]@T_list[3]@T_list[4]@T_list[5]@dT_list[6]])

        # ee = T[:3,3]
        # jacobian = torch.zeros((3,len(theta)))
        # for i in range(len(theta)):
        #     for j in range(3):
        #         grad = torch.autograd.grad(ee[j], theta, allow_unused=True)[0]
        #         jacobian[j,i]=grad if grad is not None else 0


        
        return T, dH_dq 
    
    def compute_dh_dH(self,H):
        H0 = SO3_R3(R=H[:,:3,:3],t=H[:,:3,-1])
        h0 = H0.log_map()

    
    def get_xyz_state(self):
        positions=[]
        for i in range(self.numJoints):
            link_state = p.getLinkState(bodyUniqueId=self.kukaId, linkIndex=i)
            link_position = torch.tensor(link_state[0], dtype=torch.float32)  # Ensure it's a float tensor
            positions.append(link_position)

        robot_xyz_state = torch.stack(positions,dim=0).requires_grad_(True)
        return robot_xyz_state
    
    def get_joint_states(self):
        joint_angles=[]
        for i in range(self.numJoints):
            (joint_position, _,_,_) = p.getJointState(bodyUniqueId=self.kukaId,jointIndex=i)
            joint_angles.append(joint_position)
        return joint_angles

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

if __name__=='__main__':


      env = KukaVisualization()
      theta = [math.radians(30),math.radians(30),math.radians(30),math.radians(30),math.radians(30),math.radians(30),math.radians(30)]
      jacobian = env.compute_fk(theta)
      print(jacobian)


      
    




     
     