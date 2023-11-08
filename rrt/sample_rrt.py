import numpy as np
import copy 
import torch
import scipy.spatial.transform
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib
import sys 

from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.utils import to_torch, to_numpy
from se3dif.samplers import Grasp_AnnealedLD
from se3dif.visualization import grasp_visualization
from se3dif.rrt.rrt_star import RobotArm, RRTStar
from se3dif.rrt.generate_rrt import get_approximated_grasp_diffusion_field, sample_pointcloud

show_animation = True
verbose = True
n_grasps = 10
obj_id = 10
obj_class = 'Mug'
model = 'grasp_dif_mugs'
device = 'cpu'
n_steps = 70


P, mesh, trans, _ = sample_pointcloud(obj_id, obj_class)
generator, model = get_approximated_grasp_diffusion_field(P, model, device)
H,trj_H = generator.sample(save_path=True)
vis_trj_H = trj_H[-n_steps:,0,:,:] # (70,4,4) original : 121 
vis_trj_H[:,:3,-1] *= 1/8.
H_grasp = copy.deepcopy(H)
H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
H[..., :3, -1] *=1/8.
H_grasp[..., :3, -1] *=1/8.

P *=1/8


print("finish sampling")
#print(to_numpy(vis_trj_H),P)



# [theta, alpha, a, d]
seven_joint_arm = RobotArm([[0., math.pi/2., 0., .333],
                            [0., -math.pi/2., 0., 0.],
                            [0., math.pi/2., 0.0825, 0.3160],
                            [0., -math.pi/2., -0.0825, 0.],
                            [0., math.pi/2., 0., 0.3840],
                            [0., math.pi/2., 0.088, 0.],
                            [0., 0., 0., 0.107]])

all_joint_angles = []
new_pose_x,new_pose_y,new_pose_z = [],[],[]
for i in range(len(vis_trj_H)):

    last_trj_H = vis_trj_H[i,:,:].squeeze() #(4,4)
    new_position = last_trj_H[0:3,3].tolist() # (x,y,z)
    new_pose_x.append(new_position[0]), new_pose_y.append(new_position[1]), new_pose_z.append(new_position[2])
    new_orientation = seven_joint_arm.euler_angle(target_H=last_trj_H) # (row,pitch,yaw)
    new_ref_ee_pose = np.concatenate((new_position, new_orientation))

    seven_joint_arm.inverse_kinematics(ref_ee_pose=new_ref_ee_pose,plot=False)
    new_joint_angles = [link.dh_params_[0] for link in seven_joint_arm.link_list]
    all_joint_angles.append(new_joint_angles)
print("inverse kinematics done")
# raw 50 waypoints animation in 3D xyz coordinate

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(P[:,0],P[:,1],P[:,2],c='b',marker='.', alpha=0.3, label='Point Cloud')
ax.scatter(new_pose_x, new_pose_y, new_pose_z, c='r', marker='o', alpha=0.3,label='Waypoints')
# Label the axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Show legend
ax.legend()

# Show the plot
plt.show()

# ====Search Path with RRT====
obstacle_list = [
    # (-.3, -.3, .7, .1),
    # (.0, -.3, .7, .1),
    # (.2, -.1, .3, .15),
]  # [x,y,z,size(radius)]
# start = [0 for _ in range(len(seven_joint_arm.link_list))] # 7 
# end = [1.5 for _ in range(len(seven_joint_arm.link_list))]
# Set Initial parameters
start = all_joint_angles[0]
end = all_joint_angles[-1]
print(start, end)
# Draw init, final robot config 
rrt_star = RRTStar(start=start,
                    goal=end,
                    rand_area=[0, 2],
                    max_iter=200,
                    robot=seven_joint_arm,
                    obstacle_list=obstacle_list)


path = rrt_star.planning(animation=show_animation,
                            search_until_max_iter=False)
print(path)
if path is None:
    print("Cannot find path")
else:
    print("found path!!")

    # Draw final path
    if show_animation:
        ax = rrt_star.draw_graph()

        # Plot final configuration
        ax.scatter(P[:,0],P[:,1],P[:,2],c='b',marker='.', alpha=0.3, label='Point Cloud')
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


# rrt_path = [all_joint_angles[0]]
# current_start = all_joint_angles[0]

# for waypoint in all_joint_angles[1:]:
#     subgoal = waypoint
#     rrt_star = RRTStar(start=current_start,
#                         goal=subgoal,
#                         rand_area=[0, 2],
#                         max_iter=100,
#                         robot=seven_joint_arm,
#                         obstacle_list=obstacle_list)
#     subpath = rrt_star.planning(animation=show_animation,
#                                 search_until_max_iter=False)
    
#     if subpath is None:
#         print("cannot find path")
#         break
#     else:
#         rrt_path.extend([copy.deepcopy(node) for node in subpath[1:]])
#         current_start = subpath[-1]

# #path  =[node for subpath in rrt_path for node in (subpath if isinstance(subpath, list) else [subpath])]
# path=rrt_path
# print((path))
# print(len(path))
# if path is None:
#     print("Cannot find path")
# else:
#     print("found path!!")

#     # Draw final path
#     if show_animation:
#         ax = rrt_star.draw_graph()

#         # Plot final configuration
#         x_points, y_points, z_points = seven_joint_arm.get_points(path[-1])
#         ax.plot([x for x in x_points],
#                 [y for y in y_points],
#                 [z for z in z_points],
#                 "o-", color="red", ms=5, mew=0.5)
#         ax.scatter(P[:,0],P[:,1],P[:,2],c='b',marker='.', alpha=0.3, label='Point Cloud')
#         ee_x,ee_y,ee_z = [],[],[]
#         for i, q in enumerate(path):
#             x_points, y_points, z_points = seven_joint_arm.get_points(q)
#             # ax.plot([x for x in x_points],
#             #         [y for y in y_points],
#             #         [z for z in z_points],
#             #         "o-", color="grey",  ms=4, mew=0.5)
#             # ax.plot([x_points[-1]],
#             #         [y_points[-1]],
#             #         [z_points[-1]],
#             #         "o-", color="blue",  ms=4, mew=0.5)
#             ee_x.append(x_points[-1])
#             ee_y.append(y_points[-1])
#             ee_z.append(z_points[-1])
#             plt.pause(0.01)


#         ax.plot(ee_x,ee_y,ee_z,color="red", linewidth=2, label="End-Effector Path")
#         plt.show()