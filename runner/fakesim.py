import numpy as np
import pickle

class fakesim:
    def __init__(self, robot_num) -> None:
        self.step_cnt = 0
        self.global_step_cnt = 0
        self.data = []
        self.robot_num = robot_num
        with open("/home/yangxy/yangxy/onpolicy/onpolicy/sim_to_real/data/mkn3data.pkl", "rb") as tt:
            readdata = pickle.load(tt)
            self.data.append(readdata)
        with open("/home/yangxy/yangxy/onpolicy/onpolicy/sim_to_real/data/mkn5data.pkl", "rb") as tt:
            readdata = pickle.load(tt)
            self.data.append(readdata)
    
    def reset(self):
        self.step_cnt = 0
        pos = []
        ratio = []
        explored = []
        explored_all = []
        obstacle = []
        map_pose = []
        for i in range(self.robot_num):
            pos.append(self.data[i]["poses"][0])
            explored.append(self.data[i]["explored_maps"][0])
            explored_all.append(self.data[i]["explored_maps_without_obstacles"][0])
            obstacle.append(self.data[i]["obstacle_maps"][0])
            map_pose.append(self.data[i]["map_poses"][0])
            ratio.append(self.data[i]["ratios"][0])
        max_size = (340, 500)

        return pos, ratio, max_size, explored, explored_all, obstacle, map_pose
    
    def step(self, global_goal):
        pos = []
        ratio = []
        explored = []
        explored_all = []
        obstacle = []
        map_pose = []
        if self.step_cnt % 15 == 14:
            self.global_step_cnt += 1
        print(self.global_step_cnt)
        for i in range(self.robot_num):
            pos.append(self.data[i]["poses"][self.step_cnt])
            explored.append(self.data[i]["explored_maps"][self.global_step_cnt])
            explored_all.append(self.data[i]["explored_maps_without_obstacles"][self.global_step_cnt])
            obstacle.append(self.data[i]["obstacle_maps"][self.global_step_cnt])
            map_pose.append(self.data[i]["map_poses"][self.global_step_cnt])
            ratio.append(self.data[i]["ratios"][self.step_cnt])

        self.step_cnt += 1        
        return pos, ratio, explored, explored_all, obstacle, map_pose


if __name__ == '__main__':
    env = fakesim(1)

    _, _, _,_,_,_,_, = env.reset()

    for i in range(120):
        pos, ratio, explored, explored_all, obstacle, map_pose = env.step(0)
        print(explored[0].shape)