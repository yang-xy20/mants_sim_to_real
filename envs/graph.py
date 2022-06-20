import math
import numpy as np
import sys
sys.path.append("../..")

from mants_sim_to_real.utils import pose as pu
from torch import nn

class Node(object):
    def __init__(self, info=None):
        self.node_num = None
        self.time_t = None
        self.neighbors = []
        self.neighbors_node_num = []
        self.embedding = None
        self.misc_info = None
        self.action = -1
        self.visited_time = []
        self.visited_memory = []
        if info is not None:
            for k, v in info.items():
                setattr(self, k, v)
    
class Graph(object):
    def __init__(self, args, memory_size, num_agents):
        self.args = args
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.M = memory_size
        self.num_agents = num_agents
        self.resolution = args.map_resolution
        
        self.ghost_node_size = args.ghost_node_size
        self.angles = [0, 30, 60, 90, 120, 150, 180, -35, -60, -90, -120, -150]

    def num_node(self):
        return len(self.node_position_list)

    def num_node_max(self):
        return self.graph_mask.sum(axis=0).max().astype(np.long)

    def reset(self):
        self.node_position_list = [] # This position list is only for visualizations
        self.ghost_node_position = np.zeros([self.M, self.ghost_node_size, 2])
        self.ghost_mask = np.zeros([self.M, self.ghost_node_size], dtype = np.int32)
        self.A = np.zeros([self.M, self.M],dtype=np.float64)
        self.graph_mask = np.zeros([self.M]) 
        self.last_localized_node_idx = np.zeros([self.num_agents], dtype=np.int32)
        self.last_local_node_num = np.zeros([1])
        self.num_init_nodes = 0
        self.flag_reset = False

    def reset_at(self):
        self.A = np.zeros([self.M, self.M],dtype=float)
        self.graph_mask = np.zeros([self.M])
        self.ghost_mask = np.zeros([self.M, self.ghost_node_size], dtype=np.int32)
        self.last_localized_node_idx = np.zeros([self.num_agents], dtype=np.int32)
        self.node_position_list = []
        self.ghost_node_position = np.zeros([self.M, self.ghost_node_size, 2])
        self.num_init_nodes = 0
        self.flag_reset = False

    def initialize_graph(self, b, positions):
        self.add_node(node_idx=b, agent_id=b,  position=positions)
        self.record_localized_state(node_idx=b, agent_id=b)

    def add_node(self, node_idx, agent_id,  position):
        self.node_position_list.append(position)
        self.graph_mask[node_idx] = 1.0

    def add_ghost_node(self, agent_id, num, position):
        for i in range(num) :
            _, is_localized = self.if_nearby(self.node_position_list, position[i])
            self.ghost_node_position[self.last_localized_node_idx[agent_id],i] = position[i][:2]
            if not is_localized: 
                self.ghost_mask[self.last_localized_node_idx[agent_id], i] = 1
                
    def record_localized_state(self, node_idx, agent_id):
        self.last_localized_node_idx[agent_id] = node_idx

    def add_edge(self, node_idx_a, node_idx_b):
        self.A[node_idx_a, node_idx_b] = 1.0
        self.A[node_idx_b, node_idx_a] = 1.0
        return

    def get_positions(self, b, a=None):
        if a is None:
            return self.node_position_list[b]
        else:
            return self.node_position_list[b][a]
    
    def get_ghost_positions(self, x):
        valid_ghost_position = self.ghost_node_position[self.ghost_mask == 1]
        index_i, index_j = np.where(self.ghost_mask == 1)
        
        position = valid_ghost_position[x]
        return position
    
    def get_all_ghost_positions(self):
        
        return self.ghost_node_position[self.ghost_mask == 1]

    def get_neighbor(self, b, node_idx, return_mask=False):
        if return_mask: return self.A[b, node_idx]
        else: return np.where(self.A[b, node_idx])[0]

    def calculate_multihop(self, hop):
        return np.matrix_power(self.A[ :self.num_node_max(), :self.num_node_max()].float(), hop)
    
    def if_nearby(self, position_A_list, position_B, ghost= None, target_dis = 0.8):
        mini_idx = []
        is_localized = False
        if type(position_A_list) == np.ndarray:
            for i in range(position_A_list.shape[0]):
                for j in range(position_A_list.shape[1]):
                    if self.ghost_mask[i,j] == 0:
                        continue
                    else:
                        dis = pu.get_l2_distance(position_A_list[i,j][0], position_B[0], \
                                                position_A_list[i,j][1], position_B[1])
                        if dis < target_dis :
                            if ghost is not None :
                                if (i,j) != ghost:
                                    is_localized = True
                                    mini_idx.append((i,j))
                            else:
                                is_localized = True
                                mini_idx.append((i,j))
        else:
            for i in range(len(position_A_list)):
                dis = pu.get_l2_distance(position_A_list[i][0], position_B[0], \
                                            position_A_list[i][1], position_B[1])
                if dis < target_dis :
                    is_localized = True
                    mini_idx.append(i)
                    
        return mini_idx, is_localized

    def check_around(self, pos):
        _, is_node = self.if_nearby(self.node_position_list, pos, target_dis=2)
        _, is_ghost = self.if_nearby(self.ghost_node_position, pos, target_dis=2)
        return not (is_node or is_ghost)

    def ghost_check(self):
        for j in range(len(self.node_position_list)):
            index, is_localized = self.if_nearby(self.ghost_node_position, self.node_position_list[j], target_dis=2)
            if is_localized:
                for i in range(len(index)):
                    self.ghost_mask[index[i][0],index[i][1]] = 0
    
        for i in range(self.ghost_node_position.shape[0]-1, -1, -1):
            for j in range(self.ghost_node_position.shape[1]-1, -1, -1):
                if self.ghost_mask[i,j] == 0:
                    continue
                else:
                    ghost_index, is_ghost_localized = self.if_nearby(self.ghost_node_position, self.ghost_node_position[i, j], ghost = (i,j), target_dis=0.5)
                    if is_ghost_localized:
                        for i in range(len(ghost_index)):
                            self.ghost_mask[ghost_index[i][0], ghost_index[i][1]] = 0
                            # self.ghost_node_position_list.pop(ghost_index[i])
                            # self.ghost_node_link.pop(ghost_index[i])
                            # self.ghost_node_feature.pop(ghost_index[i])
           

    def dijkstra(self, target, agent_id, length_only=False):#Src表示起点的编号，Dst表示终点的编号，N表示结点个数.
        source = self.last_localized_node_idx[agent_id]
        if length_only and source==target[agent_id]:
            return np.array([])
        matrix = self.A.copy()
        M = 1E100
        matrix[matrix==0] = M
        
        n = len(matrix)
        m = len(matrix[0])
        if source >= n or n != m:
            print('Error!')
            return
        found = [source]        # 已找到最短路径的节点
        cost = [M] * n          # source到已找到最短路径的节点的最短距离
        cost[source] = 0
        path = [[]]*n           # source到其他节点的最短路径
        path[source] = [source]
        target_path = None

        while len(found) < n:   # 当已找到最短路径的节点小于n时
            min_value = M+1
            col = -1
            row = -1
            for f in found:     # 以已找到最短路径的节点所在行为搜索对象
                for i in [x for x in range(n) if x not in found]:   # 只搜索没找出最短路径的列
                    if matrix[f][i] + cost[f] < min_value:  # 找出最小值
                        min_value = matrix[f][i] + cost[f]  # 在某行找到最小值要加上source到该行的最短路径
                        row = f         # 记录所在行列
                        col = i
            if col == -1 or row == -1:  # 若没找出最小值且节点还未找完，说明图中存在不连通的节点
                break
            found.append(col)           # 在found中添加已找到的节点
            cost[col] = min_value       # source到该节点的最短距离即为min_value
            path[col] = path[row][:]    # 复制source到已找到节点的上一节点的路径
            path[col].append(col)       # 再其后添加已找到节点即为sorcer到该节点的最短路径
            if col == int(target[agent_id]):
                target_path = path[col]
        #found, cost, path,
        return  target_path

    def check_ghost_outside(self, max_size, left_corner, map_in, explored_in, explored_all, ratio, resolution):
        
        merge_map_in = np.zeros((max_size[0],max_size[1]))
        merge_explored_in = np.zeros((max_size[0],max_size[1]))
        merge_explored_all = np.zeros((max_size[0],max_size[1]))
        map_in_shape = map_in.shape
        explored_in_shape = explored_in.shape
        explored_all_shape = explored_all.shape
        merge_map_in[left_corner[0]:left_corner[0]+map_in_shape[0],left_corner[1]:left_corner[1]+map_in_shape[1]] = map_in
        merge_explored_in[left_corner[0]:left_corner[0]+explored_in_shape[0],left_corner[1]:left_corner[1]+explored_in_shape[1]] = explored_in
        merge_explored_all[left_corner[0]:left_corner[0]+explored_all_shape[0],left_corner[1]:left_corner[1]+explored_all_shape[1]] = explored_all
       
        ratio = np.array(ratio)
        # if len(self.node_position_list) == self.num_init_nodes:
        #     pass
        # elif np.any(ratio < 0.3) and (not self.flag_reset):
        #     self.last_ghost_mask = self.ghost_mask[:self.num_init_nodes].copy()
        #     self.ghost_mask[:self.num_init_nodes,:] = 0
        #     self.flag_reset = True
        # elif np.all(ratio >= 0.3) and self.flag_reset:
        #     self.ghost_mask[:self.num_init_nodes] = self.last_ghost_mask

        #ghost_link = self.ghost_node_link
        ghost_pos = self.ghost_node_position
        world_pos = self.node_position_list
        #del_list = []
        for idx in range(ghost_pos.shape[0]):
            for idy in range(ghost_pos.shape[1]):
                if self.ghost_mask[idx,idy] == 0:
                    continue
                else:
                    curr_loc = world_pos[idx]
                    x, y = round(curr_loc[0]*100/resolution), round(curr_loc[1]*100/resolution)
                    xx, yy = round(ghost_pos[idx,idy][0]*100/resolution),round(ghost_pos[idx,idy][1]*100/resolution) 
                   
                    dx = x-xx
                    dy = y-yy
                    ref = max(abs(dx), abs(dy))
                    step_x = dx/ref
                    step_y = dy/ref
                    if abs(step_x) > abs(step_y):
                        for i in range(0,abs(dx)+1):
                            tempx, tempy = round(xx+i*step_x), round(yy+i*step_y)
                            if merge_explored_in[tempx, tempy-1] > 0.5 and merge_explored_in[tempx, tempy] > 0.5 and merge_explored_in[tempx, tempy+1] > 0.5 and \
                            merge_explored_in[round(tempx+1), round(tempy-1)] > 0.5 and merge_explored_in[round(tempx+1), round(tempy)] > 0.5 and merge_explored_in[round(tempx+1), round(tempy+1)] > 0.5 and \
                            merge_explored_in[round(tempx-1), round(tempy-1)] > 0.5 and merge_explored_in[round(tempx-1), round(tempy)] > 0.5 and merge_explored_in[round(tempx-1), round(tempy+1)] > 0.5:
                                break
                            elif merge_map_in[tempx, tempy] > 0:
                                self.ghost_mask[idx,idy] = 0
                                count = 0
                                for j in range(i+1,abs(dx)+1):
                                    if merge_explored_all[round(xx+j*step_x)-1:round(xx+j*step_x)+2, round(yy+j*step_y)-1:round(yy+j*step_y)+2].sum()>4:
                                        count += 1
                                    if count > 0.6*(abs(dx)-i):
                                        self.ghost_mask[idx,idy] = 0
                                        break
                                break
                    else:
                        for i in range(0,abs(dy)+1):
                            tempx, tempy = round(xx+i*step_x), round(yy+i*step_y)
                            if merge_explored_in[tempx, tempy-1] > 0.5 and merge_explored_in[tempx, tempy] > 0.5 and merge_explored_in[tempx, tempy+1] > 0.5 and \
                            merge_explored_in[round(tempx+1), round(tempy-1)] > 0.5 and merge_explored_in[round(tempx+1), round(tempy)] > 0.5 and merge_explored_in[round(tempx+1), round(tempy+1)] > 0.5 and \
                            merge_explored_in[round(tempx-1), round(tempy-1)] > 0.5 and merge_explored_in[round(tempx-1), round(tempy)] > 0.5 and merge_explored_in[round(tempx-1), round(tempy+1)] > 0.5:
                                break
                            elif merge_map_in[tempx, tempy] > 0:
                                self.ghost_mask[idx,idy] = 0
                                count = 0
                                for j in range(i+1,abs(dy)+1):
                                    if merge_explored_all[round(xx+j*step_x)-1:round(xx+j*step_x)+2, round(yy+j*step_y)-1:round(yy+j*step_y)+2].sum()>4:
                                        count += 1
                                    if count > 0.6*(abs(dy)-i):
                                        self.ghost_mask[idx,idy] = 0
                                        break
                                break
    def merge_ghost_nodes(self, ghost_mask):
        self.ghost_mask *= ghost_mask
