#目前未引入蓝队
#未加入玩家操作
#未加入时序特征
#未完成训练设计

import pygame
import sys
import numpy as np
import torch

# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 25, 0)
BULE = (0, 0, 225)
DARK_GREEN = (0, 120, 0)
BLACK = (0, 0, 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')##
modes = ['train',
         'practice',
         'play',
         'test',
         'pre_train']
mode = modes[0]
if mode != 'train' and mode != 'practice' and mode != 'test' and mode != 'pre_train':
    raise ValueError('mode config error')

class point_state:##
    def __init__(self,tp:int,pos1:tuple = None,pos2:tuple = None,pos3:tuple = None):
        self.type = tp
        self.speed = torch.tensor([0,0],device = device , dtype = torch.int8)
        self.accelerate_speed = torch.tensor([0,0],device = device , dtype = torch.int8)
        self.last_pos =torch.tensor([0,0],device = device , dtype = torch.int16)
        if self.type == 1:
            if pos1:
                self.pos = torch.tensor(pos1,device = device , dtype = torch.int16)
            else:
                self.pos = torch.tensor([ground_config["width"] / 2 , ground_config["height"] / 2],device = device , dtype = torch.int16)
        elif self.type == 2:
            if pos2:
                self.pos = torch.tensor(pos2,device = device , dtype = torch.int16)
            else:
                self.pos = torch.tensor([ground_config["side_width"] , ground_config["height"] / 2],device = device , dtype = torch.int16)
        elif self.type == 3:            
            if pos3:
                self.pos = torch.tensor(pos3,device = device , dtype = torch.int16)
            else:
                self.pos = torch.tensor([ground_config["width"] - ground_config["side_width"] , ground_config["height"] / 2],device = device , dtype = torch.int16)
        else:
            raise ValueError("Invalid type of point")

# 设置
##考虑到窗口为像素显示，只能显示整数位置
##于是速度也限制为整数，时间步长和总时间限制为整数
##进一步的，可以将加速度也限制为整数

ground_config = {
    #长宽不应超过屏幕大小
    "width": 800,
    "height": 600,
    "side_width": 50,
    "goal_width": 100,
    "resistance": 0.05  ##
}
weight = [1,10,10]
size= [10,20,20]
point_config = {
    "point1":{
        "type":1,
        "size": size[0],         
        #对于圆，size等于半径
        "color":YELLOW,
        "weight": weight[0],
        "speed_limit": 200 
        #最高速度不能超过场地宽度的一半，否则会发生错误
    },
    "point2":{
        "type":2,
        "size": size[1],
        "color":RED,
        "weight": weight[1],
        "speed_limit": 50,
        "accelerate_limit":20 / weight[1]
    },
    "point3":{
        "type":3,
        "size": size[2],
        "color":BULE,
        "weight": weight[2],
        "speed_limit": 70,
        "accelerate_limit": 30 / weight[2]
    }
}

if point_config["point1"]["size"] <= 0 or point_config["point2"]["size"] <= 0 or point_config["point3"]["size"] <= 0:
    raise ValueError("point size config error")
if point_config["point1"]["weight"] <= 0 or point_config["point2"]["weight"] <= 0 or point_config["point3"]["weight"] <= 0:
    raise ValueError("point weight config error")
if (point_config["point1"]["speed_limit"] <= 0 or point_config["point2"]["speed_limit"] <= 0 
    or point_config["point3"]["speed_limit"] <= 0 
    or point_config["point1"]["speed_limit"] > ground_config["width"] / 2 - ground_config["side_width"]):
    raise ValueError("point speed limit config error")
if (point_config["point2"]["accelerate_limit"] <= 0 or point_config["point3"]["accelerate_limit"] <= 0):
    raise ValueError("point accelerate limit config error")

time_step = 1 #不可更改
time_limit = 1000 #必须为 time_step 的整数倍，固定time_step后，需保证为整数
if time_limit % time_step:
    raise ValueError("time config error")

event_reward = {
    "goal":10,
    "be_goaled": -10,
    "ball_out_of_boundary": -7,
    "point_out_of_boundary": -10,
    "collide": -3,
    "kick_ball": 3,
    "out_of_time_limit": -10
}
def state_reward(state1:point_state,state2:point_state,reward_coefficient :float= 0.0001)->float:
    #状态奖励
    if state1.type != 1 or state2.type != 2:
        raise ValueError("Invalid type of point")
    else:
        return reward_coefficient * torch.norm(state1.pos - state2.pos,p = 2)
    
#最好保证resistance_coefficient为整数，便于计算加速度等
resist_coefficient1 = ground_config["resistance"] * (point_config["point1"]["size"] ** 2) / weight[0],
resist_coefficient2 = ground_config["resistance"] * (point_config["point2"]["size"] ** 2) ,

#窗口参数
width = ground_config["width"]
height = ground_config["height"]
top = ground_config["height"] - ground_config["side_width"]
bottom = ground_config["side_width"]
left = ground_config["side_width"]
right = ground_config["width"] - ground_config["side_width"]
mid = width/2
goal_top = height / 2 + ground_config["goal_width"] / 2
goal_bottom = height / 2 - ground_config["goal_width"] / 2

# 初始化Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("点的坐标变化")

def collide ( m1 :float, m2:float, v1:float, v2:float, e:float =1.0)-> tuple:
    """
    计算两个质点在碰撞后的速度矢量
    
    参数:
    m1, m2 - 两个质点的质量 (kg)
    v1, v2 - 碰撞前的速度矢量 (m/s) [作为numpy数组]
    e - 恢复系数 (0 ≤ e ≤ 1), 默认为1(完全弹性碰撞)
    
    返回:
    v1_new, v2_new - 碰撞后的速度矢量 (m/s) [numpy数组]
    """
    # 验证输入
    if not (0 <= e <= 1):
        raise ValueError("恢复系数e必须在0和1之间")
    
    if m1 <= 0 or m2 <= 0:
        raise ValueError("质量必须为正数")
    
    # 计算相对速度
    v_rel = v2 - v1
    
    # 计算碰撞方向单位向量
    distance = np.linalg.norm(v_rel)
    if distance == 0:  # 如果两质点没有相对速度，不会发生碰撞
        return v1.copy(), v2.copy()
    n = v_rel / distance
    
    # 计算沿碰撞方向的初始速度分量
    v1n = np.dot(v1, n)
    v2n = np.dot(v2, n)
    
    # 计算碰撞后的法向速度分量
    v1n_new = (m1*v1n + m2*v2n + m2*e*(v2n - v1n)) / (m1 + m2)
    v2n_new = (m1*v1n + m2*v2n + m1*e*(v1n - v2n)) / (m1 + m2)
    
    # 计算切向速度分量(保持不变)
    v1t = v1 - v1n * n
    v2t = v2 - v2n * n
    
    # 合成最终速度
    v1_new = v1t + v1n_new * n
    v2_new = v2t + v2n_new * n
    
    return v1_new, v2_new
##
##注意consume权重应远小于event
def consume(state:point_state)->float:
    """
    返回该状态下动力的大小，代表消耗
    """
    if state.type != 2:
        raise ValueError("Invalid type of point")
    return torch.norm(state.accelerate_speed * point_config["point2"]["weight"] + torch.norm(state.speed, p = 2) * state.speed * resist_coefficient2 , p = 2)

def ball_act(state:point_state) -> point_state:
    if state.type != 1:
        raise ValueError("Invalid type of point")
    '''state.pos += state.speed * time_step
    state.speed += state.accelerate_speed * time_step
    state.accelerate_speed = - resist_coefficient1 * state.speed * torch.norm(state.speed, p = 2)   # f = kv^2
'''    
    state.last_pos = state.pos
    a1 = - resist_coefficient1 * state.speed * torch.norm(state.speed, p = 2)
    state.pos += state.speed #* time_step
    v1 += a1 #* time_step
    state.accelerate_speed = - resist_coefficient1 * v1 * torch.norm(v1, p = 2)
    state.speed += (a1 + state.accelerate_speed) / 2
    return state

#大规模并行生成
def rand_n_vectors(max_norm:float,len:int =1,device = device):
    theta = torch.rand(len,device=device)
    r = torch.randn(len,device=device)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    vectors = torch.stack((x,y),dim = 1)
    return vectors

def probability_calculate(probability:torch.tensor,epsilon):
    #probability已归一化
    if epsilon >=1 or epsilon<0:
        raise ValueError("epsilon must be in [0,1)")
    size = probability.size(-1)
    result = probability * (1-epsilon) + epsilon/size
    return result
    #未归一化：
    result = probability / probability.sum(dim = (-1,-2),keepdim=True) * (1- epsilon) + epsilon / size
    #具体沿哪个维度归一化有待确认

def agent_act(state:point_state):
    state.pos += state.speed #* time_step
    state.speed += state.accelerate_speed #* time_step
    state.accelerate_speed = 

class game_state:
    def __init__(self,agent_num:int = 1,player:bool = False):
        if agent_num<=0 or not isinstance(agent_num,int):
            raise ValueError('agent number error')
        self.state1 = point_state(1)
        self.agent_num = agent_num
        self.player = player
        self.state2 = []
        for i in range(agent_num):
            self.state2.append(point_state(2,pos2 = (left,bottom + (top - bottom) * (i+1) / (agent_num+1))))
        if player:
            self.state3 = point_state(3)
    
    def out_of_boundary_check(state:point_state)->bool:
        #在界内返回0
        return state.pos[0]< left or state.pos[0]> right or state.pos[1] > bottom or state.pos[1] > top
    
    def kick_check(ballstate:point_state,agentstate:point_state)->bool:
        if ballstate.type != 1 or agentstate.type != 2:
            raise ValueError("Invalid type of point")
        #球和点的距离小于半径
        return torch.norm(ballstate.pos - agentstate.pos, p = 2) <= point_config["point2"]["size"] + point_config["point1"]["size"]
    
    def collide_check(state1:point_state,state2:point_state)->bool:
        #返回：0 未发生 1 发生
        if (state1.type != 2 and state1.type != 3) or (state2.type != 2 and state2.type != 3):
            raise ValueError("Invalid type of point")
        #距离小于半径
        return torch.norm(state1.pos - state2.pos, p = 2) <= (size[state1.type-1] + size[state2.type-1])
    
    def goal_check(state:point_state)->int:
        #返回：0 未发生 1 进1方 2 进2方
        #当检测出ball_out_of_boundary再调用
        if state.type != 1:
            raise ValueError("Invalid type of point")
        p1 = state.last_pos
        p2 = state.pos
        if p1[0] < mid:
            type = 1
            p3 = torch.tensor([left, goal_top], device=device, dtype=torch.int16)
            p4 = torch.tensor([left, goal_bottom], device=device, dtype=torch.int16)
        else:
            type = 2
            p3 = torch.tensor([right, goal_top], device=device, dtype=torch.int16)
            p4 = torch.tensor([right, goal_bottom], device=device, dtype=torch.int16)
        ##快速排除
        if (max(p1[0], p2[0]) < min(p3[0], p4[0]) 
            or max(p3[0], p4[0]) < min(p1[0], p2[0]) 
            or max(p1[1], p2[1]) < min(p3[1], p4[1])
            or max(p3[1], p4[1]) < min(p1[1], p2[1])):
            return False
         # 计算 p1p2 与垂直线 x = x_vertical 的交点 y
        # 直线方程：y = k * (x - p1.x) + p1.y

        y_min = min(p3[1], p4[1])
        y_max = max(p3[1], p4[1])
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])  # 斜率
        y_intersect = k * (p3[0] - p1[0]) + p1[1]
        # 检查交点是否在 p3p4 的 y 范围内
        return (y_min <= y_intersect <= y_max) * type
    
    def check_state(self)->dict:
        #init
        result = {
            'agent_out_of_boundary':[],
            #返回编号
            'ball_out_of_boundary':False,
            'collide':[], 
            #碰撞双方的编号，player 编号 -1
            'kick':[],
            'goal':[False,False]    #agent--player(agent2)
        }
        #check
        for i in range(self.agent_num):
            if self.out_of_boundary_check(self.state2[i]):
                result["agent_out_of_boundary"].append(i)
        if self.out_of_boundary_check(self.state1):
            if self.goal_check(self.state1) == 1:
                result["goal"] = [True, False]
            elif self.goal_check(self.state1) == 2:
                result["goal"] = [False, True]
            else:
                result["ball_out_of_boundary"] = True
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if self.collide_check(self.state2[i], self.state2[j]):
                    result["collide"].append((i, j))
            if self.player and self.collide_check(self.state2[i], self.state3):
                result["collide"].append((i, -1))
        for i in range(self.agent_num):
            if self.kick_check(self.state1, self.state2[i]):
                result["kick"].append(i)
        return result

    def MC_forward(self):
        #奖励以0为基准
        reward = []
        for i in range(self.agent_num):
            reward.append(0)
        for n in range(time_limit):
            '''
            'agent_out_of_boundary':[],
            'ball_out_of_boundary':False,
            'collide':[],  #agent--agent or player
            #碰撞双方的编号，player 编号 -1
            'kick_occur':[False],
            'goal':[False,False]    #agent--player(agent2)
            '''
            state = self.check_state()
            illegal = False
            if state['agent_out_of_boundary']:
                for i in state['agent_out_of_boundary']:
                    reward[i] += event_reward["point_out_of_boundary"]
                illegal = True
            if state['ball_out_of_boundary']:
                reward += event_reward["ball_out_of_boundary"]
                illegal = True
            if illegal:
                break
            if state['goal'][0]:
                reward += event_reward["be_goaled"]
                break
            if state['goal'][1]:
                reward += event_reward["goal"]
                break
            if state['collide']:
                for i, j in state['collide']:
                    if j == -1:
                        reward[i] += event_reward["collide"]
                    else:
                        reward[i] += event_reward["collide"]
                        reward[j] += event_reward["collide"]
            if state['kick']:
                for i in state['kick']:
                    reward[i] += event_reward["kick_ball"]
            for i in range(self.agent_num):
                reward[i] -= consume(self.state2[i])
            #更新状态
            self.state1 = ball_act(self.state1)
            pass
        
    def MC_backward(self):
        pass

def pi(state:game_state,action_startgy:torch.tensor):
    if(mode == 'train'):
        action_probability = action_probability[state.state1]
    elif(mode == 'practice'):
        action = torch.amax(action_startgy)
    pass


"""
Actor: 
策略：对应状态下选择某动作的概率
 \epsilon -> any  
 (1- \epsilon) * P(a) -> a
 超参数：\epsilon
 超参数：学习率
"""
'''
Critic
值函数：
特征提取：全连接/卷积？
    偏置 + 权重 + 卷积核
求价值：
    特征->价值
    学习对象：样本过程
超参数：
    折扣因子
    学习率
    价值的实际计算方法及参数
'''

'''
特征提取：

预计传入：
config:all
状态：
时序为t
pos:all
speed:all
acelerate:self  (or all?)
动作：
时序t+1
accelerate_new:self

Critic价值估计
输入提取到的特征，输出预估价值
'''

'''
Critic预训练：

得到样本：
    多次模拟MC
    随机动作
    得到状态-动作-价值的样本

    储存：稀疏矩阵
        集合
随机初始化
使用样本训练
'''
'''
Actor初始化：
    平均
状态传入：
    完整状态 or 特征？
'''



def prune(array:torch.tensor,method:bool = 0,threhold1: float = 0.01,threhold2: int = 50)->torch.tensor:
    if method == 0:
        threhold = threhold1
    elif method == 1:
        threhold = torch.kthvalue(array,threhold2)
    
    pass


action_startgy = torch.tensor()

"""action_startgy = torch.tensor()"""

'''
state -> possible actions -> check ->action_value -> next state
'''
# 游戏主循环
'''running = True
while running:
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # 清屏
    screen.fill(WHITE)
    
    # 在这里绘制图形
    
    # 更新显示
    pygame.display.flip()'''

# 退出Pygame
pygame.quit()
sys.exit()

