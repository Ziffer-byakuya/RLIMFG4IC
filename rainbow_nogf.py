print("Job started")
import math, random

# import gym
import LS_env as env

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from layers import NoisyLinear
from replay_buffer import ReplayBuffer
from hotgraph import Hot_Graph

import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')
print("package included")

USE_CUDA = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(0)
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

init_state = env.initial_state
act_dim = 20
state_dim = init_state.shape[0]

num_atoms = 51
Vmin = -200
Vmax = 0
SUPPORT = torch.linspace(Vmin, Vmax, num_atoms).cuda()

gamma = 0.995
lr = 1e-4
rb_size = 12000

episodes = 100
num_frames = 1000

test_per_episodes = 1
test_num_frames = 400

batch_size = 128

side_info_batch_size = 128*2

target_update_freq = 100
update_per_step = 1
update_times = 1

side_info_scale = 0
side_use_episode = 0
side_use_episode= episodes if side_use_episode>episodes else side_use_episode



if not side_info_scale:
    update_per_step_side_info = 2000
    update_times_side_info = 0
else:
    update_per_step_side_info = 1
    update_times_side_info = 1

class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax,gamma):
        super(RainbowDQN, self).__init__()
        
        self.num_inputs   = num_inputs
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        self.hidden = 512
        self.gamma = gamma
        self.epsilon = 0
        self.init_epsilon = self.epsilon
        self.linear1 = nn.Linear(num_inputs, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear3 = nn.Linear(self.hidden, self.hidden)
        self.linear4 = nn.Linear(self.hidden, self.hidden)
        self.linear5 = nn.Linear(self.hidden, self.hidden)
        
        self.noisy_value1 = NoisyLinear(self.hidden, self.hidden, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(self.hidden, self.num_atoms, use_cuda=USE_CUDA)
        
        self.noisy_advantage1 = NoisyLinear(self.hidden, self.hidden, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(self.hidden, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

        # self.noisy_value1 = nn.Linear(self.hidden, self.hidden)
        # self.noisy_value2 = nn.Linear(self.hidden, self.num_atoms)
        
        # self.noisy_advantage1 = nn.Linear(self.hidden, self.hidden)
        # self.noisy_advantage2 = nn.Linear(self.hidden, self.num_atoms * self.num_actions)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        value     = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        
        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x,dim=-1)
        
        return x
    
    def act(self, state, mh_dqn):
        rand_num = np.random.random()
        if rand_num >= self.epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            dist = self.forward(state).data.cpu()
            
            dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
            q_value = dist.sum(2)
            action = q_value.max(1)[1].numpy()[0]
        else:
            action = np.random.randint(self.num_actions)
        return action

class MHDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, gamma, num_head):
        super(MHDQN, self).__init__()
        
        self.num_inputs   = num_inputs
        self.num_actions  = num_actions
        self.hidden = 256
        self.gamma = gamma
        self.num_head = num_head
        self.linear1 = nn.Linear(num_inputs, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear3 = nn.Linear(self.hidden, self.hidden)
        self.linear4 = nn.Linear(self.hidden, self.hidden)
        self.linear5 = nn.Linear(self.hidden, self.hidden)
        
        self.noisy_value1     = []
        self.noisy_value2     = []
        self.noisy_advantage1 = []
        self.noisy_advantage2 = []
        
        for i in range(num_head):
            self.noisy_value1.append(nn.Linear(self.hidden, self.hidden,device="cuda"))
            self.noisy_value2.append(nn.Linear(self.hidden, 1,device="cuda"))
            self.noisy_advantage1.append(nn.Linear(self.hidden, self.hidden,device="cuda"))
            self.noisy_advantage2.append(nn.Linear(self.hidden, self.num_actions,device="cuda"))
        
    def forward(self, x, head_index):        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        
        value = F.relu(self.noisy_value1[head_index](x))
        value = self.noisy_value2[head_index](value)
        
        advantage = F.relu(self.noisy_advantage1[head_index](x))
        advantage = self.noisy_advantage2[head_index](advantage)
        
        x = value + advantage - advantage.mean(1).unsqueeze(dim=1)
        
        return x
    
    def all_result(self, x, a):        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        Q_value = []
        for head_index in range(self.num_head):
            value = F.relu(self.noisy_value1[head_index](x))
            value = self.noisy_value2[head_index](value)
            advantage = F.relu(self.noisy_advantage1[head_index](x))
            advantage = self.noisy_advantage2[head_index](advantage)
            
            y = value + advantage - advantage.mean(1).unsqueeze(dim=1)
            Q_value.append(y.gather(1,a.long()).view(-1, 1))
        Q_value = torch.hstack(Q_value)
        return Q_value



def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def projection_distribution(next_state, rewards, dones):
    batch_size  = next_state.size(0)
    
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)

    support = torch.linspace(Vmin, Vmax, num_atoms)
    
    next_dist   = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    
    next_dist   = target_model(next_state).data.cpu()
    next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)
    
    Tz = rewards + (1 - dones) * current_model.gamma * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()
        
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())    
    temp_a = (l + offset).view(-1)
    temp_b = (next_dist * (u.float() - b)).view(-1)
    temp_c = (u + offset).view(-1)
    temp_d = (next_dist * (b - l.float())).view(-1)
    try:
        proj_dist.view(-1).index_add_(0, temp_a, temp_b)
        proj_dist.view(-1).index_add_(0, temp_c, temp_d)
    except:
        error_index_max = torch.argmax(temp_a)
        error_index_min = torch.argmin(temp_a)
        print(proj_dist.view(-1).shape,temp_b.shape,error_index_max,torch.max(temp_a),error_index_min,torch.min(temp_a))
        print(l.view(-1)[error_index_max],l.view(-1)[error_index_min])
        print(offset.view(-1)[error_index_max],offset.view(-1)[error_index_min])
        print(Tz.view(-1)[error_index_max],Tz.view(-1)[error_index_min])
        print(rewards.view(-1)[error_index_max],rewards.view(-1)[error_index_min])

    return proj_dist


def compute_td_loss(batch_size,is_side_info):
    if is_side_info:
        state, action, reward, next_state, done, demand_real = replay_buffer_side_info.sample(batch_size) 
    else:
        state, action, reward, next_state, done, demand_real = replay_buffer.sample(batch_size) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))
    
    proj_dist = projection_distribution(next_state, reward, done)
    
    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(current_model.parameters(),0.8)
    optimizer.step()
    lr_schedule.step()
    return loss

def compute_mhdqn_loss(batch_size,is_side_info,head_index):
    if is_side_info:
        state, action, reward, next_state, done, demand_real = replay_buffer_side_info.sample(batch_size) 
    else:
        state, action, reward, next_state, done, demand_real = replay_buffer.sample(batch_size) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))

    q_eval = mh_dqn(state,head_index)
    q_eval = q_eval.gather(1,action.long().unsqueeze(1)).view(-1, 1)
    q_next = mh_dqn_target(state,head_index)
    q_next = q_next.max(1).values.view(-1,1)
    
    q_target = reward.cuda().unsqueeze(1) + mh_dqn.gamma*q_next*(1-done.cuda().unsqueeze(1))
    
    loss = torch.mean((q_eval-q_target.detach())**2)
    
    mh_dqn_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mh_dqn.parameters(),0.8)
    mh_dqn_optimizer.step()
        
    return loss

def save(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def test_env(all_test_rewards):
    state = init_state
    q_arrivals = np.ones(len(init_state))
    episode_reward = 0
    episode_in_reward = 0
    current_model.epsilon = 0
    for _ in range(1, test_num_frames + 1):
        next_state, reward,in_reward, q_arrivals = env.test_step(state,current_model,q_arrivals,mh_dqn)
        state = next_state
        episode_reward += reward
        episode_in_reward += in_reward
        all_test_rewards.append(reward)
    print("Test: ",episode,episode_reward/test_num_frames,episode_in_reward/test_num_frames)
    current_model.epsilon = current_model.init_epsilon
    return episode_reward/test_num_frames,episode_in_reward/test_num_frames

for times in range(0,50):
    setup_seed(times)
    current_model = RainbowDQN(state_dim, act_dim, num_atoms, Vmin, Vmax,gamma)
    target_model  = RainbowDQN(state_dim, act_dim, num_atoms, Vmin, Vmax,gamma)
    
    mh_dqn = MHDQN(state_dim,act_dim,gamma,8)
    mh_dqn_target = MHDQN(state_dim,act_dim,gamma,8)
    
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
        mh_dqn  = mh_dqn.cuda()
        mh_dqn_target  = mh_dqn_target.cuda()

    optimizer = optim.Adam(current_model.parameters(), lr)
    mh_dqn_optimizer = optim.Adam(mh_dqn.parameters(), lr/10)
    lr_schedule = CosineAnnealingLR(optimizer, episodes*num_frames*update_times//update_per_step)
    
    replay_buffer = ReplayBuffer(rb_size)
    replay_buffer_side_info = ReplayBuffer(rb_size*side_info_scale**2)


    update_target(current_model, target_model)



    losses = [0]
    dqn_losses = [0]
    all_rewards = []
    all_test_rewards = []
    hot_g = Hot_Graph()

    for episode in range(1,episodes+1):
        state = init_state
        q_arrivals = np.ones(len(init_state))
        episode_reward = 0
        episode_in_reward = 0

        for frame_idx in range(1, num_frames + 1):
            next_state, reward,in_reward, done, q_arrivals = env.step(state,current_model,replay_buffer,replay_buffer_side_info,q_arrivals,side_info_scale,mh_dqn,hot_g)

            state = next_state
            episode_reward += reward
            episode_in_reward += in_reward
            all_rewards.append(reward)
            if len(replay_buffer) > batch_size*10:
                if current_model.epsilon == 1:
                    current_model.epsilon = current_model.init_epsilon
                    
                if frame_idx % update_per_step == 0:
                    for _ in range(update_times):
                        loss = compute_td_loss(batch_size,0)
                        if np.random.random() < 0.05:
                            losses.append(float(loss))

            if frame_idx % target_update_freq == 0:
                update_target(current_model, target_model)

        try:
            print("Train:",episode,episode_reward/num_frames,episode_in_reward/num_frames,losses[-1],dqn_losses[-1])
        except:
            print("Train:",episode,episode_reward/num_frames,episode_in_reward/num_frames,None)
        
        if episode>=1 and episode % test_per_episodes == 0:
            test_epi_avg_rewards, test_epi_avg_intr_rewards  = test_env(all_test_rewards)
