import numpy as np
import itertools
from copy import deepcopy
from scipy.stats import poisson
import torch


Demand_Max = 20
Demand_Mean = 5

demand_realizations = np.arange(Demand_Max + 1)
#mu is the expectation of poisson distribution, which is lambda in paper
demand_probabilities = poisson.pmf(np.arange(Demand_Max + 1), mu=Demand_Mean)
demand_probabilities[-1] += 1 - np.sum(demand_probabilities)

act_dim=20
LT_s = 4
multiplier = 4
initial_state = np.ones(LT_s)*multiplier
LT_min = LT_s
LT_max = LT_s
InvMax = 100


c=0
h=1
p=4


def transition_stochLT(s, a, d, q_arrivals,LT):
    # q_arrivals 计算每个order quantity剩余的ld，ld=0的就是已经到达的
    # 对于到达的订单，因为s包含每个时刻的i和q，所以把q_arrivals对应的s里的q加起来就是到达的货物
    arrived = np.sum(s[1:][q_arrivals[1:]==0])
    # 更新s，到达了的订单清零，同时因为q到了，所以i增加
    # s = [i_t,q_t], q_arrivals = [LT_t-l+1,...,LT_t], if l=2, then it is [LT_t-1, LT_t], 但是如果这么理解，那为什么是arrived = np.sum(s[1:][q_arrivals[1:]==0])啊？
    s[1:][q_arrivals[1:]==0] = 0
    s[0] += arrived

    # order lead time at next time
    q_arrivals -= 1
    q_arrivals = np.roll(q_arrivals,-1)
    q_arrivals[-1] = LT-1
    q_arrivals = np.clip(q_arrivals,0,np.inf)

    s1 = np.roll(s, -1)

    s1[-1] = a
    # 这个不是那个订单量大于库存顾客就不买了，而是商店恰巧就卖库存的量
    s1[0] = np.clip(max(s[0] - d,0) + s[1], 0,InvMax - 1)
    reward = c*a + max(s[0] - d, 0) * h + min(s[0] - d, 0) * -p
    return reward, s1, q_arrivals

def step(state,agent,replay_buffer,replay_buffer_side_info,q_arrivals,side_info_scale,mh_dqn,hot_g):
    s = deepcopy(state)
    s2 = s/(InvMax)

    demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
    # demand = 5
    LT = np.random.randint(LT_min, LT_max+1)
    a = agent.act(s2,mh_dqn)
    # print("action:",a)
    # a_tensor = torch.Tensor([a])
    in_r = 0
    if mh_dqn is not None:
        in_r = mh_dqn.all_result(torch.FloatTensor(s2).cuda().unsqueeze(0),torch.Tensor([a]).cuda().unsqueeze(0)).std().detach().cpu()
    if demand >= s[0]:
        demand_real = s[0]
    else:
        demand_real = demand
    # print("before:","init state:",s,"init s2:",s2,"initial q_arrivals:",q_arrivals)
    if side_info_scale != 0:
        for t in range(0,side_info_scale):
            s_side = deepcopy(s)
            try:
                # loss sale, 如果demand大于库存，我们只能观测到库存以下的side info
                if t != side_info_scale -1:
                    if demand >= s[0]:
                        # s_side[0] = np.random.randint(max(s[0]-8,0),s[0])
                        s_side[0] = np.random.randint(0,s[0])
                    else:
                        # s_side[0] = np.random.randint(max(s[0]-4,0),min(s[0]+4,InvMax))
                        s_side[0] = np.random.randint(0,InvMax)
            except:
                pass
            s2_side = s_side/(InvMax)
            if mh_dqn is not None:
                gf_actions = mh_dqn.get_action(torch.FloatTensor(s2_side).cuda().unsqueeze(0),side_info_scale-1).cpu().numpy()
            for tt in range(0,side_info_scale):
                graph_feed_a = a
                if tt != side_info_scale-1:
                    if mh_dqn is not None:
                        # graph_feed_a = gf_actions[0,tt]
                        graph_feed_a = np.random.randint(max(a-2,0),min(a+2,act_dim))
                    else:
                        graph_feed_a = np.random.randint(max(a-2,0),min(a+2,act_dim))


                r, s1, _ = transition_stochLT(deepcopy(s_side), graph_feed_a, demand, deepcopy(q_arrivals), LT)
                s_side_ = deepcopy(s1)
                replay_buffer_side_info.push(s2_side, graph_feed_a, -r/10, s_side_/(InvMax), 0, demand_real)
    # print(s,a,demand,q_arrivals,LT)
    # print("after :","init state:",s,"init s2:",s2,"initial q_arrivals:",q_arrivals)
    r, s1, q_arrivals = transition_stochLT(deepcopy(s), a, demand, deepcopy(q_arrivals),LT)
    hot_g.add(s,a)
    s = deepcopy(s1)
    replay_buffer.push(s2, a, -r/10, s/(InvMax), 0, demand_real)
    # print("s:",s2*(InvMax),"a:",a,"dmd:",demand,"cost:",-r,"s':",s1,"q_arrivals:",q_arrivals)
    return s1,-r,float(in_r),0,q_arrivals

def test_step(state,agent,q_arrivals,mh_dqn):
    s = deepcopy(state)
    s2 = s/(InvMax)

    demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
    LT = np.random.randint(LT_min, LT_max+1)
    a = agent.act(s2,mh_dqn)

    in_r = mh_dqn.all_result(torch.FloatTensor(s2).cuda().unsqueeze(0),torch.Tensor([a]).cuda().unsqueeze(0)).std().detach().cpu()

    r, s1, q_arrivals = transition_stochLT(deepcopy(s), a, demand, deepcopy(q_arrivals),LT)

    s = deepcopy(s1)

    # print("s:",s2*(InvMax),"a:",a,"dmd:",demand,"cost:",-r,"s':",s1,"q_arrivals:",q_arrivals)
    return s1,-r,float(in_r),q_arrivals
