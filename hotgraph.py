import matplotlib.pyplot as plt
import numpy as np

class Hot_Graph():
    def __init__(self) -> None:
        self.state_list = []
        self.action_list = []
    
    def cal_state(self,s):
        return s
    
    def add(self,s,a):
        s_ = self.cal_state(s)
        self.state_list.append(s_)
        self.action_list.append(a)
    
    def store(self,name):
        np.savetxt(name+"_state.txt",np.array(self.state_list))
        np.savetxt(name+"_action.txt",np.array(self.action_list))
    
    # def plot(self):
    #     # 散点图
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     # s：marker标记的大小
    #     # c: 颜色  可为单个，可为序列
    #     # depthshade: 是否为散点标记着色以呈现深度外观。对 scatter() 的每次调用都将独立执行其深度着色。
    #     # marker：样式
        
    #     ax.scatter(xs=self.state_list, ys=y, zs=0, zdir='z', s=30, c="g", depthshade=True, cmap="jet", marker="^")
    #     plt.show()