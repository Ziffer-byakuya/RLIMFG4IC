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