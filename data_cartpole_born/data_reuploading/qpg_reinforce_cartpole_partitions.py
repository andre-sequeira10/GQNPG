from statistics import mean
import matplotlib.pyplot as plt
import gym

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os 

import pennylane as qml 
from pennylane.templates import StronglyEntanglingLayers as SEL 
from pennylane.templates import BasicEntanglerLayers as BEL 
from pennylane.templates import IQPEmbedding
from pennylane.templates import AngleEmbedding
from pennylane import expval as expectation
from pennylane import PauliZ as Z 
from pennylane import PauliX as X 
from numpy import linalg 

from pennylane import numpy as np 
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian
import itertools
#import wandb
import argparse
from operator import itemgetter 
import copy

from PQC_based_policies import BornPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0) #cuda device
parser.add_argument('--policy', type=str, default="Q") #policy
parser.add_argument('--ng',type=int,default=0)
parser.add_argument('--linear', type=str, default=None) #neurons for linear layer
parser.add_argument('--hidden', type=str, default=None) #neurons for single hidden layer
parser.add_argument('--lr', type=float, default=0.1)  #learning rate
parser.add_argument('--episodes', type=int, default=1000) #number of episodes    
parser.add_argument('--gamma', type=float, default=0.99) #discount factor                                  
parser.add_argument('--init', type=str, default="normal_0_1") #discount factor                                  
parser.add_argument('--entanglement', type=str, default="all2all") #discount factor                                  
parser.add_argument('--n_layers', type=int, default=4) #discount factor                                  
parser.add_argument('--batch_size', type=int, default=10) #discount factor                                  
parser.add_argument('--eigenvalue_filename', type=str, default="eigenvalue_cartpole") #discount factor                                  
parser.add_argument('--eigenvalue', type=int, default=0) #discount 
parser.add_argument('--save', type=int, default=0) #saver   
parser.add_argument('--kl_divergence', type=int, default=0)
parser.add_argument('--meyer_wallach_entanglement', type=int, default=0)
parser.add_argument('--data_reuploading', type=bool, default=False)
parser.add_argument('--environment', type=str, default='CartPole-v0')
parser.add_argument('--measurement', type=str, default='n-local')
parser.add_argument('--born_policy', type=str, default='global')
parser.add_argument('--torch_diff_method', type=str, default='backprop')
parser.add_argument('--comparator_policy', type=str, default='global')

args = parser.parse_args()

episodes=args.episodes
n_layers = args.n_layers
n_qubits = 4
lr_q = args.lr
batch_size = args.batch_size
policy = args.policy
ng=args.ng
eigenvalue_filename = args.eigenvalue_filename
eigenvalue = args.eigenvalue
save = args.save

print("Initializing ... QFIM - {}".format(ng))
print("initializing ... ",args.init)
if args.linear == None:
    nn_linear=None
else:
    nn_linear=int(args.linear)

if args.hidden == None:
    nn_hidden=None
else:
    nn_hidden=int(args.hidden)

basis_change=False 
ent=args.entanglement
initialization=args.init
kl_divergence_ = args.kl_divergence
meyer_wallach_entanglement = args.meyer_wallach_entanglement
environment = args.environment
data_reuploading = args.data_reuploading
measurement = args.measurement
born_policy = args.born_policy
torch_diff_method = args.torch_diff_method
compp_policy = args.comparator_policy

#wandb.init(name=nm,project="qPG")#, entity="quantumai")

'''
wandb.config = {
  "learning_rate": lr_q,
  "epochs": 1000,
  "batch_size": batch_size,
  "layers": n_layers
}
'''
device = qml.device("default.qubit", wires = n_qubits)
device_fisher = qml.device("default.qubit", wires = n_qubits)#,shots=10000)#)
device_meyer_wallach = qml.device("default.qubit", wires = n_qubits)

def normalize(vector):
    norm = np.max(np.abs(np.asarray(vector)))
    return vector/norm
    
def ansatz_jerbi(state, weights,input_scaling=None, n_qubits=2, n_layers=1, change_of_basis=False, entanglement="all2all"):
    if change_of_basis==True:
        for l in range(len(weights)):
            for i in range(n_qubits):
                qml.Rot(*weights[l][i],wires=i)
                #qml.RY(weights[l][i][0],wires=i)
                #qml.RZ(weights[l][i][1],wires=i)
    else:          
        for l in range(len(weights)):
            for i in range(n_qubits):
                qml.RZ(weights[l][i][0],wires=i)
                qml.RY(weights[l][i][1],wires=i)
                #qml.RZ(weights[l][i][2],wires=i)

            #if l < n_layers:
            if entanglement == "all2all":
                for q1 in range(n_qubits-1):    
                    for q2 in range(q1+1, n_qubits):
                        qml.CNOT(wires=[q1,q2])
                        #qml.CZ(wires=[q1,q2])
            else:
                for q in range (1,n_qubits):
                    qml.CNOT(wires=[q,0])
                for q in range (2,n_qubits):
                    qml.CNOT(wires=[q,1])
            
            if l < n_layers:
                if l == n_layers-1:
                    if input_scaling is not None:
                        qml.AngleEmbedding(state*input_scaling[l][0], wires=range(n_qubits),rotation="Z")
                        qml.AngleEmbedding(state*input_scaling[l][1], wires=range(n_qubits),rotation="Y")
                    else:
                        qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")
                        qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
                else:
                    if input_scaling is not None:
                        qml.AngleEmbedding(state*input_scaling[l][0], wires=range(n_qubits),rotation="Y")
                        qml.AngleEmbedding(state*input_scaling[l][1], wires=range(n_qubits),rotation="Z")
                    else:
                        qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
                        qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")

def ansatz(state, weights, n_layers=1, change_of_basis=False, entanglement="all2all"):
        if change_of_basis==True:
            for l in range(len(weights)):
                for i in range(n_qubits):
                    qml.Rot(*weights[l][i],wires=i)
                    #qml.RY(weights[l][i][0],wires=i)
                    #qml.RZ(weights[l][i][1],wires=i)
        else:          
            for l in range(len(weights)):
                for i in range(n_qubits):
                    qml.RZ(weights[l][i][0],wires=i)
                    qml.RY(weights[l][i][1],wires=i)
                    #qml.RZ(weights[l][i][2],wires=i)

                #if l < n_layers:
                if entanglement == "all2all":
                    for q1 in range(n_qubits-1):    
                        for q2 in range(q1+1, n_qubits):
                            qml.CNOT(wires=[q1,q2])
                            #qml.CZ(wires=[q1,q2])

                
                elif entanglement == "mod":
                    if not (l+1)%n_qubits:
                        l=0
                    for q1 in range(n_qubits):
                        #qml.CNOT(wires=[q1,(q1+l+1)%n_qubits])
                        qml.CNOT(wires=[q1,(q1+l+1)%n_qubits])

                elif entanglement == "linear":
                    for q1 in range(n_qubits-1):    
                        qml.CNOT(wires=[q1,q1+1])

                elif entanglement == "circular":
                    #if l+1 < n_layers:
                    for q1 in range(n_qubits):
                        qml.CNOT(wires=[q1,(q1+1)%n_qubits])
                        #qml.CZ(wires=[q1,(q1+1)%n_qubits])

                
                elif entanglement == "nn":
                    qml.CNOT(wires=[0,1])
                    qml.CNOT(wires=[2,3])
                    qml.CNOT(wires=[1,2])
                else:
                    for q in range (1,n_qubits):
                        qml.CNOT(wires=[q,0])
                    for q in range (2,n_qubits):
                        qml.CNOT(wires=[q,1])
                
                if l < n_layers-1:
                    qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
                    qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")

#@qml.batch_input(argnum=0)
@qml.qnode(device, diff_method="backprop")
def qcircuit(inputs, weights0):
    
    for q in range(n_qubits):
        qml.Hadamard(wires=q)

    ansatz(inputs, weights0,n_layers=n_layers, entanglement=ent)

    ### SINGLE QUBIT MEASUREMENT EQUIVALENT TO TENSOR PRODUCT MEASUREMENT 
    for q in range(n_qubits-1):
        #qml.CNOT(wires=[q,n_qubits])
        qml.CNOT(wires=[q,q+1])
    return qml.probs(wires=n_qubits-1)
    ### OPTIMAL PARTITIONING 
    #return qml.probs(wires=range(n_qubits))

@qml.qnode(device_fisher,diff_method="backprop")#, parallel=True)#, interface="autograd")
def qcircuit_fisher(inputs, weights0):
    
    for q in range(n_qubits):
        qml.Hadamard(wires=q)

    ansatz(inputs, weights0,n_layers=n_layers, entanglement=ent)
    
    for q in range(n_qubits-1):
        #qml.CNOT(wires=[q,n_qubits])
        qml.CNOT(wires=[q,q+1])
    return qml.probs(wires=n_qubits-1)

    #return qml.probs(wires=range(n_qubits))

@qml.qnode(device_meyer_wallach)
def meyer_wallach_circuit(inputs, weights0):
    
    AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
    ansatz(weights0,n_layers=n_layers, entanglement=ent)

    return qml.state()

class QNGOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, diag_approx=False, lam=0.01):
        defaults = dict(lr=lr, diag_approx=diag_approx, lam=lam)
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss, fisher_info_matrix = closure()

        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                #g = metric_tensor([p.data.numpy()], diag_approx=group["diag_approx"])

                fisher_info_matrix += group["lam"] * np.identity(fisher_info_matrix.shape[0])

                f_inv_grad = np.linalg.solve(fisher_info_matrix, grad.reshape(fisher_info_matrix.shape[0]))
                #lr = np.sqrt((2*group['lr'])/(np.dot(grad.reshape(fisher_info_matrix.shape[0]),f_inv_grad)))
                lr = np.sqrt((2*0.01)/(np.dot(grad.reshape(fisher_info_matrix.shape[0]),f_inv_grad)))
                state["step"] += 1
                d_p = torch.tensor(-lr * f_inv_grad)
                p.data.add_(d_p.reshape(grad.shape))

        return loss

class policy_estimator_q(nn.Module):        
    def __init__(self, env):
        super(policy_estimator_q, self).__init__()
        #weight_shapes = {"weights0":(n_layers, n_qubits, 2)}#,"coeffs":(3)}#,"weights2":(n_layers,n_qubits,3),"weights3":(n_layers,n_qubits,3),"weights4":(n_layers,n_qubits,3)}#, "weights5":(1,n_qubits,3)}
        if policy == "Q":
            weight_shapes = {"weights0":(n_layers, n_qubits,2)}#,"weights1":(1,n_qubits,3)}#,"weights2":(n_layers,n_qubits,3),"weights3":(n_layers,n_qubits,3),"weights4":(n_layers,n_qubits,3)}#, "weights5":(1,n_qubits,3)}
            import functools

            #self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

            if args.init == "random_0_2pi":
                #self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes)
                self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=2*np.pi)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
            elif args.init == "glorot":
                self.init_method_normal = functools.partial(torch.nn.init.normal_, mean=0.0, std=np.sqrt(3/4))
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method_normal)
            elif args.init == "random_-1_1":
                self.init_method = functools.partial(torch.nn.init.uniform_, a=-1, b=1)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
            elif args.init == "random_0_1":
                self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=1)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
            elif args.init == "random_-pi_pi":
                self.init_method = functools.partial(torch.nn.init.uniform_, a=-np.pi, b=np.pi)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
            elif args.init == "zeros":
                self.init_method = functools.partial(torch.nn.init.zeros_)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
            elif args.init == "normal_0_1":
                self.init_method_normal = functools.partial(torch.nn.init.normal_, mean=0.0, std=1)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method_normal)
            elif args.init == "normal_0_01":
                self.init_method_normal = functools.partial(torch.nn.init.normal_, mean=0.0, std=0.1)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method_normal)
            elif args.init == "normal_0_1_3":
                self.init_method_normal = functools.partial(torch.nn.init.normal_, mean=0.0, std=np.sqrt(1/3))
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method_normal)
            elif args.init == "xavier":
                self.init_method = functools.partial(torch.nn.init.xavier_normal_)
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
            elif args.init == "xavier_uniform":
                n_i = 4
                n_j = 4
                self.init_method = functools.partial(torch.nn.init.uniform_, a=-np.sqrt(6 /(n_i + n_j)), b=np.sqrt(6 /(n_i + n_j)))
                self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.init_method)
        else:
            self.fc1 = nn.Linear(4, nn_linear)
            if nn_hidden is not None:
                self.fc2 = nn.Linear(nn_linear, 64)
                self.fc3 = nn.Linear(64,2)
                #self.dropout = nn.Dropout(p=0.2)

                #self.fc4 = nn.Linear(16,2)
            else:
                self.fc2 = nn.Linear(nn_linear,2)
            
        #self.uniform = functools.partial(torch.nn.init.uniform_, a=-np.pi, b=np.pi)
        #self.glorot = functools.partial(torch.nn.init.normal_, mean=0.0, std=np.sqrt(1/3))
        #self.normal = functools.partial(torch.nn.init.normal_, mean=0.0, std=1)
        #self.uniform_values = torch.nn.init.uniform_(weight_shapes["weights0"],a=min_value,b=max_value)
        #self.normal = torch.nn.init.normal_
        #self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, self.normal)
        #self.fc1 = nn.Linear(4, 16)
        #self.ws = nn.Parameter(torch.ones(3), requires_grad=True)
    


    def forward(self, state):
        #QUANTUM ACTION SELECTION
        if policy == "Q":
            out = self.qlayer(torch.FloatTensor(state))
            #optimal partitioning
            '''
            p0=[]
            p1=[]
            for l in range(2**n_qubits):
                if np.binary_repr(l, width=n_qubits).count("1") % 2 == 0:
                    #i0.append(l)
                    p0.append(out[l])
                else:
                    #i1.append(l)
                    p1.append(out[l])

            #p0 = torch.sum(torch.stack(itemgetter(*i0)(out)))
            #p1 = torch.sum(torch.stack(itemgetter(*i1)(out)))
            p0 = torch.sum(torch.stack(p0))
            p1 = torch.sum(torch.stack(p1))
            #p1 = torch.sum(torch.cat(p1))
            action_probs = torch.stack((p0,p1))
            '''
            #single-qubit
            action_probs = out

            #out = self.qlayer(torch.FloatTensor(state))
            #action_probs.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
            #p0 = torch.sum(out[:int((2**n_qubits)/2)])
            #p1 = torch.sum(out[int((2**n_qubits)/2):])
            #action_probs = torch.stack((p1,p0))
            #out=self.fc1(out)
            #out = torch.multiply(self.ws,out)
            #action_probs = F.softmax(self.beta*out, dim=-1)
            #action_probs = F.softmax(2*out, dim=-1)

        else:
            #self.dropout(torch.FloatTensor(state))
            out=F.relu(self.fc1(torch.FloatTensor(state)))
            if nn_hidden is not None:
                #self.dropout(out)
                out=F.relu(self.fc2(out))
                #self.dropout(out)  
                out = self.fc3(out)
                #out = self.fc4(out)
            else:
                out = self.fc2(out) 

            action_probs = F.softmax(out, dim=-1)

        m = Categorical(action_probs)
        action = m.sample()
        log_probb = m.log_prob(action)
        #log_probb.register_hook(lambda x: x.clamp(min=0, max=2.5))
        #return action.item(), m.log_prob(action)
        return action.item(), log_probb
        #action_probs = F.softmax(3*out, dim=-1)

    def hessian_log_likelihood(self,state):
        if policy == "Q":
            out = self.qlayer(torch.FloatTensor(state))
            #out=self.fc1(out)
            #out = torch.multiply(self.ws,out)
            #action_probs = F.softmax(self.beta*out, dim=-1)
            p0=[]
            p1=[]
            for l in range(2**n_qubits):
                if np.binary_repr(l, width=n_qubits).count("1") % 2 == 0:
                    #i0.append(l)
                    p0.append(out[l])
                else:
                    #i1.append(l)
                    p1.append(out[l])

            #p0 = torch.sum(torch.stack(itemgetter(*i0)(out)))
            #p1 = torch.sum(torch.stack(itemgetter(*i1)(out)))
            p0 = torch.sum(torch.stack(p0))
            p1 = torch.sum(torch.stack(p1))
            #p1 = torch.sum(torch.cat(p1))
            action_probs = torch.stack((p0,p1))

        else:
            #self.dropout(torch.FloatTensor(state))
            out=F.relu(self.fc1(torch.FloatTensor(state)))
            if nn_hidden is not None:
                #self.dropout(out)
                out=F.relu(self.fc2(out))
                #self.dropout(out)
                out = self.fc3(out)
                #out = self.fc4(out)
            else:
                out = self.fc2(out) 

            action_probs = F.softmax(out, dim=-1)
        #m = Categorical(action_probs)
        #action = m.sample()
        #return m.log_prob(action).detach()
        return action_probs

    def get_kl(self, x):
        action_prob1=[]
        for s in x:
            action_prob1.append(self.hessian_log_likelihood(s))
        # calling .data detaches action_prob0 from the graph, so it will not be part of the gradient computation.
        # Also, starting PyTorch 0.4, the Variable wrapper is no longer needed. 
        #action_prob0 = action_prob1.data

        kl = torch.stack([action_prob0 * (torch.log(action_prob0) - torch.log(action_prob0.data)) for action_prob0 in action_prob1])
        return kl.sum()#1, keepdim=True)
    
def discount_rewards(rewards, gamma=0.999):
    
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards

    return discounted_rewards
    
def reinforce(env, policy_estimator, num_episodes=600,
              batch_size=10, gamma=0.99, lr=0.01 ,ng=0, label=None, ent=False, kl_divergence=False, comparator_policy='global'):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    avg_rewards = []
    batch_actions = []
    batch_actions_tensor=[]
    batch_states = []
    batch_counter = 0
    eigen_total=[]
    meyer_wallach_avg = []
    kl_divergence_avg=[]
    tv_distance_avg=[]
    LEARNING_RATE = lr
    lr_qnpg = 0.02
    best_episode = []
    max_reward=0
   #for name,param in policy_estimator.named_parameters():
        #print(name,"\n")
        #print(param,"\n")

    if policy == "Q":
        #optimizer = optim.Adam(policy_estimator.parameters(),
        if ng:
            optimizer = QNGOptimizer([
                           {"params": policy_estimator.qlayer.weights}])#, "lr": LEARNING_RATE}])
        else:
           optimizer = optim.Adam(policy_estimator.parameters(),
                            lr=LEARNING_RATE,amsgrad=True)
                           #{"params": policy_estimator.beta, "lr": 0.3}],
                            #lr=LEARNING_RATE
    else:
        optimizer = optim.Adam(policy_estimator.parameters(),
                            lr=LEARNING_RATE)
   
    grads = []
    vars=[]

    import time 

    for ep in range(num_episodes):
        s_0 = env.reset()   
        states = []
        max_reward=0
        rewards = []
        actions = []
        eigenvalues_ep = []
        log_actions = []
        complete = False
        meyer_wallach_ep = []
        kl_divergence_ep = []
        tv_distance_ep = []
        while complete == False:
            #action_probs = policy_estimator.forward(s_0).detach().numpy()
            s_0 = torch.from_numpy(normalize(s_0)).float().unsqueeze(0)
            action, action_log_prob, pi = policy_estimator.forward(s_0)
            log_actions.append(action_log_prob)
            #action_probs_sampler = torch.clone(action_probs).detach().numpy()


            if ent:
                Q = policy_estimator.get_meyer_wallach(s_0)
                meyer_wallach_ep.append(Q)
            
            if kl_divergence:
                original_weights_dict = policy_estimator.get_weights()

                #global_policy = BornPolicy(circuit=None, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy=comparator_policy, device="default.qubit", shots=None, diff_method=torch_diff_method)
                
                if comparator_policy == 'global' or comparator_policy == 'product-approx':
                    global_policy = BornPolicy(circuit=None, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy=comparator_policy, device="default.qubit", shots=None, diff_method=torch_diff_method)

                    #global_policy.set_weights(policy_estimator.get_weights())
                    global_policy.set_weights(original_weights_dict)
                elif comparator_policy == 'optimal':
                    global_policy = BornPolicy(circuit=None, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement=measurement, measurement_qubits=None, policy="product-approx", device="default.qubit", shots=None, diff_method=torch_diff_method)

                    global_policy.load_state_dict(torch.load('optimal_policy_weights.pt'))

                action_global, action_log_prob_global, pi_global = global_policy.forward(s_0)

                pi_global = pi_global.detach().numpy()
                pi = pi.detach().numpy()

                kl_divergence_ep.append(np.sum(pi_global*np.log(pi_global/pi)))

                tv_distance_ep.append(0.5*np.sum(np.abs(pi_global-pi)))

            #action = np.random.choice([-1,0,1], p=action_probs_sampler)
            

            #Cartpole and Mountaincar
            s_1, r, complete, _ = env.step(action)
            #s_1, r, terminated, truncated, _ = env.step(action)
            #complete = terminated or truncated

            #rw = -(s_1[1] + np.sin(np.arcsin(s_1[1])+np.arcsin(s_1[3])))
            #Acrobot
            #s_1, r, complete, _ = env.step(action-1)
            
            states.append(s_0)
            
            rewards.append(r)
            actions.append(action)
            tmp = s_0
            s_0 = s_1

            if complete:
                meyer_wallach_avg.append(np.mean(meyer_wallach_ep))
                kl_divergence_avg.append(np.mean(kl_divergence_ep))
                tv_distance_avg.append(np.mean(tv_distance_ep))
                #### QUANTUM FISHER INFORMATION EIGENVALUE DISTRIBUTION ####
                if policy == "Q":
                    if eigenvalue:
                        w = torch.clone(policy_estimator.qlayer.weights0).detach().numpy()
                        qfim = np.array(qml.qinfo.quantum_fisher(qcircuit_fisher)(tmp, np.array(w, requires_grad=True))) 
                        qfim = qfim.reshape((n_qubits*2*n_layers, n_qubits*2*n_layers))
                        eigvalues, v = linalg.eig(qfim)  
                        eigenvalues_ep.extend(np.round(eigvalues,1))
                ####################################################
                elif policy=="C":
                    if eigenvalue:
                 
                        def hessian(network, states):
                            #pa = network.forward(states)
                            pa_sum = policy_estimator.get_kl(states)
                            # calculate the first derivative of the loss wrt network parameters
                            J = torch.autograd.grad(pa_sum, policy_estimator.parameters(), create_graph=True, retain_graph=True)
                            J_ = torch.Tensor()
                            # concatenate the various gradient tensors (for each layer) into one vector
                            for grad in J:
                                J_ = torch.cat((J_, grad.view(-1)), 0)
                        
                            H = torch.Tensor()
                            # calculate gradient wrt each element and concatenate into the Hessian matrix
                            for Ji in J_:
                                JJ = torch.autograd.grad(Ji, policy_estimator.parameters(), create_graph=False, retain_graph=True)
                                JJ_ = torch.cat([grad.contiguous().view(-1) for grad in JJ])
                                H = torch.cat((H, JJ_), 0)
                            # numParams is the number of parameters in the network
                            numParams = sum(p.numel() for p in policy_estimator.parameters() if p.requires_grad)
                            HH = H.view((numParams, numParams))
                            return HH
                        
                        hessian_m = hessian(policy_estimator, tmp)
                        eigvalues, v = linalg.eig(hessian_m)  
                        eigenvalues_ep.extend(np.round(eigvalues,1))
                 
                discounted_r = discount_rewards(rewards, gamma)
                batch_rewards.extend(discounted_r)
                avg_rewards.append(discounted_r)
                avg_rewards_2 = [sum(x) for x in itertools.zip_longest(*avg_rewards, fillvalue=0)]
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_actions_tensor.extend(log_actions) 
                batch_counter += 1
                total_rewards.append(sum(rewards))
                sum_rewards = sum(rewards)
                if sum_rewards >= max_reward:
                    max_reward = sum_rewards
                    best_episode = states
                #batch_avg_reward += sum(rewards)
                mean_r = np.mean(total_rewards[-10:])
                
                # If batch is complete, update network
                if batch_counter == batch_size-1:
                    t_init = time.time()
                    def closure():
                        optimizer.zero_grad(set_to_none=True)

                        #state_tensor = torch.FloatTensor(np.array(batch_states))
                        
                        lens = list(map(len, avg_rewards))
                        baseline = np.array(avg_rewards_2)
                        for ep in range(len(avg_rewards)):
                            for i in range(len(avg_rewards[ep])):
                                tam = 0 
                                for p in lens:
                                    if p >= i:
                                        tam+=1
                                avg_rewards[ep][i] -= baseline[i]/tam

                        batch_rewards = [] 
                        for ep in avg_rewards:
                            batch_rewards.extend(ep)

                        reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                        action_tensor = torch.LongTensor(np.array(batch_actions))

                        #outs = policy_estimator.forward(state_tensor)
                        #logprob = torch.log(outs)
                        logprob = torch.stack(batch_actions_tensor)
                        #print("logprob ",logprob)
                        #entropy2 = outs.entropy()
                        selected_logprobs = torch.multiply(reward_tensor,logprob)#[np.arange(len(action_tensor)), action_tensor]
                        #print("selected logprob", selected_logprobs)
                        loss = -torch.mean(selected_logprobs)
                        #loss = loss / batch_size 
                        #print("mean - " , loss)

                        loss.backward()

                        if ng:
                            w = copy.deepcopy(policy_estimator.qlayer.weights0.data)
                            #qfim = qml.qinfo.quantum_fisher(qcircuit_fisher)(np.mean(states,axis=0), np.array(w, requires_grad=True))[1]
                            qfim = np.zeros((n_qubits*2*n_layers, n_qubits*2*n_layers))
                            states_executed = []
                            qfim_dict = {}
                            for s in best_episode:
                                if str(s) not in states_executed:
                                    states_executed.append(str(s))
                                    #qfim_ += qml.qinfo.classical_fisher(qcircuit_fisher)(np.mean(states,axis=0), np.array(w, requires_grad=True))[1]
                                    #qfim_ = qml.qinfo.quantum_fisher(qcircuit_fisher)(s, np.array(w, requires_grad=True))[1]
                                    mt_fn = qml.metric_tensor(qcircuit_fisher, approx="block-diag",hybrid=True)
                                    qfim_ = mt_fn(s, np.array(w, requires_grad=True))[1]

                                    qfim_dict[str(s)] = qfim_
                                    #qfim_ = qml.qinfo.quantum_fisher(qcircuit_fisher)(s, np.array(w, requires_grad=True))[1]
                                    qfim_ = qfim_.reshape((n_qubits*2*n_layers, n_qubits*2*n_layers))
                                    #qfim_ = np.diag(qfim_)
                                else:
                                    qfim_ = qfim_dict[str(s)]
                                qfim += qfim_
                            qfim /= len(best_episode)
                            return loss, qfim
                        else:
                            return loss
                    
                    #if clip:
                        #torch.nn.utils.clip_grad_norm_(policy_estimator.parameters(), 2*np.pi)
                    optimizer.step(closure)

                    t_end = time.time()

                    print("TIME - ", t_end-t_init)
                    
                    #for name,param in policy_estimator.named_parameters():
                        #if name == "ws":
                            #print(name,"\n")
                            #print(param,"\n")
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_actions_tensor=[]
                    avg_rewards = []
                    batch_counter = 0
                    
                    for name, param in policy_estimator.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad)
                            grad_var = torch.var(param.grad)
                    #grads_step = torch.cat(grads_step).pow(2).numpy().mean()
                    
                            grads.append(grad_norm)
                            vars.append(grad_var)
                    #wandb.log({"grads": grads[-1]})

                mean_r = np.mean(total_rewards[-10:])
                
                if (mean_r >= 190) and ng:
                    ng=0
                    optimizer = optim.Adam([
                           {"params": policy_estimator.qlayer.weights0, "lr": 0.01}])#, "momentum":0.9}])#,
                           #{"params": policy_estimator.beta, "lr": 0.3}],
                            #lr=LEARNING_RATE, 
                        #amsgrad=True)
                
                #wandb.log({"total_rewards": total_rewards[-1]})
                #wandb.log({"mean_rewards_10": mean_r})

                if eigenvalue:
                    eigenvalues_ep = np.array(eigenvalues_ep)
                    if np.iscomplexobj(eigenvalues_ep):
                        eigenvalues_ep = abs(eigenvalues_ep)
                    eigen_total.extend(eigenvalues_ep)
                '''
                #create list of occurences from eigenvalue list.
                eigen_counter = len(eigen_total)
                eigenvalue_occurences = np.unique(eigen_total, return_counts=True)
                data = []
                column_data = []
                numParams = sum(p.numel() for p in policy_estimator.parameters() if p.requires_grad)
                for i in range(len(eigenvalue_occurences[0])):
                    #column_data.append(str(eigenvalue_occurences[0][i]))
                    data.append([str(eigenvalue_occurences[0][i]),eigenvalue_occurences[1][i]])#/eigen_counter])
                
                #table = wandb.Table(data=[[i] for i in eigenvalues_ep])#, columns=["scores"])
                #wandb.log({"eigenvalue_dist": eigenvalues_ep})

                #table = wandb.Table(data=data, columns=["eigenvalue","counts"])
                table = wandb.Table(data=data, columns=["counts"])
                wandb.log({'my_hist': wandb.plot.histogram(table, "counts")})
                #wandb.log({'my_barchart': wandb.plot.bar(table, "eigenvalue", "counts")})
                #wandb.log({'my_barchart': wandb.plot.histogram(table, "counts")})
                '''
                # Optional
                #wandb.watch(policy_estimator)
            
                mean_meyer_wallach = meyer_wallach_avg[-1]
                mean_kl = kl_divergence_avg[-1]
                mean_tv = tv_distance_avg[-1]
                print("Ep: {} || Reward - {} || Entanglement - {} || KL - {} || TV - {}".format(
                    ep + 1, mean_r, mean_meyer_wallach, mean_kl, mean_tv))
                
                #if np.mean(total_rewards[-100:]) >= 195:
                    #print("done!")
                    #np.save("optimal_policy_weights.npy", policy_estimator.get_weights().detach().numpy())
                    #torch.save(policy_estimator.state_dict(), "optimal_policy_weights.pt")
                    #break
                #print("Meyer-Wallach entanglement mean - {}".format(mean_meyer_wallach))
    return total_rewards, grads, vars, eigen_total, meyer_wallach_avg, kl_divergence_avg, tv_distance_avg

env = gym.make(environment)

pe_q = BornPolicy(circuit=None, n_actions=env.action_space.n, n_qubits=n_qubits, n_layers=n_layers, reuploading=data_reuploading, init=initialization, measurement="1-local", measurement_qubits=[n_qubits-1], policy=born_policy, device="default.qubit", shots=None, diff_method=torch_diff_method)


#model_q = torch.nn.DataParallel(pe_q)
print(kl_divergence_)

rewards_q , grads_q,vars,  eigenvalues, meyer_wallach_ent, kl_divergence, tv_distance = reinforce(env, pe_q , num_episodes=episodes, batch_size=batch_size, lr=lr_q, ng=ng, gamma=0.99,ent=meyer_wallach_entanglement, kl_divergence=kl_divergence_, comparator_policy=compp_policy)

if eigenvalue:  
    with open(eigenvalue_filename+'.npy', 'wb') as f:
        np.save(f, eigenvalues)

processid = os.getpid()

np.save("cartpole_zeros_ones_near_optimal_{}_is_{}_n_0_1_rewards || {}.npy".format(born_policy,data_reuploading,str(processid)), rewards_q)
np.save("cartpole_zeros_ones_near_optimal_{}_is_{}_n_0_1_grads_norm || {}.npy".format(born_policy,data_reuploading,str(processid)), grads_q)
np.save("cartpole_zeros_ones_near_optimal_{}_is_{}_n_0_1_vars || {}.npy".format(born_policy,data_reuploading,str(processid)), vars)
np.save("cartpole_zeros_ones_near_optimal_{}_is_{}_n_0_1_meyer_wallach || {}.npy".format(born_policy,data_reuploading,str(processid)), meyer_wallach_ent)
np.save("cartpole_zeros_ones_near_optimal_{}_is_{}_n_0_1_kl_divergence_global || {}.npy".format(born_policy,data_reuploading,str(processid)), kl_divergence)
np.save("cartpole_zeros_ones_near_optimal_{}_is_{}_n_0_1_TV_distance_global || {}.npy".format(born_policy,data_reuploading,str(processid)), tv_distance)

#np.save("cartpole_meyer_wallach"+policy+"_"+str(init)+"_"+str(processid)+".npy", meyer_wallach_ent)
'''
for i in range(10):
    s0 = env.reset()
    complete = False
    while not complete:
        #action_probs = pe_q.forward(s0).detach().numpy()
        action, action_log_prob = pe_q.forward(s0)

                #action = np.random.choice(action_space, p=action_probs)
        #action = np.random.choice([-1,0,1], p=action_probs)
        s_1, r, complete, _ = env.step(action)
        env.render()
        s0 = s_1
'''