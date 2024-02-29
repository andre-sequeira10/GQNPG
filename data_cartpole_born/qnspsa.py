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

#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as npp

import autograd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0) #cuda device
parser.add_argument('--policy', type=str, default="Q") #policy
parser.add_argument('--ng',type=int,default=0)
parser.add_argument('--linear', type=str, default=None) #neurons for linear layer
parser.add_argument('--hidden', type=str, default=None) #neurons for single hidden layer
parser.add_argument('--lr', type=float, default=0.1)  #learning rate
parser.add_argument('--episodes', type=int, default=1000) #number of episodes    
parser.add_argument('--gamma', type=float, default=0.99) #discount factor                                  
parser.add_argument('--init', type=str, default="random_0_2pi") #discount factor                                  
parser.add_argument('--entanglement', type=str, default="mod") #discount factor                                  
parser.add_argument('--n_layers', type=int, default=3) #discount factor                                  
parser.add_argument('--batch_size', type=int, default=10) #discount factor                                  
parser.add_argument('--eigenvalue_filename', type=str, default="eigenvalue_cartpole") #discount factor                                  
parser.add_argument('--filename_save', type=str, default="default") #discount factor                                  
parser.add_argument('--eigenvalue', type=int, default=0) #discount 
parser.add_argument('--save', type=int, default=0) #saver         
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
filename_save = args.filename_save

print("Initializing ... QFIM - {}".format(ng))
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
init=args.init
#print("init ---> ",init)

if policy == "Q":   
    nm = "nn{}-RX-layers-{}||lr-{}||entanglement-{}||basis_change-{}||batch-{}||episodes-{}".format(init,n_layers,lr_q,ent,basis_change,batch_size,episodes)
else:
    nm = "C||4-32-64-4||linear-{}||hidden-{}".format(nn_linear,nn_hidden)


device = qml.device("default.qubit", wires = n_qubits+1)
device2 = qml.device("default.qubit", wires = n_qubits+1,shots=1)

def normalize(vector):
    norm = np.max(np.abs(np.asarray(vector)))
    return vector/norm
    
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

#@qml.batch_input(argnum=0)
@qml.qnode(device2)
def qcircuit_shot(inputs, weights0):
    
    for q in range(n_qubits):
        qml.Hadamard(wires=q)

    ansatz(inputs, weights0,n_layers=n_layers, entanglement=ent)

    ### SINGLE QUBIT MEASUREMENT EQUIVALENT TO TENSOR PRODUCT MEASUREMENT 
    for q in range(n_qubits-1):
        #qml.CNOT(wires=[q,n_qubits])
        qml.CNOT(wires=[q,q+1])

    return qml.sample(wires=n_qubits-1)
    ### OPTIMAL PARTITIONING 
    #return qml.probs(wires=range(n_qubits))

def discount_rewards(rewards, gamma=0.999):
    
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards

    return discounted_rewards

def step_wrapper(action):
    return env.step(action)

def step_vjp(ans, action):
    # The backward pass, returning None for gradients
    def gradient(g):
        return (None,)
    return gradient

# Tell autograd to use the custom vjp for step_wrapper
autograd.extend.defvjp(step_wrapper, step_vjp)
               
def reinforce(env, num_episodes=600,
              batch_size=10, gamma=0.99, lr=0.01 ,ng=0, label=None, parameters=None):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    avg_rewards = []
    batch_actions = []
    batch_actions_tensor=[]
    batch_states = []

    LEARNING_RATE = lr

    parameters = parameters

    if policy == "Q":
        if ng:
            optimizer = qml.QNGOptimizer(stepsize=0.1)
            g=1
        else:
            optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)
   
    grads = []
    vars=[]

    import time 

                
    for ep in range(1,num_episodes,batch_size):
        def loss_func(ep,parameters):
            p_nograd = np.array(parameters,requires_grad=False)
            for e in range(batch_size):
                s_0 = env.reset()   
                states = []
                max_reward=0
                rewards = []
                actions = []
                log_actions = []
                complete = False
                while complete == False:

                    s_0 = normalize(s_0)
                    #out = qcircuit(s_0, parameters)
                    
                    action = qcircuit_shot(s_0, p_nograd)

                    s_1, r, complete, _ = env.step(action)

                    
                    states.append(s_0)
                    
                    rewards.append(r)
                    actions.append(action)
                    tmp = s_0
                    s_0 = s_1

                    if complete:
                        
                        discounted_r = discount_rewards(rewards, gamma)
                        batch_rewards.extend(discounted_r)
                        avg_rewards.append(discounted_r)
                        avg_rewards_2 = [sum(x) for x in itertools.zip_longest(*avg_rewards, fillvalue=0)]
                        batch_states.extend(states)
                        batch_actions.extend(actions)
                        #batch_actions_tensor.extend(log_actions) 
                        total_rewards.append(sum(rewards))
                        sum_rewards = sum(rewards)
                        if sum_rewards >= max_reward:
                            max_reward = sum_rewards
                            best_episode = states
                        #batch_avg_reward += sum(rewards)
                        mean_r = np.mean(total_rewards[-10:])
                        
                        print("Ep: {} Average of last 10: {:.2f}".format(
                            ep + e, mean_r))
                        # If batch is complete, update network

            #return loss_f
        
            lens = list(map(len, avg_rewards))
            avg_rewards_2 = [sum(x) for x in itertools.zip_longest(*avg_rewards, fillvalue=0)]
            baseline = np.array(avg_rewards_2)
            for ep in range(len(avg_rewards)):
                for i in range(len(avg_rewards[ep])):
                    tam = 0 
                    for p in lens:
                        if p >= i:
                            tam+=1
                    avg_rewards[ep][i] -= baseline[i]/tam

            reward_tensor = [] 
            for ep in avg_rewards:
                reward_tensor.extend(ep)

            return reward_tensor

        def l_func(batch_states, batch_actions, reward_tensor, parameters):
            log_actions_tensor=0
            for (s,a,r) in zip(batch_states, batch_actions,reward_tensor):
                log_actions_tensor -= np.log(qcircuit(s,parameters)[a])*r
            
            loss_f = log_actions_tensor/len(batch_states)

            return loss_f

        reward_tensor = loss_func(ep,parameters)

        t_init = time.time()
        #print("p before",parameters)
        parameters, loss = optimizer.step_and_cost(lambda w: l_func(batch_states,batch_actions,reward_tensor, w), parameters)
        #print("p after", parameters)

        t_end = time.time()

        print("TIME - ", t_end-t_init)
        

        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_actions_tensor=[]
        avg_rewards = []
        
        '''
        for name, param in policy_estimator.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                grad_var = torch.var(param.grad)
        #grads_step = torch.cat(grads_step).pow(2).numpy().mean()
        
                grads.append(grad_norm)
                vars.append(grad_var)
        #wandb.log({"grads": grads[-1]})
        '''

    return total_rewards

env = gym.make('CartPole-v0')
    
p = np.random.normal(0, 1, (n_layers, n_qubits, 2),requires_grad=True)
    
rewards_q = reinforce(env , num_episodes=episodes, batch_size=batch_size, lr=lr_q, ng=ng, gamma=0.99, parameters=p)

processid = os.getpid()

np.save("cartpole_{}_NG_ - {} || {}.npy".format(filename_save,ng,str(processid)), rewards_q)
#np.save("cartpole_{}_NG_grads_norm - {} || {}.npy".format(filename_save,ng,str(processid)), grads_q)
#np.save("cartpole_{}_NG_vars - {} || {}.npy".format(filename_save,ng,str(processid)), vars)
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