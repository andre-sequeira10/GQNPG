import matplotlib.pyplot as plt
import numpy as np
import glob

#product approximation
reward_prod = []
grad_prod = []
var_prod = []
entanglement_prod = []
kl_prod = []
tv_distance=[]

#global
reward_global = []
grad_global = []
var_global = []
entanglement_global = []
kl_global = []
tv_distance_global=[]

#parity
reward_parity = []
grad_parity = []
var_parity = []
entanglement_parity=[]

#mean-approx
reward_mean_approx = []
grad_mean_approx = []
var_mean_approx = []
entanglement_mean_approx = []
kl_mean_approx = []
tv_distance_mean_approx=[]

#distances to optimal 
kl_distance_global_optimal = []
tv_distance_global_optimal = []
kl_distance_prod_optimal = []
tv_distance_prod_optimal = []
kl_distance_parity_optimal = []
tv_distance_parity_optimal = []


# Load data
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_product-approx_is_True_n_0_1_grads_norm*'):
    grad_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_product-approx_is_True_n_0_1_vars*'):
    var_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_product-approx_is_True_n_0_1_rewards*'):
    reward_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_product-approx_is_True_n_0_1_meyer_wallach*'):
    entanglement_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_product-approx_is_True_n_0_1_kl_divergence*'):
    kl_prod.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_product-approx_is_True_n_0_1_TV_distance*'):
    tv_distance.append(np.load(np_name))

#for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_grads_norm*'):
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_global_is_True_n_0_1_grads_norm*'):
    grad_global.append(np.load(np_name))
#for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_vars*'):
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_global_is_True_n_0_1_vars*'):
    var_global.append(np.load(np_name))
#for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_rewards*'):
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_global_is_True_n_0_1_rewards*'):
    reward_global.append(np.load(np_name))
#for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_meyer_wallach*'):
for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_meyer_wallach*'):
    entanglement_global.append(np.load(np_name))
#for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_kl_divergence*'):
for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_kl_divergence*'):
    kl_global.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_global_is_True_n_0_1_TV_distance*'):
    tv_distance_global.append(np.load(np_name))

#parity
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_parity_is_True_n_0_1_grads_norm*'):
    grad_parity.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_parity_is_True_n_0_1_vars*'):
    var_parity.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_parity_is_True_n_0_1_rewards*'):
    reward_parity.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_parity_is_True_n_0_1_meyer_wallach*'):
    entanglement_parity.append(np.load(np_name))

#mean-approx
for np_name in glob.glob('cartpole_zeros_ones_mean-approx_is_True_n_0_1_grads_norm*'):
    grad_mean_approx.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_mean-approx_is_True_n_0_1_vars*'):
    var_mean_approx.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_mean-approx_is_True_n_0_1_rewards*'):
    reward_mean_approx.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_mean-approx_is_True_n_0_1_meyer_wallach*'):
    entanglement_mean_approx.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_mean-approx_is_True_n_0_1_kl_divergence*'):
    kl_mean_approx.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_mean-approx_is_True_n_0_1_TV_distance*'):
    tv_distance_mean_approx.append(np.load(np_name))

#load distances to optimal policy 
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_global_is_True_n_0_1_kl_divergence*'):
    kl_distance_global_optimal.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_global_is_True_n_0_1_TV_distance*'):
    tv_distance_global_optimal.append(np.load(np_name))

for np_name in glob.glob('cartpole_zeros_ones_near_optimal_product-approx_is_True_n_0_1_kl_divergence*'):
    kl_distance_prod_optimal.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_product-approx_is_True_n_0_1_TV_distance*'):
    tv_distance_prod_optimal.append(np.load(np_name))

for np_name in glob.glob('cartpole_zeros_ones_near_optimal_parity_is_True_n_0_1_kl_divergence*'):
    kl_distance_parity_optimal.append(np.load(np_name))
for np_name in glob.glob('cartpole_zeros_ones_near_optimal_parity_is_True_n_0_1_TV_distance*'):
    tv_distance_parity_optimal.append(np.load(np_name))




# Calculate means and standard deviations
reward_prod_mean = np.array(reward_prod).mean(axis=0)
reward_prod_std = np.array(reward_prod).std(axis=0)


grad_prod_mean = np.array(grad_prod).mean(axis=0)
grad_prod_std = np.array(grad_prod).std(axis=0)


var_prod_mean = np.array(var_prod).mean(axis=0)
var_prod_std = np.array(var_prod).std(axis=0)


entanglement_prod_mean = np.array(entanglement_prod).mean(axis=0)
entanglement_prod_std = np.array(entanglement_prod).std(axis=0)


kl_prod_mean = np.array(kl_prod).mean(axis=0)
kl_prod_std = np.array(kl_prod).std(axis=0)

tv_distance_mean = np.array(tv_distance).mean(axis=0)
tv_distance_std = np.array(tv_distance).std(axis=0)

#global
reward_global_mean = np.array(reward_global).mean(axis=0)
reward_global_std = np.array(reward_global).std(axis=0)

grad_global_mean = np.array(grad_global).mean(axis=0)
grad_global_std = np.array(grad_global).std(axis=0)

var_global_mean = np.array(var_global).mean(axis=0)
var_global_std = np.array(var_global).std(axis=0)

entanglement_global_mean = np.array(entanglement_global).mean(axis=0)
entanglement_global_std = np.array(entanglement_global).std(axis=0)

kl_global_mean = np.array(kl_global).mean(axis=0)
kl_global_std = np.array(kl_global).std(axis=0)

tv_distance_global_mean = np.array(tv_distance_global).mean(axis=0)
tv_distance_global_std = np.array(tv_distance_global).std(axis=0)

#parity
reward_parity_mean = np.array(reward_parity).mean(axis=0)
reward_parity_std = np.array(reward_parity).std(axis=0)

grad_parity_mean = np.array(grad_parity).mean(axis=0)
grad_parity_std = np.array(grad_parity).std(axis=0)

var_parity_mean = np.array(var_parity).mean(axis=0)
var_parity_std = np.array(var_parity).std(axis=0)

entanglement_parity_mean = np.array(entanglement_parity).mean(axis=0)
entanglement_parity_std = np.array(entanglement_parity).std(axis=0)

#mean-approx
reward_mean_approx_mean = np.array(reward_mean_approx).mean(axis=0)
reward_mean_approx_std = np.array(reward_mean_approx).std(axis=0)

grad_mean_approx_mean = np.array(grad_mean_approx).mean(axis=0)
grad_mean_approx_std = np.array(grad_mean_approx).std(axis=0)

var_mean_approx_mean = np.array(var_mean_approx).mean(axis=0)
var_mean_approx_std = np.array(var_mean_approx).std(axis=0)

entanglement_mean_approx_mean = np.array(entanglement_mean_approx).mean(axis=0)
entanglement_mean_approx_std = np.array(entanglement_mean_approx).std(axis=0)

kl_mean_approx_mean = np.array(kl_mean_approx).mean(axis=0)
kl_mean_approx_std = np.array(kl_mean_approx).std(axis=0)

tv_distance_mean_approx_mean = np.array(tv_distance_mean_approx).mean(axis=0)
tv_distance_mean_approx_std = np.array(tv_distance_mean_approx).std(axis=0)

#distances to optimal mean
kl_distance_global_optimal_mean = np.array(kl_distance_global_optimal).mean(axis=0)
kl_distance_global_optimal_std = np.array(kl_distance_global_optimal).std(axis=0)

tv_distance_global_optimal_mean = np.array(tv_distance_global_optimal).mean(axis=0) 
tv_distance_global_optimal_std = np.array(tv_distance_global_optimal).std(axis=0)

kl_distance_prod_optimal_mean = np.array(kl_distance_prod_optimal).mean(axis=0)
kl_distance_prod_optimal_std = np.array(kl_distance_prod_optimal).std(axis=0)

tv_distance_prod_optimal_mean = np.array(tv_distance_prod_optimal).mean(axis=0)
tv_distance_prod_optimal_std = np.array(tv_distance_prod_optimal).std(axis=0)

kl_distance_parity_optimal_mean = np.array(kl_distance_parity_optimal).mean(axis=0)
kl_distance_parity_optimal_std = np.array(kl_distance_parity_optimal).std(axis=0)

tv_distance_parity_optimal_mean = np.array(tv_distance_parity_optimal).mean(axis=0)
tv_distance_parity_optimal_std = np.array(tv_distance_parity_optimal).std(axis=0)


# Smoothing
window = 10
smoothed_reward_prod = [np.mean(reward_prod_mean[i-window:i+1]) if i > window 
                        else np.mean(reward_prod_mean[:i+1]) for i in range(len(reward_prod_mean))]
smoothed_grad_prod = [np.mean(grad_prod_mean[i-window:i+1]) if i > window
                        else np.mean(grad_prod_mean[:i+1]) for i in range(len(grad_prod_mean))]
smoothed_var_prod = [np.mean(var_prod_mean[i-window:i+1]) if i > window
                        else np.mean(var_prod_mean[:i+1]) for i in range(len(var_prod_mean))]
smoothed_entanglement_prod = [np.mean(entanglement_prod_mean[i-window:i+1]) if i > window
                        else np.mean(entanglement_prod_mean[:i+1]) for i in range(len(entanglement_prod_mean))]
smoothed_kl_prod = [np.mean(kl_prod_mean[i-window:i+1]) if i > window
                        else np.mean(kl_prod_mean[:i+1]) for i in range(len(kl_prod_mean))]
smoothed_tv_distance = [np.mean(tv_distance_mean[i-window:i+1]) if i > window
                        else np.mean(tv_distance_mean[:i+1]) for i in range(len(tv_distance_mean))]

#global smoothing 

smoothed_reward_global = [np.mean(reward_global_mean[i-window:i+1]) if i > window
                        else np.mean(reward_global_mean[:i+1]) for i in range(len(reward_global_mean))]
smoothed_grad_global = [np.mean(grad_global_mean[i-window:i+1]) if i > window
                        else np.mean(grad_global_mean[:i+1]) for i in range(len(grad_global_mean))]
smoothed_var_global = [np.mean(var_global_mean[i-window:i+1]) if i > window
                           else np.mean(var_global_mean[:i+1]) for i in range(len(var_global_mean))]
smoothed_entanglement_global = [np.mean(entanglement_global_mean[i-window:i+1]) if i > window
                        else np.mean(entanglement_global_mean[:i+1]) for i in range(len(entanglement_global_mean))]
smoothed_kl_global = [np.mean(kl_global_mean[i-window:i+1]) if i > window
                          else np.mean(kl_global_mean[:i+1]) for i in range(len(kl_global_mean))]
smoothed_tv_distance_global = [np.mean(tv_distance_global_mean[i-window:i+1]) if i > window
                        else np.mean(tv_distance_global_mean[:i+1]) for i in range(len(tv_distance_global_mean))]

#parity smoothing
smoothed_reward_parity = [np.mean(reward_parity_mean[i-window:i+1]) if i > window
                        else np.mean(reward_parity_mean[:i+1]) for i in range(len(reward_parity_mean))]
smoothed_grad_parity = [np.mean(grad_parity_mean[i-window:i+1]) if i > window
                        else np.mean(grad_parity_mean[:i+1]) for i in range(len(grad_parity_mean))]
smoothed_var_parity = [np.mean(var_parity_mean[i-window:i+1]) if i > window
                           else np.mean(var_parity_mean[:i+1]) for i in range(len(var_parity_mean))]
smoothed_entanglement_parity = [np.mean(entanglement_parity_mean[i-window:i+1]) if i > window
                        else np.mean(entanglement_parity_mean[:i+1]) for i in range(len(entanglement_parity_mean))]

#mean-approx smoothing
smoothed_reward_mean_approx = [np.mean(reward_mean_approx_mean[i-window:i+1]) if i > window
                        else np.mean(reward_mean_approx_mean[:i+1]) for i in range(len(reward_mean_approx_mean))]
smoothed_grad_mean_approx = [np.mean(grad_mean_approx_mean[i-window:i+1]) if i > window
                        else np.mean(grad_mean_approx_mean[:i+1]) for i in range(len(grad_mean_approx_mean))]
smoothed_var_mean_approx = [np.mean(var_mean_approx_mean[i-window:i+1]) if i > window
                            else np.mean(var_mean_approx_mean[:i+1]) for i in range(len(var_mean_approx_mean))]
smoothed_entanglement_mean_approx = [np.mean(entanglement_mean_approx_mean[i-window:i+1]) if i > window
                        else np.mean(entanglement_mean_approx_mean[:i+1]) for i in range(len(entanglement_mean_approx_mean))]
smoothed_kl_mean_approx = [np.mean(kl_mean_approx_mean[i-window:i+1]) if i > window
                        else np.mean(kl_mean_approx_mean[:i+1]) for i in range(len(kl_mean_approx_mean))]
smoothed_tv_distance_mean_approx = [np.mean(tv_distance_mean_approx_mean[i-window:i+1]) if i > window
                        else np.mean(tv_distance_mean_approx_mean[:i+1]) for i in range(len(tv_distance_mean_approx_mean))]

#distance to optimal smoothing 
smoothed_kl_distance_global_optimal = [np.mean(kl_distance_global_optimal_mean[i-window:i+1]) if i > window
                        else np.mean(kl_distance_global_optimal_mean[:i+1]) for i in range(len(kl_distance_global_optimal_mean))]
smoothed_tv_distance_global_optimal = [np.mean(tv_distance_global_optimal_mean[i-window:i+1]) if i > window
                        else np.mean(tv_distance_global_optimal_mean[:i+1]) for i in range(len(tv_distance_global_optimal_mean))]
smoothed_kl_distance_prod_optimal = [np.mean(kl_distance_prod_optimal_mean[i-window:i+1]) if i > window
                        else np.mean(kl_distance_prod_optimal_mean[:i+1]) for i in range(len(kl_distance_prod_optimal_mean))]
smoothed_tv_distance_prod_optimal = [np.mean(tv_distance_prod_optimal_mean[i-window:i+1]) if i > window
                        else np.mean(tv_distance_prod_optimal_mean[:i+1]) for i in range(len(tv_distance_prod_optimal_mean))]
smoothed_kl_distance_parity_optimal = [np.mean(kl_distance_parity_optimal_mean[i-window:i+1]) if i > window
                        else np.mean(kl_distance_parity_optimal_mean[:i+1]) for i in range(len(kl_distance_parity_optimal_mean))]
smoothed_tv_distance_parity_optimal = [np.mean(tv_distance_parity_optimal_mean[i-window:i+1]) if i > window
                        else np.mean(tv_distance_parity_optimal_mean[:i+1]) for i in range(len(tv_distance_parity_optimal_mean))]
 
'''
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
listt = list(range(len(reward_prod_mean)))

# First subplot: Policy gradient norm
axs[0].set_xlim([0, len(reward_prod_mean)])
#axs[0].fill_between(listt, reward_prod_mean - reward_prod_std, reward_prod_mean + reward_prod_std, color="darkblue", alpha=0.1)

axs[0].plot(listt, reward_prod_mean, color="darkblue", alpha=0.3)
axs[0].plot(listt, smoothed_reward_prod, label="prod approx", color="darkblue", alpha=0.7)

axs[0].plot(listt, reward_global_mean, color="darkred", alpha=0.3)
axs[0].plot(listt, smoothed_reward_global, label="global", color="darkred", alpha=0.7)

axs[0].plot(listt, reward_parity_mean, color="purple", alpha=0.3)
axs[0].plot(listt, smoothed_reward_parity, label="parity", color="purple", alpha=0.7)

axs[0].plot(listt, reward_mean_approx_mean, color="darkorange", alpha=0.3)
axs[0].plot(listt, smoothed_reward_mean_approx, label="mean-approx", color="darkorange", alpha=0.7)

#axs[0].legend()
axs[0].set_ylabel('Rewards')
axs[0].set_xlabel('Episodes')

# Second subplot: Policy gradient variance
axs[1].set_xlim([0, len(var_prod_mean)])
#axs[1].fill_between(listt, var_prod_mean - var_prod_std, var_prod_mean + var_prod_std, color="darkblue", alpha=0.1)

axs[1].plot(list(range(len(smoothed_var_prod))), smoothed_var_prod, label="prod approx", color="darkblue", alpha=0.7)
axs[1].plot(list(range(len(smoothed_var_prod))), var_prod_mean, color="darkblue", alpha=0.3)

axs[1].plot(list(range(len(smoothed_var_global))), smoothed_var_global, label="global", color="darkred", alpha=0.7)
axs[1].plot(list(range(len(smoothed_var_global))), var_global_mean, color="darkred", alpha=0.3)

axs[1].plot(list(range(len(smoothed_var_parity))), smoothed_var_parity, label="parity", color="purple", alpha=0.7)
axs[1].plot(list(range(len(smoothed_var_parity))), var_parity_mean, color="purple", alpha=0.3)

axs[1].plot(list(range(len(smoothed_var_mean_approx))), smoothed_var_mean_approx, label="mean-approx", color="darkorange", alpha=0.7)
axs[1].plot(list(range(len(smoothed_var_mean_approx))), var_mean_approx_mean, color="darkorange", alpha=0.3)

#axs[1].legend()
axs[1].set_ylabel('Policy gradient variance')
axs[1].set_xlabel('Episodes/Batch size')

# Third subplot: Policy gradient norm
axs[2].set_xlim([0, len(grad_prod_mean)])
#axs[2].fill_between(listt, grad_prod_mean - grad_prod_std, grad_prod_mean + grad_prod_std, color="darkblue", alpha=0.1)

axs[2].plot(list(range(len(grad_prod_mean))), grad_prod_mean, color="darkblue", alpha=0.3)
axs[2].plot(list(range(len(grad_prod_mean))), smoothed_grad_prod, label="product-approx", color="darkblue", alpha=0.7)
axs[2].plot(list(range(len(grad_global_mean))), grad_global_mean, color="darkred", alpha=0.3)
axs[2].plot(list(range(len(grad_global_mean))), smoothed_grad_global, label="global", color="darkred", alpha=0.7)

axs[2].plot(list(range(len(grad_parity_mean))), grad_parity_mean, color="purple", alpha=0.3)
axs[2].plot(list(range(len(grad_parity_mean))), smoothed_grad_parity, label="parity", color="purple", alpha=0.7)

axs[2].plot(list(range(len(grad_mean_approx_mean))), grad_mean_approx_mean, color="darkorange", alpha=0.3)
axs[2].plot(list(range(len(grad_mean_approx_mean))), smoothed_grad_mean_approx, label="mean-approx", color="darkorange", alpha=0.7)



#axs[2].legend()
axs[2].set_ylabel('Policy gradient norm')
axs[2].set_xlabel('Episodes/Batch size')

fig.subplots_adjust(right=0.7)

# Now, create the legend using the fourth subplot's lines and labels
# and place it to the right of the fifth subplot using bbox_to_anchor
lines, labels = axs[2].get_legend_handles_labels()
axs[2].legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()  # Adjust the spacing between subplots
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

#add text label at top left corner of each subplot:
axs[0].text(0.05, 0.90, '(a)', transform=axs[0].transAxes, size=12, weight='bold')  
axs[1].text(0.05, 0.90, '(b)', transform=axs[1].transAxes, size=12, weight='bold')
axs[2].text(0.05, 0.90, '(c)', transform=axs[2].transAxes, size=12, weight='bold')


plt.show()
plt.close()
'''

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
listt = list(range(len(reward_prod_mean)))

# Fourth subplot: Entanglement
axs[0].set_xlim([0, len(entanglement_prod_mean)])
#axs[0].fill_between(listt, entanglement_prod_mean - entanglement_prod_std, entanglement_prod_mean + entanglement_prod_std, color="darkblue", alpha=0.1)

axs[0].plot(listt, entanglement_prod_mean, color="darkblue", alpha=0.3)
axs[0].plot(listt, smoothed_entanglement_prod, label="product-approx", color="darkblue", alpha=0.7)
axs[0].plot(listt, entanglement_global_mean, color="darkred", alpha=0.3)
axs[0].plot(listt, smoothed_entanglement_global, label="global", color="darkred", alpha=0.7)

axs[0].plot(listt, entanglement_parity_mean, color="purple", alpha=0.3)
axs[0].plot(listt, smoothed_entanglement_parity, label="parity", color="purple", alpha=0.7)

axs[0].plot(listt, entanglement_mean_approx_mean, color="darkorange", alpha=0.3)
axs[0].plot(listt, smoothed_entanglement_mean_approx, label="mean-approx", color="darkorange", alpha=0.7)


#axs[0].legend()
axs[0].set_ylabel('Entanglement')
axs[0].set_xlabel('Episodes')

# Fifth subplot: KL divergence
axs[1].set_xlim([0, len(kl_prod_mean)])
#axs[1].fill_between(listt, kl_prod_mean - kl_prod_std, kl_prod_mean + kl_prod_std, color="darkblue", alpha=0.1)

#
#axs[1].plot(listt, kl_prod_mean, color="darkblue", alpha=0.3)
#axs[1].plot(listt, smoothed_kl_prod, label="product-approx - KL divergence", color="darkblue", alpha=0.7)
axs[1].plot(listt, tv_distance_mean, color="darkblue", alpha=0.3)
axs[1].plot(listt, smoothed_tv_distance, label="product-approx", color="darkblue", alpha=0.7)

#global
#axs[1].plot(listt, kl_global_mean, color="darkred", alpha=0.3)
#axs[1].plot(listt, smoothed_kl_global, label="global - KL divergence", color="darkred", alpha=0.7)
axs[1].plot(listt, tv_distance_global_mean, color="darkred", alpha=0.3)
axs[1].plot(listt, smoothed_tv_distance_global, label="global", color="darkred", alpha=0.7)

#mean-approx
#axs[1].plot(listt, kl_mean_approx_mean, color="darkorange", alpha=0.3)
#axs[1].plot(listt, smoothed_kl_mean_approx, label="mean-approx - KL divergence", color="darkorange", alpha=0.7)
#axs[1].plot(listt, tv_distance_mean_approx_mean, color="darkorange", alpha=0.3)
#axs[1].plot(listt, smoothed_tv_distance_mean_approx, label="mean-approx", color="darkorange", alpha=0.7)

#axs[1].legend(loc="upper right")
axs[1].set_ylabel('Total variation distance')
axs[1].set_xlabel('Episodes')

#sixth subplot: distance to optimal policy
axs[2].set_xlim([0, len(kl_distance_prod_optimal_mean)])

#axs[2].plot(listt, kl_distance_prod_optimal_mean, color="darkblue", alpha=0.3)
#axs[2].plot(listt, smoothed_kl_distance_prod_optimal, label="product-approx", color="darkblue", alpha=0.7)
axs[2].plot(listt, tv_distance_prod_optimal_mean, color="darkblue", alpha=0.3)
axs[2].plot(listt, smoothed_tv_distance_prod_optimal, label="product-approx", color="darkblue", alpha=0.7)

#axs[2].plot(listt, kl_distance_global_optimal_mean, color="darkred", alpha=0.3)
#axs[2].plot(listt, smoothed_kl_distance_global_optimal, label="global", color="darkred", alpha=0.7)
axs[2].plot(listt, tv_distance_global_optimal_mean, color="darkred", alpha=0.3)
axs[2].plot(listt, smoothed_tv_distance_global_optimal, label="global", color="darkred", alpha=0.7)

#parity
# axs[2].plot(listt, kl_distance_parity_optimal_mean, color="purple", alpha=0.3)
# axs[2].plot(listt, smoothed_kl_distance_parity_optimal, label="parity", color="purple", alpha=0.7)
axs[2].plot(listt, tv_distance_parity_optimal_mean, color="purple", alpha=0.3)
axs[2].plot(listt, smoothed_tv_distance_parity_optimal, label="parity", color="purple", alpha=0.7)
    
axs[2].set_ylabel('Distance to near-optimal policy')

fig.subplots_adjust(right=0.7)

# Now, create the legend using the fourth subplot's lines and labels
# and place it to the right of the fifth subplot using bbox_to_anchor
lines, labels = axs[0].get_legend_handles_labels()
axs[2].legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()  # Adjust the spacing between subplots
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


#add text label at bottom left corner of each subplot:
axs[0].text(0.05, 0.05, '(a)', transform=axs[0].transAxes, size=12, weight='bold')
axs[1].text(0.05, 0.05, '(b)', transform=axs[1].transAxes, size=12, weight='bold')
axs[2].text(0.05, 0.05, '(c)', transform=axs[2].transAxes, size=12, weight='bold')



plt.show()

