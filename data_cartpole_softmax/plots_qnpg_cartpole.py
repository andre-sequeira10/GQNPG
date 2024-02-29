import matplotlib.pyplot as plt
import numpy as np
import glob

#qfim
reward_qfim = []
grad_qfim = []
var_qfim = []

#qfim_half
reward_qfim_half = []
grad_qfim_half = []
var_qfim_half = []

#fim
reward_fim = []
grad_fim = []
var_fim = []

#qfim block_diag
reward_qfim_block_diag = []
grad_qfim_block_diag = []
var_qfim_block_diag = []

#adam
reward_adam = []
grad_adam = []
var_adam = []

# Load data qfim
for np_name in glob.glob('cartpole_qfim_NG_ - 1*'):
    reward_qfim.append(np.load(np_name))
for np_name in glob.glob('cartpole_qfim_NG_grads_norm*'):
    grad_qfim.append(np.load(np_name))
for np_name in glob.glob('cartpole_qfim_NG_vars*'):
    var_qfim.append(np.load(np_name))

# Load data qfim_half
for np_name in glob.glob('cartpole_qfim_half_NG_ - 1*'):
    reward_qfim_half.append(np.load(np_name))
for np_name in glob.glob('cartpole_qfim_half_NG_grads_norm*'):
    grad_qfim_half.append(np.load(np_name))
for np_name in glob.glob('cartpole_qfim_half_NG_vars*'):
    var_qfim_half.append(np.load(np_name))

# Load data fim
for np_name in glob.glob('cartpole_fim_classical_NG_ - 1*'):
    reward_fim.append(np.load(np_name))
for np_name in glob.glob('cartpole_fim_classical_NG_grads_norm*'):
    grad_fim.append(np.load(np_name))
for np_name in glob.glob('cartpole_fim_classical_NG_vars*'):
    var_fim.append(np.load(np_name))

# Load data qfim block_diag
for np_name in glob.glob('../data_cartpole_born/cartpole_qfim_block_diag_nNG_ - 1*'):
    reward_qfim_block_diag.append(np.load(np_name))
for np_name in glob.glob('../data_cartpole_born/cartpole_qfim_block_diag_nNG_grads_norm*'):
    grad_qfim_block_diag.append(np.load(np_name))
for np_name in glob.glob('../data_cartpole_born/cartpole_qfim_block_diag_nNG_vars*'):
    var_qfim_block_diag.append(np.load(np_name))

#load data adam
for np_name in glob.glob('cartpole_adao_0.02_NG_ - 0*'):
    reward_adam.append(np.load(np_name))
for np_name in glob.glob('cartpole_adao_0.02_NG_grads_norm*'):
    grad_adam.append(np.load(np_name))
for np_name in glob.glob('cartpole_adao_0.02_NG_vars*'):
    var_adam.append(np.load(np_name))

# Calculate means and standard deviations qfim
reward_qfim_mean = np.array(reward_qfim).mean(axis=0)
reward_qfim_std = np.array(reward_qfim).std(axis=0)

grad_qfim_mean = np.array(grad_qfim).mean(axis=0)
grad_qfim_std = np.array(grad_qfim).std(axis=0)

var_qfim_mean = np.array(var_qfim).mean(axis=0)
var_qfim_std = np.array(var_qfim).std(axis=0)

# Calculate means and standard deviations qfim_half
reward_qfim_half_mean = np.array(reward_qfim_half).mean(axis=0)
reward_qfim_half_std = np.array(reward_qfim_half).std(axis=0)

grad_qfim_half_mean = np.array(grad_qfim_half).mean(axis=0)
grad_qfim_half_std = np.array(grad_qfim_half).std(axis=0)

var_qfim_half_mean = np.array(var_qfim_half).mean(axis=0)
var_qfim_half_std = np.array(var_qfim_half).std(axis=0)

# Calculate means and standard deviations fim
reward_fim_mean = np.array(reward_fim).mean(axis=0)
reward_fim_std = np.array(reward_fim).std(axis=0)

grad_fim_mean = np.array(grad_fim).mean(axis=0)
grad_fim_std = np.array(grad_fim).std(axis=0)

var_fim_mean = np.array(var_fim).mean(axis=0)
var_fim_std = np.array(var_fim).std(axis=0)

# Calculate means and standard deviations qfim block_diag
reward_qfim_block_diag_mean = np.array(reward_qfim_block_diag).mean(axis=0)
reward_qfim_block_diag_std = np.array(reward_qfim_block_diag).std(axis=0)

#calculate means and standard deviations adam
reward_adam_mean = np.array(reward_adam).mean(axis=0)
reward_adam_std = np.array(reward_adam).std(axis=0)

grad_adam_mean = np.array(grad_adam).mean(axis=0)
grad_adam_std = np.array(grad_adam).std(axis=0)

var_adam_mean = np.array(var_adam).mean(axis=0)
var_adam_std = np.array(var_adam).std(axis=0)


'''
grad_qfim_block_diag_mean = np.array(grad_qfim_block_diag).mean(axis=0)
grad_qfim_block_diag_std = np.array(grad_qfim_block_diag).std(axis=0)

var_qfim_block_diag_mean = np.array(var_qfim_block_diag).mean(axis=0)
var_qfim_block_diag_std = np.array(var_qfim_block_diag).std(axis=0)
'''

window = 15

# Smoothing qfim
smoothed_reward_qfim = [np.mean(reward_qfim_mean[i-window:i+1]) if i > window 
                        else np.mean(reward_qfim_mean[:i+1]) for i in range(len(reward_qfim_mean))]
smoothed_grad_qfim = [np.mean(grad_qfim_mean[i-window:i+1]) if i > window
                        else np.mean(grad_qfim_mean[:i+1]) for i in range(len(grad_qfim_mean))]
smoothed_var_qfim = [np.mean(var_qfim_mean[i-window:i+1]) if i > window
                        else np.mean(var_qfim_mean[:i+1]) for i in range(len(var_qfim_mean))]

# Smoothing qfim_half
smoothed_reward_qfim_half = [np.mean(reward_qfim_half_mean[i-window:i+1]) if i > window
                        else np.mean(reward_qfim_half_mean[:i+1]) for i in range(len(reward_qfim_half_mean))]
smoothed_grad_qfim_half = [np.mean(grad_qfim_half_mean[i-window:i+1]) if i > window
                        else np.mean(grad_qfim_half_mean[:i+1]) for i in range(len(grad_qfim_half_mean))]
smoothed_var_qfim_half = [np.mean(var_qfim_half_mean[i-window:i+1]) if i > window
                        else np.mean(var_qfim_half_mean[:i+1]) for i in range(len(var_qfim_half_mean))]

# Smoothing fim
smoothed_reward_fim = [np.mean(reward_fim_mean[i-window:i+1]) if i > window
                        else np.mean(reward_fim_mean[:i+1]) for i in range(len(reward_fim_mean))]
smoothed_grad_fim = [np.mean(grad_fim_mean[i-window:i+1]) if i > window
                        else np.mean(grad_fim_mean[:i+1]) for i in range(len(grad_fim_mean))]
smoothed_var_fim = [np.mean(var_fim_mean[i-window:i+1]) if i > window
                        else np.mean(var_fim_mean[:i+1]) for i in range(len(var_fim_mean))]

# Smoothing qfim block_diag
smoothed_reward_qfim_block_diag = [np.mean(reward_qfim_block_diag_mean[i-window:i+1]) if i > window
                        else np.mean(reward_qfim_block_diag_mean[:i+1]) for i in range(len(reward_qfim_block_diag_mean))]

# Smoothing adam
smoothed_reward_adam = [np.mean(reward_adam_mean[i-window:i+1]) if i > window
                        else np.mean(reward_adam_mean[:i+1]) for i in range(len(reward_adam_mean))]
smoothed_grad_adam = [np.mean(grad_adam_mean[i-window:i+1]) if i > window
                        else np.mean(grad_adam_mean[:i+1]) for i in range(len(grad_adam_mean))]
smoothed_var_adam = [np.mean(var_adam_mean[i-window:i+1]) if i > window
                        else np.mean(var_adam_mean[:i+1]) for i in range(len(var_adam_mean))]


'''
smoothed_grad_qfim_block_diag = [np.mean(grad_qfim_block_diag_mean[i-window:i+1]) if i > window
                        else np.mean(grad_qfim_block_diag_mean[:i+1]) for i in range(len(grad_qfim_block_diag_mean))]
smoothed_var_qfim_block_diag = [np.mean(var_qfim_block_diag_mean[i-window:i+1]) if i > window
                        else np.mean(var_qfim_block_diag_mean[:i+1]) for i in range(len(var_qfim_block_diag_mean))]
'''

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
listt = list(range(len(reward_qfim_mean)))

# First subplot: Policy gradient norm
axs[0].set_xlim([0, len(reward_qfim_mean)])
#axs[0].fill_between(listt, reward_prod_mean - reward_prod_std, reward_prod_mean + reward_prod_std, color="darkblue", alpha=0.1)

axs[0].plot(listt, reward_qfim_mean, color="darkblue", alpha=0.3)
axs[0].plot(listt, smoothed_reward_qfim, label="GQNGP $\gamma=1$", color="darkblue", alpha=0.9)

axs[0].plot(listt, reward_qfim_half_mean, color="darkorange", alpha=0.3)
axs[0].plot(listt, smoothed_reward_qfim_half, label="GQNGP $\gamma=0.5$", color="darkorange", alpha=0.9)

axs[0].plot(listt, reward_fim_mean, color="purple", alpha=0.3)
axs[0].plot(listt, smoothed_reward_fim, label="NPG", color="purple", alpha=0.7)

#axs[0].plot(listt, reward_qfim_block_diag_mean, color="darkgreen", alpha=0.3)
#axs[0].plot(listt, smoothed_reward_qfim_block_diag, label="QNPG block diag", color="darkgreen", alpha=0.9)

axs[0].plot(listt, reward_adam_mean, color="darkred", alpha=0.3)
axs[0].plot(listt, smoothed_reward_adam, label="Adam", color="darkred", alpha=0.9)
#axs[0].axhline(y=195, color="black", linestyle="--", alpha=0.5)
#axs[0].legend()
axs[0].set_ylabel('Rewards')
axs[0].set_xlabel('Episodes')

# Second subplot: Policy gradient variance
listt = list(range(len(var_qfim_mean)))

axs[1].set_xlim([0, len(var_qfim_mean)])
#axs[1].fill_between(listt, var_prod_mean - var_prod_std, var_prod_mean + var_prod_std, color="darkblue", alpha=0.1)

axs[1].plot(listt, var_qfim_mean, color="darkblue", alpha=0.3)
axs[1].plot(listt, smoothed_var_qfim, label="GQNGP $\gamma=1$", color="darkblue", alpha=0.7)

axs[1].plot(listt, var_qfim_half_mean, color="darkorange", alpha=0.3)
axs[1].plot(listt, smoothed_var_qfim_half, label="GQNGP $\gamma=0.5$", color="darkorange", alpha=0.7)

axs[1].plot(listt, var_fim_mean, color="purple", alpha=0.3)
axs[1].plot(listt, smoothed_var_fim, label="NPG", color="purple", alpha=0.7)

#adam
axs[1].plot(list(range(len(var_adam_mean))), var_adam_mean, color="darkred", alpha=0.3)
axs[1].plot(list(range(len(var_adam_mean))), smoothed_var_adam, label="Adam", color="darkred", alpha=0.9)


#axs[1].plot(listt, var_qfim_block_diag_mean, color="cyan", alpha=0.3)
#axs[1].plot(listt, smoothed_var_qfim_block_diag, label="NPG", color="purple", alpha=0.7)

#axs[1].legend()
axs[1].set_ylabel('(Q)NPG variance')

# Third subplot: Policy gradient norm
axs[2].set_xlim([0, len(grad_qfim_mean)])
#axs[2].fill_between(listt, grad_prod_mean - grad_prod_std, grad_prod_mean + grad_prod_std, color="darkblue", alpha=0.1)

axs[2].plot(listt, grad_qfim_mean, color="darkblue", alpha=0.3)
axs[2].plot(listt, smoothed_grad_qfim, label="GQNGP $\gamma=1$", color="darkblue", alpha=0.7)

axs[2].plot(listt, grad_qfim_half_mean, color="darkorange", alpha=0.3)
axs[2].plot(listt, smoothed_grad_qfim_half, label="GQNGP $\gamma=0.5$", color="darkorange", alpha=0.7)

axs[2].plot(listt, grad_fim_mean, color="purple", alpha=0.3)
axs[2].plot(listt, smoothed_grad_fim, label="NPG", color="purple", alpha=0.7)

#adam
axs[2].plot(list(range(len(grad_adam_mean))), grad_adam_mean, color="darkred", alpha=0.3)
axs[2].plot(list(range(len(grad_adam_mean))), smoothed_grad_adam, label="Adam", color="darkred", alpha=0.9)

#axs[2].plot(listt, grad_qfim_block_diag_mean, color="cyan", alpha=0.3)
#axs[2].plot(listt, smoothed_grad_qfim_block_diag, label="NPG", color="purple", alpha=0.7)

#axs[2].legend()
axs[2].set_ylabel('(Q)NPG norm')
axs[2].set_xlabel('Episodes/Batch size')

fig.subplots_adjust(right=0.7)

# Now, create the legend using the fourth subplot's lines and labels
# and place it to the right of the fifth subplot using bbox_to_anchor
lines, labels = axs[0].get_legend_handles_labels()
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


