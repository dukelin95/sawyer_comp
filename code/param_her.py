# DDPG
action_noise = None
normalize = True
nb_train_steps = 50
nb_rollout_steps = 100
batch_size = 256
critic_l2_reg = 0.01
buffer_size=int(1e6)
random_exploration=0.2

# how long to train
total_timesteps = int(0.25e6)