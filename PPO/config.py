class AgentConfig:
    # Learning
    gamma = 0.99
    update_freq = 1
    eval_freq = 5
    k_epoch = 3
    learning_rate = 0.01
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01

    # Memory
    memory_size = 100