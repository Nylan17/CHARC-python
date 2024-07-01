import numpy as np

def collect_ror_states(individual, input_sequence, config, target_output=None):
    # Check for single input entries and adjust accordingly
    if input_sequence.shape[0] == 1:
        input_sequence = np.vstack([np.zeros_like(input_sequence), input_sequence])
    
    # Initialize states and other variables
    states = {}
    x = {}
    for i in range(config['num_reservoirs']):
        states[i] = np.zeros((input_sequence.shape[0], individual['nodes'][i]))
        x[i] = np.zeros_like(states[i])
        if input_sequence.shape[0] == 2:
            states[i][0, :] = individual['last_state'][i]
    
    # Preassign activation function calls
    activ_functions = {}
    if config['multi_activ']:
        for i in range(config['num_reservoirs']):
            activ_functions[i] = [config['activ_list'][index] for index in individual['activ_Fcn'][i]]

    # Main loop to calculate states
    for n in range(1, input_sequence.shape[0]):
        for i in range(config['num_reservoirs']):
            for k in range(config['num_reservoirs']):
                x[i][n, :] += (individual['W'][i][k] * individual['W_scaling'][i][k]) @ states[k][n-1, :]

            if config['multi_activ']:
                for p, func in enumerate(activ_functions[i]):
                    states[i][n, p] = func(x[i][n, p] + (individual['input_weights'][i][p, :] * individual['input_scaling'][i]) @ np.hstack([individual['bias_node'], input_sequence[n, :]]))
            else:
                if config.get('teacher_forcing', False) and np.sum(input_sequence[n-1:n+1, :]) != 0:
                    target_input = target_output[n-1, :] + config['noise_ratio'] * np.random.rand(*target_output[n-1, :].shape)
                    states[i][n, :] = individual['activ_Fcn'][i]((individual['input_weights'][i] * individual['input_scaling'][i]) @ np.hstack([individual['bias_node'], target_input]) + x[i][n, :])
                else:
                    states[i][n, :] = individual['activ_Fcn'][i]((individual['input_weights'][i] * individual['input_scaling'][i]) @ np.hstack([individual['bias_node'], input_sequence[n, :]]) + x[i][n, :])

    # Account for leakage
    #if config.get('leak_on', False):
    #    states = get_leak_states(states, individual, input_sequence, config)

    # Concatenate all states for output
    final_states = np.hstack([states[i] for i in range(config['num_reservoirs'])])
    
    # Save last state
    for i in range(config['num_reservoirs']):
        individual['last_state'][i] = states[i][-1, :]

    # Add input states if configured
    if config.get('add_input_states', False):
        final_states = np.hstack([final_states, input_sequence])

    # Remove washout period
    if input_sequence.shape[0] == 2:
        final_states = final_states[-1, :]  # Remove washout for single step prediction
    else:
        final_states = final_states[config['wash_out'] + 1:, :]  # Standard washout removal

    return final_states, individual


