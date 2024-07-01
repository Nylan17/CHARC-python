import numpy as np
import testMC
#from Support_files.reservoir_type.Multi_reservoir.RoR_architecture.assessESNonTask import assess_esn_on_task

def evaluate_individual(individual, config):
    metrics = []
    num_timesteps = round(individual['total_units'] * 1.5) + config['wash_out']
    MC_num_timesteps = 500 + config['wash_out'] * 2
    n_input_units = individual['n_input_units']

    for metric_item in config['metrics']:
        if metric_item == 'KR':
            input_sequence = 2 * np.random.rand(num_timesteps, n_input_units) - 1
            M = assess_individual(individual, input_sequence, config)
            M[np.isnan(M)] = 0
            M[np.isinf(M)] = 0
            s = np.linalg.svd(M, compute_uv=False)
            kernel_rank = calculate_effective_rank(s)
            metrics.append(kernel_rank)

        elif metric_item == 'GR':
            input_sequence = 0.5 + 0.1 * np.random.rand(num_timesteps, n_input_units) - 0.05
            G = assess_individual(individual, input_sequence, config)
            G[np.isnan(G)] = 0
            G[np.isinf(G)] = 0
            s = np.linalg.svd(G, compute_uv=False)
            gen_rank = calculate_effective_rank(s)
            metrics.append(gen_rank)

        elif metric_item == 'linearMC':
            mc_seed = 1
            temp_MC = testMC.test_memory_capacity(individual, config, mc_seed, MC_num_timesteps)
            MC = np.mean(temp_MC)
            metrics.append(MC)

        # Add more metrics as needed...
    
    return metrics

#def assess_individual(individual, input_sequence, config): TODO
    # Placeholder for actual assessment function like a neural network forward pass
    # Return matrix/vector from processing input_sequence through the model defined by 'individual'
#    return np.random.rand(input_sequence.shape[0], input_sequence.shape[1])  # Placeholder

def calculate_effective_rank(s, threshold=0.99):
    # Calculate the effective rank based on the singular values
    full_rank_sum = np.sum(s)
    tmp_rank_sum = 0
    e_rank = 0
    for value in s:
        tmp_rank_sum += value
        e_rank += 1
        if tmp_rank_sum >= threshold * full_rank_sum:
            break
    return e_rank
