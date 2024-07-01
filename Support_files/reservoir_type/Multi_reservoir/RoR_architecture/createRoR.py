import numpy as np
from scipy.sparse import random as sparse_random

def create_ror(config):
    population = []
    for pop_index in range(config['pop_size']):
        individual = {
            'train_error': 1,
            'val_error': 1,
            'test_error': 1,
            'bias_node': 1,
            'n_input_units': 1 if 'train_input_sequence' not in config else config['train_input_sequence'].shape[1],
            'n_output_units': 1 if 'train_output_sequence' not in config else config['train_output_sequence'].shape[1],
            'nodes': [],
            'input_scaling': [],
            'leak_rate': [],
            'input_weights': [],
            'activ_Fcn': [],
            'iir_weights': [],
            'last_state': [],
            'W': {},
            'output_weights': None,
            'feedback_weights': None,
            'total_units': 0,
            'behaviours': []
        }

        for i in range(config['num_reservoirs']):
            num_units = config['num_nodes'][i]
            individual['nodes'].append(num_units)
            individual['input_scaling'].append(2 * np.random.rand() - 1)
            individual['leak_rate'].append(np.random.rand())

            # Input weights initialization
            if config['input_weight_initialisation'] == 'norm':
                input_weights = np.random.randn(num_units, individual['n_input_units'] + 1)
            elif config['input_weight_initialisation'] == 'uniform':
                input_weights = 2 * np.random.rand(num_units, individual['n_input_units'] + 1) - 1
            elif config['input_weight_initialisation'] == 'orth':
                input_weights = np.linalg.qr(np.random.rand(num_units, individual['n_input_units'] + 1))[0]

            individual['input_weights'].append(input_weights)
            
            # Internal weights and connectivity for all reservoirs
            for j in range(config['num_reservoirs']):
                if i == j:
                    connectivity = config['internal_sparsity']
                else:
                    connectivity = config['connecting_sparsity']

                internal_weights = 2 * np.random.rand(num_units, num_units) - 1  # Simplified uniform initialization
                individual['W'][(i, j)] = internal_weights
            
            individual['last_state'].append(np.zeros(num_units))
        
        # Output weights
        total_inputs = sum(individual['nodes']) + individual['n_input_units']
        individual['output_weights'] = 2 * np.random.rand(total_inputs, individual['n_output_units']) - 1
        
        # Optionally, initialize feedback weights
        if config.get('evolve_feedback_weights', False):
            feedback_weights = 2 * np.random.rand(total_inputs, individual['n_input_units']) - 1
            individual['feedback_weights'] = feedback_weights

        population.append(individual)
    
    return population
