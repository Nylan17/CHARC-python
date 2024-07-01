import numpy as np

def mutate_ror(offspring, config):
    # Helper function to mutate weights
    def mutate_weight(value, config):
        if config['mutate_type'] == 'gaussian':
            value += np.random.randn(*value.shape) * 0.15
        elif config['mutate_type'] == 'uniform':
            value = 2 * np.random.rand(*value.shape) - 1
        return value

    # Mutation for input scaling and leak rates
    for attr in ['input_scaling', 'leak_rate', 'W_scaling', 'output_weights', 'feedback_scaling']:
        array = np.array(offspring[attr])
        mutation_mask = np.random.rand(*array.shape) < config['mut_rate']
        array[mutation_mask] = mutate_weight(array[mutation_mask], config)
        offspring[attr] = array.tolist()  # Assuming offspring is a dict and the attributes are stored as lists

    # Mutation for matrices like input_weights, W, feedback_weights
    for key in ['input_weights', 'W', 'feedback_weights']:
        for i in range(len(offspring[key])):
            for j in range(len(offspring[key][i])):
                matrix = np.array(offspring[key][i][j])
                mutation_mask = np.random.rand(*matrix.shape) < config['mut_rate']
                matrix[mutation_mask] = mutate_weight(matrix[mutation_mask], config)
                offspring[key][i][j] = matrix.tolist()

    # Mutate activation functions if multiple are allowed
    if config.get('multi_activ', False):
        for i in range(len(offspring['activ_Fcn'])):
            activ_array = np.array(offspring['activ_Fcn'][i])
            mutation_mask = np.random.rand(activ_array.shape[0]) < config['mut_rate']
            activ_array[mutation_mask] = np.random.choice(config['activ_list'], size=mutation_mask.sum())
            offspring['activ_Fcn'][i] = activ_array.tolist()

    # Special handling for IIR filter weights if used
    if config.get('iir_filter_on', False):
        for i in range(len(offspring['iir_weights'])):
            for j in range(len(offspring['iir_weights'][i])):
                array = np.array(offspring['iir_weights'][i][j])
                mutation_mask = np.random.rand(*array.shape) < config['mut_rate']
                modified = mutate_weight(array[mutation_mask], config)
                alpha = np.sin(modified) * np.sinh((np.log(2) / 2) * 3 * np.random.rand(*modified.shape) * modified / np.sin(modified))
                if j == 0:  # Feedforward part
                    offspring['iir_weights'][i][j] = (alpha * np.array([1, 0, -1])).tolist()
                else:  # Feedback part
                    offspring['iir_weights'][i][j] = np.array([1 + alpha, -2 * np.cos(modified), 1 - alpha]).tolist()

    return offspring

# Configuration for mutation
config = {
    'mut_rate': 0.05,
    'mutate_type': 'gaussian',
    'activ_list': ['tanh', 'relu', 'sigmoid'],  # Example activation functions
    'multi_activ': True,
    'iir_filter_on': True
}

# Example offspring structure (simplified)
offspring = {
    'input_scaling': [1.0],
    'leak_rate': [0.1],
    'W_scaling': [[1.0]],
    'output_weights': [[1.0]],
    'feedback_scaling': [0.5],
    'input_weights': [[[0.1]]],
    'W': [[[0.1]]],
    'feedback_weights': [[[0.1]]],
    'activ_Fcn': [['tanh']],
    'iir_weights': [[[1, 0, -1]]]
}

# Perform mutation
mutated_offspring = mutate_ror(offspring, config)
