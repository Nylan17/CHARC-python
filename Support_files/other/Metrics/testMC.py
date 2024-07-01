import numpy as np
#from Support_files.reservoir_type.Multi_reservoir.RoR_architecture.assessESNonTask import assess_esn_on_task

def test_memory_capacity(individual, config, seed, data_length):
    np.random.seed(seed)

    n_internal_units = individual['total_units']
    n_output_units = n_internal_units
    n_input_units = individual['n_input_units']

    # Generate input data
    data_sequence = 2 * np.random.rand(n_input_units, data_length + 1 + n_output_units) - 1
    data_sequence = feature_normalize(data_sequence, config['preprocess'])

    if config.get('discrete', False):
        data_sequence = np.floor(np.heaviside(data_sequence, 0))

    # Prepare memory task sequences
    sequence_length = data_length // 2
    mem_input_sequence = data_sequence[:, n_output_units:data_length + n_output_units].T

    train_input_sequence = np.tile(mem_input_sequence[:sequence_length, :], (1, n_input_units))
    test_input_sequence = np.tile(mem_input_sequence[sequence_length:, :], (1, n_input_units))

    train_output_sequence = np.zeros((sequence_length, n_output_units))
    test_output_sequence = np.zeros((data_length - sequence_length, n_output_units))

    for i in range(n_output_units):
        train_output_sequence[:, i] = data_sequence[n_output_units - 1 - i:data_length + n_output_units - 1 - i]
        test_output_sequence[:, i] = data_sequence[n_output_units - 1 - i + sequence_length:data_length + n_output_units - 1 - i]

    # Assess the individual to get reservoir states
    states = assess_individual(individual, train_input_sequence, config)
    test_states = assess_individual(individual, test_input_sequence, config)

    # Ridge regression to train output weights
    reg_params = [10e-1, 10e-3, 10e-5, 10e-7, 10e-9, 10e-11]
    best_error = float('inf')
    best_output_weights = None

    for reg in reg_params:
        output_weights = np.linalg.pinv(states.T @ states + reg * np.eye(states.shape[1])) @ states.T @ train_output_sequence
        predicted_test = test_states @ output_weights
        test_error = calculate_error(predicted_test, test_output_sequence, config)

        if test_error < best_error:
            best_error = test_error
            best_output_weights = output_weights

    # Calculate memory capacity
    MC = calculate_memory_capacity(test_states, best_output_weights, test_output_sequence, config)

    return MC

def calculate_memory_capacity(test_states, output_weights, test_output_sequence, config):
    predictions = test_states @ output_weights
    MC_k = []

    for i in range(test_output_sequence.shape[1]):
        MC_k.append(calculate_single_mc(predictions[:, i], test_output_sequence[:, i]))

    return np.sum(MC_k)

def calculate_single_mc(predictions, targets):
    # Calculate single memory capacity component
    cov = np.cov(predictions, targets)
    coVar = cov[0, 1]
    outVar = np.var(predictions)
    targVar = np.var(targets)

    return (coVar ** 2) / (outVar * targVar) if outVar * targVar > 0 else 0

def feature_normalize(data, preprocess):
    # Implement feature normalization based on preprocess settings
    if preprocess == 'scaling':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    return data

def calculate_error(predictions, targets, config):
    return np.mean((predictions - targets) ** 2)

#def assess_individual(individual, input_sequence, config): TODO
    # This should mimic the behavior of the `assessFcn` in MATLAB
    # For simplicity, returning a random matrix
#    return np.random.rand(input_sequence.shape[0], individual['total_units'])
