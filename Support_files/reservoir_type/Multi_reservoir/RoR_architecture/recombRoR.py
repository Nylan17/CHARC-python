import numpy as np

def recomb_ror(winner, loser, config):
    # Helper function to perform element-wise recombination
    def recombine_arrays(winner_array, loser_array, rec_rate):
        assert len(winner_array) == len(loser_array), "Arrays must be of equal length"
        indices = np.random.rand(len(loser_array)) < rec_rate  # Determine indices to recombine based on rec_rate
        loser_array[indices] = winner_array[indices]  # Perform recombination
        return loser_array

    # Recombine simple attributes
    for attr in ['input_scaling', 'leak_rate', 'W_scaling', 'output_weights', 'feedback_scaling']:
        winner_array = np.array(winner[attr])
        loser_array = np.array(loser[attr])
        loser[attr] = recombine_arrays(winner_array, loser_array, config['rec_rate']).tolist()

    # Recombine weights and matrices
    for key in ['input_weights', 'W', 'feedback_weights']:
        for i in range(len(winner[key])):
            for j in range(len(winner[key][i])):
                winner_matrix = np.array(winner[key][i][j])
                loser_matrix = np.array(loser[key][i][j])
                loser[key][i][j] = recombine_arrays(winner_matrix, loser_matrix, config['rec_rate']).tolist()

    # Recombine activation functions if applicable
    if config.get('multi_activ', False):
        for i in range(len(winner['activ_Fcn'])):
            winner_activ = np.array(winner['activ_Fcn'][i])
            loser_activ = np.array(loser['activ_Fcn'][i])
            loser['activ_Fcn'][i] = recombine_arrays(winner_activ, loser_activ, config['rec_rate']).tolist()

    # Special handling for IIR filter weights if used
    if config.get('iir_filter_on', False):
        for i in range(len(winner['iir_weights'])):
            for k in range(len(winner['iir_weights'][i])):
                winner_filter = np.array(winner['iir_filters'][i][k])
                loser_filter = np.array(loser['iir_filters'][i][k])
                loser['iir_weights'][i][k] = recombine_arrays(winner_filter, loser_filter, config['rec_rate']).tolist()

    return loser

# Example usage:
config = {
    'rec_rate': 0.5,
    'multi_activ': True,
    'iir_filter_on': True
}
winner = {
    # Assume these are filled with actual data structures representing the individual
    'input_scaling': [1.0],
    'leak_rate': [0.1],
    # More attributes...
}
loser = {
    # Assume these are filled with actual data structures representing the individual
    'input_scaling': [0.5],
    'leak_rate': [0.05],
    # More attributes...
}

loser = recomb_ror(winner, loser, config)