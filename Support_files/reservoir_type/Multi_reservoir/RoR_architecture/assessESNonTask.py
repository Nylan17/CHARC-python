import numpy as np
from collectRoRStates import collect_ror_states

def assess_esn_on_task(esn_major, esn_minor, train_input_sequence, train_output_sequence, val_input_sequence, val_output_sequence, test_input_sequence, test_output_sequence, n_forget_points, leak_on, err_type, res_type):
    # Setup initial weights for minor ESNs based on input dimensions
    if train_input_sequence.shape[1] > 1:
        np.random.seed(1)
        esn_major['n_input_units'] = train_input_sequence.shape[1]
        for i in range(len(esn_minor)):
            esn_minor[i]['input_weights'] = 2 * np.random.rand(esn_minor[i]['n_internal_units'], train_input_sequence.shape[1] + 1) - 1

    # Collect states depending on the reservoir type
    if res_type == 'RoR':
        states_ext = collect_deep_states_non_ia(esn_major, esn_minor, train_input_sequence, n_forget_points, leak_on)
        states_ext_val = collect_deep_states_non_ia(esn_major, esn_minor, val_input_sequence, n_forget_points, leak_on)
    elif res_type == 'RoR_IA':
        states_ext = collect_deep_states_ia(esn_major, esn_minor, train_input_sequence, n_forget_points, leak_on)
        states_ext_val = collect_deep_states_ia(esn_major, esn_minor, val_input_sequence, n_forget_points, leak_on)

    # Train and evaluate model
    reg_train_error = []
    reg_val_error = []
    reg_weights = []
    reg_param = [10**-1, 10**-3, 10**-5, 10**-7, 10**-9]

    for rp in reg_param:
        esn_major['reg_param'] = rp
        output_weights = train_output_sequence[n_forget_points+1:].T @ states_ext @ np.linalg.inv(states_ext.T @ states_ext + rp * np.eye(states_ext.T @ states_ext.shape[0]))
        output_sequence = states_ext @ output_weights.T
        reg_train_error.append(calculate_error(output_sequence, train_output_sequence[n_forget_points+1:], err_type))

        output_val_sequence = states_ext_val @ output_weights.T
        reg_val_error.append(calculate_error(output_val_sequence, val_output_sequence[n_forget_points+1:], err_type))
        reg_weights.append(output_weights)

    best_idx = np.argmin(np.sum(reg_val_error, axis=1))
    train_error = reg_train_error[best_idx]
    val_error = reg_val_error[best_idx]
    test_weights = reg_weights[best_idx]

    # Testing phase
    if res_type == 'RoR':
        test_states = collect_deep_states_non_ia(esn_major, esn_minor, test_input_sequence, n_forget_points, leak_on)
    elif res_type == 'RoR_IA':
        test_states = collect_deep_states_ia(esn_major, esn_minor, test_input_sequence, n_forget_points, leak_on)

    test_sequence = test_states @ test_weights.T
    test_error = calculate_error(test_sequence, test_output_sequence[n_forget_points+1:], err_type)

    return test_error


def calculate_error(predicted, actual, err_type='mse'):
    if err_type.lower() == 'mse':
        return np.mean((predicted - actual) ** 2)
    else:
        raise ValueError("Unsupported error type. Please implement the required metric.")
