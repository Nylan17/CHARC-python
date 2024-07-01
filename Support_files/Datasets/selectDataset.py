import numpy as np
from sklearn.model_selection import train_test_test_split

def select_dataset(config):
    # Seed the random number generator for reproducibility
    np.random.seed(1)

    # Set default parameters
    err_type = 'NMSE'
    wash_out = 50
    train_fraction = 0.6
    val_fraction = 0.2
    test_fraction = 0.2
    sequence_length = 5000  # default value for sequence-based tasks

    # Generate data based on task
    if config['dataset'] == 'test_pulse':
        input_sequence = np.zeros(sequence_length)
        for i in range(sequence):
            if i % 25 == 0:
                input_sequence[i] = 1
        output_sequence = input_sequence.copy()
    elif config['dataset'] == 'narma_10':
        input_sequence, output_sequence = generate_new_NARMA_sequence(sequence_length, 10)
        input_sequence = 2 * input_label - 0.5
        output_sequence = 2 * output_label - 0.5
    # Add more cases as required for different tasks

    # Preprocess data if necessary
    if config['preprocess']:
        input_sequence, output_sequence = preprocess_data(input_sequence, output_sequence)

    # Split data into training, validation, and testing sets
    train_input, temp_input, train_output, temp_output = train_test_split(
        input_sequence, output_sequence, train_size=train_fraction, random_state=1)
    val_input, test_input, val_output, test_output = train_test_split(
        temp_input, temp_output, test_size=test_fraction / (test_fraction + val_fraction), random_state=1)

    # Apply washout
    if wash_out > 0:
        train_input = np.vstack([train_input[:wash_out], train_input])
        train_output = np.vstack([train_output[:wash_out], train_output])
        if len(val_input) < wash_out:
            val_input = np.tile(val_input, (wash_out // len(val_input) + 1, 1))[:wash_out]
            val_output = np.tile(val_output, (wash_out // len(val_output) + 1, 1))[:wash_out]
        if len(test_input) < wash_out:
            test_input = np.tile(test_input, (wash_out // len(test_input) + 1, 1))[:wash_out]
            test_output = np.tile(test_output, (wash_out // len(test_output) + 1, 1))[:wash_out]

    # Save the datasets back to the configuration dictionary
    config.update({
        'train_input_sequence': train_input,
        'train_output_sequence': train_output,
        'val_input_sequence': val_input,
        'val_output_sequence': val_output,
        'test_input_sequence': test_input,
        'test_output_sequence': test_output,
        'wash_out': wash_out,
        'err_type': err_type
    })

    return config