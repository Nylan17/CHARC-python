def get_error(error_to_check, individual):
    try:
        if error_to_check == 'train':
            error = individual['train_error']
        elif error_to_check == 'val':
            error = individual['val_error']
        elif error_to_check == 'test':
            error = individual['test_error']
        elif error_to_check == 'train&val':
            error = individual['train_error'] + individual['val_error']
        elif error_to_check == 'val&test':
            error = individual['val_error'] + individual['test_error']
        elif error_to_check == 'train&val&test':
            error = individual['train_error'] + individual['val_error'] + individual['test_error']
        return error
    except KeyError:
        print(f"Error: Missing error data in individual for requested '{error_to_check}'.")
        return None
