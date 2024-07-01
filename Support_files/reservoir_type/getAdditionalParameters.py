def get_additional_parameters(config):
    # Set Default parameters
    config['mutate_type'] = 'gaussian'
    config['num_reservoirs'] = len(config['num_nodes'])
    config['leak_on'] = 1
    config['add_input_states'] = 1
    config['sparse_input_weights'] = 1
    config['sparsity'] = [1/x for x in config['num_nodes']]
    config['internal_sparsity'] = 0.1
    config['input_weight_initialisation'] = 'norm'
    config['connecting_sparsity'] = 0.1
    config['internal_weight_initialisation'] = 'norm'
    config['evolve_output_weights'] = 1
    config['evolve_feedback_weights'] = 0
    config['feedback_weight_initialisation'] = 'norm'
    config['feedback_connectivity'] = 0
    config['teacher_forcing'] = 0
    config['multi_activ'] = 0
    config['activ_list'] = ['tanh']  # assuming a simple function mapping can replace MATLAB's function handle
    config['training_type'] = 'Ridge'
    config['undirected'] = 0
    config['undirected_ensemble'] = 0
    config['scaler'] = 1
    config['discrete'] = 0
    config['noise_ratio'] = 0
    config['preprocess'] = 'scaling' if config['dataset'] != 'test_pulse' else 0
    config['run_sim'] = 0
    config['film'] = 0
    config['iir_filter_on'] = 0
    config['iir_filter_order'] = 3

    # Change/add parameters depending on reservoir type
    res_type = config['res_type']
    if res_type == 'ELM':
        config['leak_on'] = 1
    elif res_type == 'BZ':
        config['fft'] = 0
        config['evolve_output_weights'] = 0
    elif res_type in ['Graph', 'SW']:
        config['graph_type'] = ['fullLattice']
        config['self_loop'] = [1]
        config['ensemble_graph'] = 0 if res_type == 'Graph' else config.get('ensemble_graph', 0)
        config['SW_type'] = 'topology' if res_type == 'Graph' else 'topology_plus_weights'
    elif res_type == 'DNA':
        config['tau'] = 20
        config['step_size'] = 1
        config['concat_states'] = 0
    elif res_type in ['RBN', 'elementary_CA', '2D_CA']:
        config['k'] = 2 if res_type == 'RBN' else 3
        config['mono_rule'] = 1
        config['rule_list'] = ['evolveCRBN']
        config['leak_on'] = 0
        config['discrete'] = 1
        config['torus_rings'] = 1 if res_type in ['elementary_CA', '2D_CA'] else 0
        config['rule_type'] = 'Moores' if res_type == '2D_CA' else 0
    elif res_type == 'DL':
        config['preprocess'] = 0
        config['tau'] = [x * 0.2 for x in config['num_nodes']]
        config['binary_weights'] = 0
    elif res_type == 'CNT':
        config['volt_range'] = 5
        config['num_input_electrodes'] = 64
        config['num_output_electrodes'] = 32

    # Task parameters - apply task-specific parameters
    if config['dataset'] in ['autoencoder', 'attractor', 'pole_balance', 'robot']:
        config['leak_on'] = 0 if config['dataset'] in ['autoencoder', 'attractor'] else config['leak_on']
        config['add_input_states'] = 0 if config['dataset'] in ['autoencoder', 'pole_balance'] else 1
        config['sparse_input_weights'] = 0 if config['dataset'] == 'autoencoder' else 1
        config['figure_array'] = [plt.figure()] if config['dataset'] == 'autoencoder' else config.get('figure_array', [])
        
        if config['dataset'] == 'pole_balance':
            config['time_steps'] = 1000
            config['simple_task'] = 2
            config['pole_tests'] = 2
            config['velocity'] = 1
            config['run_sim'] = 0
            config['evolve_output_weights'] = 1
        elif config['dataset'] == 'robot':
            config['robot_behaviour'] = 'explore_maze'
            config['time_steps'] = 500
            config['sensor_range'] = 0.5
            config['sensor_radius'] = 2 * np.pi
            config['run_sim'] = 0
            config['robot_tests'] = 1
            config['show_robot_tests'] = 1
            config['sim_speed'] = 25
            config['bounds_x'] = 5
            config['bounds_y'] = 5
            config['num_obstacles'] = 0
            config['num_target_points'] = 1000
            config['maze_size'] = 5
        elif config['dataset'] == 'attractor':
            config['attractor_type'] = 'mackey_glass'
            config['evolve_output_weights'] = 0
            config['teacher_forcing'] = 1
            config['preprocess'] = 0
            config['noise_ratio'] = 10e-6

    return config

# Example usage
config = {
    'num_nodes': [100],
    'dataset': 'test_pulse',
    'res_type': 'ELMA'
}
config = get_additional_parameters(config)
