def select_reservoir_type(config):
    # Determine the appropriate reservoir type
    res_type = 'Heterotic' if isinstance(config['res_type'], list) else config['res_type']

    # Define pointers to functions for each reservoir type
    reservoir_functions = {
        'ELM': ('create_elm', 'collect_elm_states', 'mutate_ror', 'recomb_ror'),
        'RoR': ('create_ror', 'collect_ror_states', 'mutate_ror', 'recomb_ror'),
        'Pipeline': ('create_pipeline', 'collect_pipeline_states', 'mutate_ror', 'recomb_ror'),
        'Ensemble': ('create_ensemble', 'collect_ensemble_states', 'mutate_ror', 'recomb_ror'),
        'Graph': ('create_graph_reservoir', 'assess_graph_reservoir', 'mutate_sw', 'recomb_sw'),
        'BZ': ('create_bz_reservoir', 'assess_bz_reservoir', 'mutate_bz', 'recomb_bz'),
        'RBN': ('create_rbn_reservoir', 'assess_rbn_reservoir', 'mutate_rbn', 'recomb_rbn'),
        'elementary_CA': ('create_rbn_reservoir', 'assess_rbn_reservoir', 'mutate_rbn', 'recomb_rbn'),
        '2D_CA': ('create_rbn_reservoir', 'assess_rbn_reservoir', 'mutate_rbn', 'recomb_rbn'),
        'DNA': ('create_dna_reservoir', 'assess_dna_reservoir', 'mutate_dna', 'recomb_dna'),
        'DL': ('create_dl_reservoir', 'collect_dl_states', 'mutate_dl', 'recomb_dl'),
        'CNT': ('create_cnt', 'collect_cnt_states', 'mutate_cnt', 'recomb_cnt'),
        'Wave': ('create_wave', 'collect_wave_states', 'mutate_wave', 'recomb_wave'),
        'GOL': ('create_gol', 'assess_gol', 'mutate_gol', 'recomb_gol'),
        'Ising': ('create_ising', 'assess_ising', 'mutate_ising', 'recomb_ising'),
        'SW': ('create_sw_reservoir', 'collect_ror_states', 'mutate_sw', 'recomb_sw'),
        'Heterotic': ('create_heterotic', 'assess_heterotic', 'mutate_heterotic', 'recomb_heterotic')
    }

    # Set function pointers in config
    create_fn, assess_fn, mutate_fn, recomb_fn = reservoir_functions[res_type]
    config['create_fn'] = globals()[create_fn]
    config['assess_fn'] = globals()[assess_fn]
    config['mutate_fn'] = globals()[mutate_fn]
    config['recomb_fn'] = globals()[recomb_postfn]

    # Set default test function, with special case for CNT
    config['test_fn'] = globals()['test_hardware_reservoir'] if res_type == 'CNT' else globals()['test_reservoir']

    return config

# Example usage
config = {
    'res_type': 'RoR',
    'num_nodes': [100]
}
config = select_reservoir_type(config)
