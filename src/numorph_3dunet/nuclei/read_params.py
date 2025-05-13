import pandas as pd

def read_params_from_csv(csv_path):
    #set default params: 
    
    default_params = {
        'chunk_size': [112, 112, 32],
        'chunk_overlap': [16, 16, 8],
        'pred_threshold': 0.5,
        'int_threshold': 200,
        'normalize_intensity': True,
        'resample_chunks': False,
        'use_mask': False,
        'mask_file': '',
        'acquired_img_resolution': [0.75, 0.75, 4],
        'trained_img_resolution': [0.75, 0.75, 2.5],
        'resample_resolution': 25,
        'tree_radius': 2,
        'measure_coloc': False,
        'n_channels': 1,
        'gpu': 0
    }
    # Read CSV file into DataFrame
    params_df = pd.read_csv(csv_path, index_col=0)
    
    # Convert to dictionary (Series actually)
    #params_dict = params_df[params_df.columns[0]]
    params_dict = params_df[params_df.columns[0]].to_dict()
    

    # check all params and replace nan values with defaults 
    for param, default_val in default_params.items():
        if param not in params_dict or pd.isna(params_dict[param]) or params_dict[param] == 'None' or params_dict[param] == '':
            print(f"Using default value for {param}: {default_val}")
            params_dict[param] = default_val


 
    # Handle list parameters like chunk_size, chunk_overlap and convert them to the correct type
    list_params = ['chunk_size', 'chunk_overlap', 'acquired_img_resolution', 'trained_img_resolution', 'resample_resolution']
    for param in list_params:
        if not isinstance(params_dict[param], list):
            try:
                # Convert string "16,16,8" to list [16,16,8]
                params_dict[param] = [int(x.strip()) for x in str(params_dict[param]).split(',')]
            except ValueError:
                print(f"Error converting {param} to list. Using default value.")
                params_dict[param] = default_params[param]
    
    # Convert boolean parameters
    bool_params = ['normalize_intensity', 'resample_chunks', 'use_mask', 'measure_coloc']
    for param in bool_params:
        if not isinstance(params_dict[param], bool):
            if str(params_dict[param]).lower() == 'true':
                params_dict[param] = True
            elif str(params_dict[param]).lower() == 'false':
                params_dict[param] = False
            else:
                print(f"Invalid boolean value for {param}. Using default value.")
                params_dict[param] = default_params[param]
    
    # Convert numeric parameters
    num_params = ['pred_threshold', 'int_threshold', 'tree_radius', 'n_channels', 'gpu']
    for param in num_params:
        if not isinstance(params_dict[param], (int, float)):
            try:
                params_dict[param] = float(params_dict[param])
            except (ValueError, TypeError):
                print(f"Invalid numeric value for {param}. Using default value.")
                params_dict[param] = default_params[param]
    
    return params_dict