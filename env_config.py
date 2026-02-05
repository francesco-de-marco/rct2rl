NUM_ROLLOUT_WORKERS = 2 #8
#if NUM_ROLLOUT_WORKERS < 8:
#    input("Warning: Rollout workers less than 8.")

env_config = {
    'map_size': (87, 87), # (87, 87), #(18, 18), #(33, 33), # (12, 12), # (12, 12),
    "action_type": 'wide', # 'turtle',
    "actions": ['add'], #['replace', 'add'], #['replace'],
    "action_space_type": 'normal', # 'extended',
    "random_invalid_rides": True,
    "headless": True,
    
    'park_filename': 'small_parks/Electric Fields.SC6',
    "use_park_dataset": False, # True,
    #"load_model": "checkpoint",
    
    "render": False,
    "verbose": False,

    # Forza la fine dell'episodio dopo 12 mesi (step) per debugging rapido
    #"max_timesteps": 12,
}
