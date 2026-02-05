import os
import json
import random
import pickle
import platform
from glob import glob
from pprint import pprint
from collections import defaultdict, OrderedDict, Counter

import ray
import numpy as np
from scipy.ndimage import convolve
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from paths import *
from bridge import Bridge
from gen_envs.consts import *
from gen_envs.helpers import *
from gen_envs.base import BasePathEnv, Target

class Ride:
    def __init__(self, entrance_pos, exit_pos, ride_type, price, entrance_dir, exit_dir, track_dir, positions=None, ride_pos=None, all_positions=None, entry_name="default"):
        self.entrance_pos = entrance_pos
        assert len(entrance_pos) == 3, entrance_pos #and entrance_pos[-1] > 0
        self.exit_pos = exit_pos
        assert exit_pos is None or len(exit_pos) == 3, exit_pos
        self.ride_type = ride_type
        self.price = price
        self.positions = positions if positions is not None else []
        self.entrance_dir = entrance_dir
        self.exit_dir = exit_dir
        self.track_dir = track_dir
        self.ride_pos = ride_pos
        self.all_positions = all_positions
        self.entry_name = entry_name

MAX_NUM_MONTHS = 48
MAX_LOAN = 200000
MAX_HAPPINESS = 255
MAX_NUM_GUESTS = 1500
MAX_COPIES_PER_RIDE = MAX_NUM_MONTHS * 2
MAX_WEEKLY_PROFIT_FACTOR = 450 #250
Park = namedtuple('Park', 'filename starting_map curriculum_num saved_fname')
StartingMap = namedtuple('StartingMap', 'starting_map map_dy map_dx paths rides shops size_y size_x all_owned_positions total_num_rides occupied_map num_months objective_num_guests free_park_entry zs owned_map')

def get_parks(config):
    park_filenames = []
    dataset = config.get('use_dataset', None)
    if config.get('use_park_dataset', False):
        for park_filename in sum([ glob(PARK_DIR+"/"+x) for x in ("*.park","*.sc6","*.sc4") ], []):
            park_filenames.append(park_filename)
    elif dataset is not None:
        for park_filename in sum([ glob(f"{dataset}/*."+x) for x in ("park","sc6","sc4", "SC6") ], []):
            park_filenames.append(park_filename)
    elif 'park_filename' in config:
        park_filenames.append(config['park_filename'])
    else:
        park_filenames.append(DEFAULT_PARK)
    return park_filenames

def normalize(value, min_old=0.10, max_old=1, min_new=1, max_new=200):
    normalized_value = ((value - min_old) / (max_old - min_old)) * (max_new - min_new) + min_new
    return normalized_value

def gaussian_list(n, X, peak_index, smoothness=1.0):
    """
    Returns a list of n integers that sum to X.
    
    The values are arranged in a roughly Gaussian (bell-curve) shape with the peak
    at index `peak_index`. The `smoothness` parameter scales the width of the curve:
      - smoothness > 1.0  --> a wider (smoother) distribution,
      - smoothness < 1.0  --> a narrower (sharper) distribution.
    
    The amplitude A is determined so that a continuous Gaussian would have
    an area ≈ X:
        A * sqrt(2*pi)*sigma ≈ X,
    where sigma = smoothness * (n/8). Any rounding error is fixed by adjusting the
    peak element.
    
    Parameters:
      n           : int    -- total number of elements in the list.
      X           : int    -- desired total sum.
      peak_index  : int    -- the index at which the Gaussian peaks (should be 0 <= peak_index < n).
      smoothness  : float  -- factor controlling the spread (default 1.0).
    
    Returns:
      List[int] that sums to X.
    """
    
    # Choose sigma based on the length of the list and the smoothness factor.
    # n/8 is a rough baseline that tends to keep most of the distribution within the list.
    sigma = smoothness * (n / 8.0)
    
    # Compute the amplitude A such that the total area approximates X.
    # The area under a continuous Gaussian is A * sqrt(2*pi)*sigma.
    A = X / (math.sqrt(2 * math.pi) * sigma)
    
    # Generate the raw Gaussian values for each index.
    raw = []
    for j in range(n):
        val = A * math.exp(-((j - peak_index) ** 2) / (2 * sigma**2))
        raw.append(val)
    
    # Compute the sum of the raw values.
    raw_sum = sum(raw)
    
    # Scale the raw values so that they sum exactly to X (or very nearly).
    scale_factor = X / raw_sum
    scaled = [v * scale_factor for v in raw]
    
    # Round the scaled values to integers.
    int_vals = [int(round(v)) for v in scaled]
    
    # Adjust for rounding error by adding the leftover to the peak element.
    current_sum = sum(int_vals)
    diff = X - current_sum
    int_vals[peak_index] += diff
    
    return int_vals

class RCTEnv(BasePathEnv):
    ride_types = RCT_SMALL_RIDE_TYPES + RCT_VERY_SMALL_RIDE_TYPES + RCT_5x1_RIDE_TYPES + RCT_7x1_RIDE_TYPES + RCT_LARGE_RIDE_TYPES + RCT_SHOP_RIDE_TYPES
    
    # note: cur vals will cap at max target vals
    target_ranges = [
        Target('park_rating', 1, 1, False, 1),

        Target('avg_happiness', 1, 1, False, 1),
        *[Target(f'avg_happiness_month{k}', 1, 1, True, 1) for k in range(MAX_NUM_MONTHS)],

        Target('num_guests', 1, 1, False, 1),
        *[Target(f'num_guests_month{k}', 1, 1, True, 2) for k in range(MAX_NUM_MONTHS)],

        Target('park_value', 1, 1, False, 1), # park value is worth of all rides; diminishes with age of rides # 7*numguests + ridevalues         
        Target('company_value', 1, 1, False, 1), # company value is park value plus cash on hand

        Target('cash', 1, 1, False, 1),
        Target('ride_diversity', 1, 1, True, 5.0),  # ATTIVATO: penalizza spam giostre identiche
        Target('won_objective', 1, 1, True, 3),
        Target('num_added_rides', 1, 1, False, 1),
        #Target('loan', 1, 1, True, 1),

        Target('num_hungry_thoughts', 1, 1, True, 1),
        Target('num_thirsty_thoughts', 1, 1, True, 1),
        Target('num_toilet_thoughts', 1, 1, True, 1),
        Target('num_cantfind_thoughts', 1, 1, True, 1),
        Target('num_notpaying_thoughts', 1, 1, True, 1),
    ]
    active_target_ranges = [target for target in target_ranges if target.active]
    channels = {
        'pct_to_objective_end': 1,
        'norm_month': 1,
        'rides': len(ride_types),
        'ride_validities': len(ride_types),
        'ride_quantities': len(ride_types),
        'prices': 1,
        'popularity': 1,
        'excitement': 1,
        'intensity': 1,
        'nausea': 1,
        'profit': 1,
        'visit_counts': 1,
        'paths': 1,
        'sloped_paths': 1,
        'queues': 1,
        'zs': 1,
        'occupied_map': 1,
        'cur_values_mat': len(target_ranges),
        'num_guests_for_objective': 1,
        'awards': 17,
        'ride_schedule': MAX_NUM_MONTHS,
    }
    grid_channels = ['rides', 'prices', 'popularity', 'excitement', 'intensity', 'nausea', 'profit', 'visit_counts', 'paths', 'sloped_paths', 'queues', 'zs', 'occupied_map']
    flat_channels = []
    for channel in channels:
        if channel not in grid_channels:
            flat_channels.append(channel)
    num_grid_channels, num_flat_channels = 0, 0
    for channel in grid_channels:
        num_grid_channels += channels[channel]
    for channel in flat_channels:
        num_flat_channels += channels[channel]
    #num_channels = sum(channels.values())
    #assert num_channels == 24+15+17+len(ride_types)*2+len(target_ranges), num_channels #+len(active_target_ranges)

    def __init__(self, config):
        self.action_space_type = config.get('action_space_type', 'normal')
        assert self.action_space_type in ['normal', 'extended']

        super().__init__(config)
        
        self.map_type = config.get('map_type', 4)
        self.actions = config.get('actions')
        self.is_baseline = config.get('baseline')
        self.use_park_dataset = config.get('use_park_dataset', False)
        if 'add' in self.actions:
            self.max_timesteps = config.get('max_timesteps', -1)
        self.random_invalid_rides = config.get('random_invalid_rides', False)

        if hasattr(config, 'worker_index'):
            starting_port = str(5555 + config.worker_index*2)
        elif 'worker_index' in config:
            starting_port = str(5555 + config['worker_index']*2)
        else:
            starting_port = '5550'
        self.bridge = Bridge(starting_port, headless=config.get('headless', False))
        self.bridge.bind()
        self.bridge.start()
                
        # load parks
        if not os.path.exists(f'{BASE_DIR}/starting_map_data/'):
            #print("making starting map data directory")
            os.makedirs(f'{BASE_DIR}/starting_map_data/')
            #assert os.path.exists('starting_map_data/')
        
        # --- MODIFICA INIZIO: Recupera l'indice del worker ---
        worker_idx = 0
        if hasattr(config, 'worker_index'):
            worker_idx = config.worker_index
        elif 'worker_index' in config:
            worker_idx = config['worker_index']
        # -----------------------------------------------------

        self.parks = []
        #self.park_stats = {}
        for park_filename in get_parks(config):
            park_name = park_filename.split('/')[-1]
            
            # --- MODIFICA QUI: Aggiungi l'indice del worker al nome del file ---
            # Vecchia riga:
            # saved_fname = f'{BASE_DIR}/saved_parks/' + park_name.split('.')[0]+"_rl.park"
            
            # Nuova riga (rende il file unico per ogni processo):
            saved_fname = f'{BASE_DIR}/saved_parks/' + park_name.split('.')[0] + f"_rl_w{worker_idx}.park"
            # -------------------------------------------------------------------

            if not os.path.exists(f'{BASE_DIR}/starting_map_data/{park_name}.pkl'):
                print(f"gathering starting map data for {park_name} (Worker {worker_idx})")
                ##self.bridge.send_action('unpause')
                self.bridge.send_park(park_filename)
                if self.random_invalid_rides:
                    self.bridge.send_action("load_objects")
                #self.bridge.send_action("close_park")
                #self.bridge.send_action('pause')
                starting_map = self.get_starting_map(self.map_type, park_filename)
                
                # Ora salva il file con il nome specifico del worker
                if not os.path.exists(saved_fname):
                    self.bridge.send_action("save_park", fname=saved_fname)

                pickle.dump(starting_map, open(f'{BASE_DIR}/starting_map_data/{park_name}.pkl', 'wb'))
            else:
                starting_map = pickle.load(open(f'{BASE_DIR}/starting_map_data/{park_name}.pkl', 'rb'))
                
                # --- AGGIUNTA DI SICUREZZA ---
                # Se il file .pkl esiste già (fatto da un altro worker), assicuriamoci
                # che esista comunque il .park specifico per QUESTO worker.
                if not os.path.exists(saved_fname):
                     # Carichiamo il parco originale solo per salvarne la copia del worker
                     self.bridge.send_park(park_filename)
                     if self.random_invalid_rides:
                        self.bridge.send_action("load_objects")
                     self.bridge.send_action("save_park", fname=saved_fname)
                # -----------------------------

            curriculum_num = 1
            if park_name[0].isdigit():
                curriculum_num = int(park_name.split('-')[0])
                park_name = park_name.split('-')[1]
            self.parks.append(Park(park_filename, starting_map, curriculum_num, saved_fname))
            #self.park_stats[park_name] = { target.name: [] for target in self.target_ranges }
        self.cur_parks = [park for park in self.parks if park.curriculum_num == 1]
        self.cur_curriculum_num = 1
        self.scenario_ride_schedules = defaultdict(list)

        self.matrix = np.zeros((*self.map_size, self.num_grid_channels), dtype=np.float32)
        self.flat_vector = np.zeros((self.num_flat_channels, 1), dtype=np.float32)
        self.cur_episode = -1
    
    def normalize_targets(self, targets):
        normalize_funcs = {
            'park_rating': lambda x: min(1, x / 1000), # min(1, (x - self.initial_park_rating) / (850 - self.initial_park_rating)),
            
            # LOG-MONEY: utilità marginale decrescente per evitare money hoarding
            'cash': lambda x: np.log1p(max(0, x - self.initial_cash)) / np.log1p(self.total_num_rides_to_place * 2000),
            'park_value': lambda x: (((x - self.initial_park_value) / 10) / (self.total_num_rides + 1) / 10000),
            'company_value': lambda x: (x - self.initial_company_value) / self.num_months / (self.num_months * 3 * 2000),

            'avg_happiness': lambda x: x / MAX_HAPPINESS,
            **{f'avg_happiness_month{k}': lambda x: x / MAX_HAPPINESS for k in range(MAX_NUM_MONTHS)},

            'num_guests': lambda x: min(1, x / MAX_NUM_GUESTS), #min(1, (x - self.initial_num_guests) / (self.objective_num_guests - self.initial_num_guests)),
            **{f'num_guests_month{k}': lambda x: min(1, x / MAX_NUM_GUESTS) for k in range(MAX_NUM_MONTHS)},

            'ride_diversity': lambda x: x / len(self.available_ride_types),
            'won_objective': lambda x: x,
            'num_added_rides': lambda x: min(1, x / self.total_num_rides_to_place),
            'loan': lambda x: (MAX_LOAN - x) / MAX_LOAN,
        }
        normalized_targets = { target: normalize_funcs[target](value) for target, value in targets.items() }
        for target_name, value in normalized_targets.items():
            if not (-1 <= value <= 1) and target_name != 'num_added_rides':
                print(f"Warning: target {target_name} out of bounds: {value}. Capping.")
                normalized_targets[target_name] = max(-1, min(1, value))
            #assert -1 <= value <= 1, f"target {target_name} out of bounds: {value}"
        return normalized_targets

    def denormalize_targets(self, targets):
        denormalize_funcs = {
            'park_rating': lambda x: x * 1000,

            'cash': lambda x: x * 10 * (self.total_num_rides+1) * 5000 + self.initial_cash,
            'park_value': lambda x: x * 10 * (self.total_num_rides+1) * 10000 + self.initial_park_value,
            'company_value': lambda x: x * self.num_months * (self.num_months * 3 * 2000) + self.initial_company_value,

            'avg_happiness': lambda x: x * MAX_HAPPINESS,
            **{f'avg_happiness_month{k}': lambda x: x * MAX_HAPPINESS for k in range(MAX_NUM_MONTHS)},

            'num_guests': lambda x: x * MAX_NUM_GUESTS,
            **{f'num_guests_month{k}': lambda x: x * MAX_NUM_GUESTS for k in range(MAX_NUM_MONTHS)},
            
            'ride_diversity': lambda x: x * len(self.available_ride_types),
            'won_objective': lambda x: x,
            'num_added_rides': lambda x: x * self.total_num_rides_to_place,
            'loan': lambda x: MAX_LOAN - x * MAX_LOAN,
        }
        denormalized_targets = { target: denormalize_funcs[target](value) if target in denormalize_funcs else value for target, value in targets.items() }
        return denormalized_targets
    
    def get_action_space(self):
        num_actions = self.map_size[0] * self.map_size[1]
        if self.action_space_type == 'normal':
            action_space = Dict({
                'position_and_type': MultiDiscrete((self.map_size[0]*self.map_size[1],len(self.ride_types)+1)),
                'price': Box(0.10, 1.0, shape=(1,), dtype=np.float32),
                'queue_line_length': Discrete(5),
            })
        elif self.action_space_type == 'extended':
            ride_type_lengths = [len(self.ride_types) for _, _ in self.extended_ride_types.items()]
            action_space = Dict({
                'position_and_type': MultiDiscrete((self.map_size[0]*self.map_size[1],*ride_type_lengths)),
                'price': Box(0.10, 1.0, shape=(1,), dtype=np.float32),
            })
        return num_actions, action_space
    
    def close(self):
        self.bridge.rct_process.kill()
    
    def _fix_action_mask(self):
        assert self.action_type == 'wide'
        valid_actions = np.zeros(self.map_size, dtype=bool)

        # :, :, 0 -> set elmt to PATH      :, :, 1 -> set elmt to 0
        paths = self.paths.astype(bool)
        sloped_paths = self.sloped_paths.astype(bool)
        good_paths = np.logical_and(paths, np.logical_not(sloped_paths))
        rides = np.any(self.rides > 0, axis=2) # np.sum(self.rides, axis=-1) > 0 #.astype(bool)
        queues = self.queues.astype(bool)
        occupied_map = self.occupied_map.astype(bool)

        # this is the set of squares that are not empty.
        occupied_squares = paths | sloped_paths | rides | queues | occupied_map
        empty_squares = np.logical_not(occupied_squares) & self.owned_map.astype(bool)

        if 'add' in self.actions:
            # can turn empty squares adjacent to paths into paths.

            def compute_y(x, zs, direction='left'):
                # Ensure zs is a numpy array
                zs = np.asarray(zs)

                # Get the maximum zs value for padding
                zs_max = zs.max()

                if direction == 'left':
                    # Pad x and zs to handle boundaries for left direction
                    x_padded = np.pad(x, pad_width=((1, 1), (7, 0)), mode='constant', constant_values=False)
                    zs_padded = np.pad(zs, pad_width=((1, 1), (7, 0)), mode='constant', constant_values=zs_max + 1)
                    # Create sliding window views of shape (3, 8)
                    window_shape = (3, 8)
                    x_windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=window_shape)
                    zs_windows = np.lib.stride_tricks.sliding_window_view(zs_padded, window_shape=window_shape)
                    # Center cell is at position [1, 7]
                    zs_ij = zs_windows[:, :, 1, 7]
                elif direction == 'right':
                    # Pad x and zs to handle boundaries for right direction
                    x_padded = np.pad(x, pad_width=((1, 1), (0, 7)), mode='constant', constant_values=False)
                    zs_padded = np.pad(zs, pad_width=((1, 1), (0, 7)), mode='constant', constant_values=zs_max + 1)
                    # Create sliding window views of shape (3, 8)
                    window_shape = (3, 8)
                    x_windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=window_shape)
                    zs_windows = np.lib.stride_tricks.sliding_window_view(zs_padded, window_shape=window_shape)
                    # Center cell is at position [1, 0]
                    zs_ij = zs_windows[:, :, 1, 0]
                elif direction == 'up':
                    # Pad x and zs to handle boundaries for up direction
                    x_padded = np.pad(x, pad_width=((7, 0), (1, 1)), mode='constant', constant_values=False)
                    zs_padded = np.pad(zs, pad_width=((7, 0), (1, 1)), mode='constant', constant_values=zs_max + 1)
                    # Create sliding window views of shape (8, 3)
                    window_shape = (8, 3)
                    x_windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=window_shape)
                    zs_windows = np.lib.stride_tricks.sliding_window_view(zs_padded, window_shape=window_shape)
                    # Center cell is at position [7, 1]
                    zs_ij = zs_windows[:, :, 7, 1]
                elif direction == 'down':
                    # Pad x and zs to handle boundaries for down direction
                    x_padded = np.pad(x, pad_width=((0, 7), (1, 1)), mode='constant', constant_values=False)
                    zs_padded = np.pad(zs, pad_width=((0, 7), (1, 1)), mode='constant', constant_values=zs_max + 1)
                    # Create sliding window views of shape (8, 3)
                    window_shape = (8, 3)
                    x_windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=window_shape)
                    zs_windows = np.lib.stride_tricks.sliding_window_view(zs_padded, window_shape=window_shape)
                    # Center cell is at position [0, 1]
                    zs_ij = zs_windows[:, :, 0, 1]
                else:
                    raise ValueError("direction must be 'left', 'right', 'up', or 'down'")

                # Condition 1: All x values in the window are True
                condition1 = x_windows.all(axis=(-1, -2))

                # Condition 2: All zs values in the window are <= zs[i, j]
                condition2 = (zs_windows <= zs_ij[:, :, None, None]).all(axis=(-1, -2))

                # Condition 3: The difference between zs[i, j] and zs in the window is <= 0.127
                difference = zs_ij[:, :, None, None] - zs_windows
                condition3 = (difference <= 0.127).all(axis=(-1, -2))

                # Combine all conditions
                y = condition1 & condition2 & condition3
                return y

            self.top[:, :] = 0
            self.bottom[:, :] = 0
            self.left[:, :] = 0
            self.right[:, :] = 0
            z_threshold = 0.127
            self.top[1:, :] = (good_paths[:-1, :] & (self.zs[:-1, :] >= self.zs[1:, :]) & (self.zs[:-1, :] <= self.zs[1:, :] + z_threshold))
            self.bottom[:-1, :] = (good_paths[1:, :] & (self.zs[1:, :] >= self.zs[:-1, :]) & (self.zs[1:, :] <= self.zs[:-1, :] + z_threshold))
            self.left[:, 1:] = (good_paths[:, :-1] & (self.zs[:, :-1] >= self.zs[:, 1:]) & (self.zs[:, :-1] <= self.zs[:, 1:] + z_threshold))
            self.right[:, :-1] = (good_paths[:, 1:] & (self.zs[:, 1:] >= self.zs[:, :-1]) & (self.zs[:, 1:] <= self.zs[:, :-1] + z_threshold))

            valid_actions |= self.top
            valid_actions |= self.bottom
            valid_actions |= self.left
            valid_actions |= self.right

            sqaures_empty_left = compute_y(empty_squares, self.zs, 'left')
            squares_empty_right = compute_y(empty_squares, self.zs, 'right')
            squares_empty_up = compute_y(empty_squares, self.zs, 'up')
            squares_empty_down = compute_y(empty_squares, self.zs, 'down')
            squares_empty = sqaures_empty_left | squares_empty_right | squares_empty_up | squares_empty_down
            valid_actions = valid_actions & squares_empty

            # can't turn squares adjacent to rides into rides.
            mask = rides.copy() + occupied_map + queues

            # Expanding mask to adjacent cells
            mask_up = np.roll(mask, 1, axis=0)
            mask_down = np.roll(mask, -1, axis=0)
            mask_left = np.roll(mask, 1, axis=1)
            mask_right = np.roll(mask, -1, axis=1)
            mask_up_left = np.roll(mask_up, 1, axis=1)
            mask_up_right = np.roll(mask_up, -1, axis=1)
            mask_down_left = np.roll(mask_down, 1, axis=1)
            mask_down_right = np.roll(mask_down, -1, axis=1)

            # Combining the masks
            expanded_mask = (mask | mask_up | mask_down | mask_left | mask_right |
                     mask_up_left | mask_up_right | mask_down_left | mask_down_right)

            # Step 2: Apply the mask to the second array
            valid_actions[expanded_mask] = False

            # Define the kernel to count surrounding True cells (including diagonals)
            kernel = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ])
            # Step 1: Convolve the first array with the kernel to count surrounding True cells
            surrounding_true_count = convolve(good_paths.astype(int), kernel, mode='constant', cval=0)

            # Step 2: Create a mask for cells surrounded by 2 or more True cells
            mask = surrounding_true_count >= 2

            # Step 3: Apply this mask to the second array
            valid_actions[mask] = False

            valid_actions[occupied_squares] = False

            # zero out one square from every unowned square.
            # TODO?

            # zero out the three top-most, bottom-most, left-most, and right-most rows and columns
            # note: the size of the matrix is larger than the map size, but we need to use the map size to zero out the correct cells
            # the map size is given by self.size_y, self.size_x
            valid_actions[:3, :] = False
            valid_actions[self.size_y-3:, :] = False
            valid_actions[:, :3] = False
            valid_actions[:, self.size_x-3:] = False
        
        if 'replace' in self.actions:
            valid_actions[self.entrances.astype(bool)] = True
        else:
            valid_actions[self.entrances.astype(bool)] = False
            valid_actions[rides] = False
        #if 'remove' in self.actions:
        #    valid_actions[self.entrances] = True
        
        self.valid_actions = valid_actions.reshape(-1)

        #if np.sum(self.valid_actions) == 0:
        #    print("Warning: no valid actions!")

        #self.plot_valid_actions()
   
    def set_channel_vars(self):
        self.matrix.fill(0)
        self.flat_vector.fill(0)

        cur_num_channels = 0
        for channel_name in self.grid_channels:
            channel_size = self.channels[channel_name]
            if channel_size == 1:
                setattr(self, channel_name, self.matrix[:, :, cur_num_channels])
            else:
                setattr(self, channel_name, self.matrix[:, :, cur_num_channels:cur_num_channels+channel_size])
            cur_num_channels += channel_size
        
        cur_num_channels = 0
        for channel_name in self.flat_channels:
            channel_size = self.channels[channel_name]
            if channel_size == 1:
                setattr(self, channel_name, self.flat_vector[cur_num_channels, :])
            else:
                setattr(self, channel_name, self.flat_vector[cur_num_channels:cur_num_channels+channel_size, :])
            cur_num_channels += channel_size

    def choose_park(self):
        if len(self.parks) == 1:
            return self.parks[0]
        
        if len(set([park.curriculum_num for park in self.parks])) == 1:
            return random.choice(self.parks)
        
        assert False
        metric_stats = ray.get_actor("metrics")
        park_stats = ray.get(metric_stats.get_park_stats.remote())
        
        # if we have been winning the objective for the current parks, then we should move on to the next
        success = True
        num_runs_for_success = 50 #20
        min_win_rate = 0.95 #0.75 #0
        for park in self.cur_parks:
            fname = park.filename.split('/')[-1]
            if fname[0].isdigit():
                fname = fname.split('-')[1]

            if fname not in park_stats:
                print(f"park {fname} not in park stats")
                success = False
                break
            if len(park_stats[fname]['won_objective']) < num_runs_for_success:
                print(f"park {fname} has not been played enough: {len(park_stats[fname]['won_objective'])}")
                success = False
                break
            avg_won_objective = sum(park_stats[fname]['won_objective'][-num_runs_for_success:]) / num_runs_for_success
            if avg_won_objective < min_win_rate:
                print(f"park {fname} has not been winning the objective: {avg_won_objective}: {park_stats[fname]['won_objective'][-num_runs_for_success:]}, {avg_won_objective}")
                success = False
                break
        if success:
            print("success")
            self.cur_curriculum_num += 1
            self.cur_parks = [park for park in self.parks if park.curriculum_num <= self.cur_curriculum_num]

        # sample based on number of months
        sum_months = sum([park.starting_map.num_months for park in self.cur_parks])
        probs = [park.starting_map.num_months / sum_months for park in self.cur_parks]
        return random.choices(self.cur_parks, weights=probs)[0]

    def reset(self, *, seed=None, options=None, targets=None, target_weights=None):
        self.cur_episode += 1
        self.last_action_num = None
        self.sim_ran = False

        self.chosen_park = self.choose_park()

        print(f"\n[RCT_START] Avvio scenario: {self.chosen_park.filename} (Episodio locale: {self.cur_episode})\n")

        # --- MODIFICA dall'originale ---
        max_retries = 3
        loaded = False
        
        for i in range(max_retries):
            print(f"[RCTEnv] Tentativo caricamento parco {self.chosen_park.filename} ({i+1}/{max_retries})")
            response = self.bridge.send_park(self.chosen_park.saved_fname if self.random_invalid_rides else self.chosen_park.filename)
            
            if response is not None and "error" not in str(response) and "TIMEOUT" not in str(response):
                loaded = True
                break
            else:
                print(f"[RCTEnv] Fallito caricamento parco: {response}. Riprovo...")
                
        if not loaded:
            print("[RCTEnv] ERRORE CRITICO: Impossibile caricare il parco dopo vari tentativi.")
            
        # --------------------

        self.bridge.send_action('pause')
        self.bridge.send_action("open_park")
        self.bridge.send_action('set_speed', speed=1) #16)
        
        cash = float(self.bridge.send_action("get_cash"))
        loan = float(self.bridge.send_action("get_loan"))
        max_loan = float(self.bridge.send_action("get_max_loan"))
        self.initial_cash = (max_loan - loan) + cash

        message = self.bridge.send_action("get_park_value")
        self.initial_park_value = float(message) #/ 10
        message = self.bridge.send_action("get_company_value")
        self.initial_company_value = float(message) #/ 10
        message = self.bridge.send_action("get_num_guests")
        self.initial_num_guests = int(message)
        message = self.bridge.send_action("get_park_rating")
        self.initial_park_rating = float(message)

        #self.bridge.send_action('pause')
        
        if targets is None:
            targets = { target.name: random.randint(target.min_val, target.max_val) if type(target.min_val) is int else random.uniform(target.min_val, target.max_val) for target in self.target_ranges }
        else:
            targets = { target.name: targets.get(target.name, random.randint(target.min_val, target.max_val) if type(target.min_val) is int else random.uniform(target.min_val, target.max_val)) for target in self.target_ranges }
        self.targets = targets
        
        self.path_positions = []
        self.z_for_path_positions = {}
        self.rides_seen_from_point = {}
        self.ride_research_history = defaultdict(list)
        
        self.taken_actions = []
        self.prev_valid_actions = {}
        self.rewards = []
        
        if self.action_type in ['wide', 'wide_turtle']:
            # temp, always overwritten
            self.top = np.zeros(self.map_size, dtype=bool)
            self.bottom = np.zeros(self.map_size, dtype=bool)
            self.left = np.zeros(self.map_size, dtype=bool)
            self.right = np.zeros(self.map_size, dtype=bool)
        
            self.top_left = np.zeros(self.map_size, dtype=bool)
            self.top_right = np.zeros(self.map_size, dtype=bool)
            self.bottom_left = np.zeros(self.map_size, dtype=bool)
            self.bottom_right = np.zeros(self.map_size, dtype=bool)

        self.cur_values = { target.name: 0 for target in self.target_ranges }
        self.cur_num_steps = 0

        #self.replaced = np.zeros(self.map_size)
        self.ride_list = []
        self.entrances = np.zeros(self.map_size)
        self.loss = 0
        self.prev_loss = 0
        self.available_ride_types = set()
        self.cur_month = 0

        self.apply_starting_map(self.chosen_park.filename)
        self.guests = { k: 0 for k in range(MAX_NUM_MONTHS) }
        
        self.happinesses = { k: 0 for k in range(MAX_NUM_MONTHS) }
        self.months_with_no_money = set()

        self.update_target_arrays()
        self._fix_action_mask()
        observation = self._get_obs()
        info = self._get_info()
        
        #return self._convert_to_float16(observation), info
        return observation, info
    
    def apply_starting_map(self, park_filename):
        starting_map = self.chosen_park.starting_map

        self.cur_park_filename = park_filename.split('/')[-1].split('.')[0]
        self.all_owned_positions = starting_map.all_owned_positions
        self.total_num_rides = starting_map.total_num_rides
        self.owned_map = starting_map.owned_map

        self.map_dy = starting_map.map_dy
        self.map_dx = starting_map.map_dx
        self.size_y, self.size_x = starting_map.size_y, starting_map.size_x
        self.num_months = starting_map.num_months

        self.total_num_rides_to_place = int(self.num_months * 2.5) # * 2 #* 1.5
        peak_index = int(self.num_months * (3/5))
        smoothness = 2.4 #2.3 # 1.5 #2

        if self.cur_park_filename not in self.scenario_ride_schedules:
            self.scenario_ride_schedules[self.cur_park_filename] = gaussian_list(self.num_months, self.total_num_rides_to_place, peak_index, smoothness)
        self.num_rides_per_month = self.scenario_ride_schedules[self.cur_park_filename]

        self.objective_num_guests = starting_map.objective_num_guests
        self.free_park_entry = starting_map.free_park_entry

        self.padding_y = (self.map_size[0] - self.size_y) // 2
        self.padding_x = (self.map_size[1] - self.size_x) // 2
        
        self.set_channel_vars()
        self.refresh_available_ride_list()
        self.num_guests_for_objective[:] = self.objective_num_guests / 1600

        assert np.all(self.zs <= 1000), np.max(self.zs)
        self.zs[:] = starting_map.zs[:] / 1000
        self.occupied_map[:] = starting_map.occupied_map[:]

        for y, x, queue, z, is_sloped in starting_map.paths: # list(zip(*np.where(starting_map == PATH))):
            #print("placing starting path at", y, x)
            self.apply_action(1, y, x, z=z, apply_in_game=False, queue=queue, is_sloped=is_sloped)
        
        for shop in starting_map.shops:
            # elmt is index into ride_types
            #for r, ride_data in enumerate(self.ride_types):
            #    if ride_data.ride_type == shop.ride_type and (shop.entry_name == '' or shop.entry_name == ride_data.entry_name):
            #        break
            print("apply_starting_map:: shop:", shop, self.ride_types[shop.ride_type])
            self.apply_action(shop.ride_type, shop.entrance_pos[0], shop.entrance_pos[1], entrance_z=shop.entrance_pos[2], ride=True, price=shop.price, apply_in_game=False, z=shop.entrance_pos[2], entry_name=shop.entry_name)
        
        for ride in starting_map.rides:
            #for r, ride_data in enumerate(self.ride_types):
            #    if ride_data.ride_type == ride.ride_type: # and (ride.entry_name == '' or ride.entry_name == ride_data.entry_name):
            #        break
            print("apply_starting_map:: ride:", ride, self.ride_types[ride.ride_type])
            self.apply_action(ride.ride_type, ride.entrance_pos[0], ride.entrance_pos[1], entrance_z=ride.entrance_pos[2], ride=True, price=ride.price, apply_in_game=False, exit_pos=ride.exit_pos, ride_positions=ride.positions, entrance_dir=ride.entrance_dir, exit_dir=ride.exit_dir, z=ride.entrance_pos[2], track_dir=ride.track_dir)
        
        if 'add' not in self.actions:
            self.max_timesteps = len(starting_map.shops) + len(starting_map.rides)

        max_timesteps_for_parks = {}
        if os.path.exists('max_timesteps.pkl'):
            max_timesteps_for_parks = pickle.load(open('max_timesteps.pkl', 'rb'))
        name = park_filename.split('/')[-1].split('.park')[0]
        if name not in max_timesteps_for_parks:
            max_timesteps_for_parks[name] = self.max_timesteps
            pickle.dump(max_timesteps_for_parks, open('max_timesteps.pkl', 'wb'))
            
    def get_size_of_ride_at_pos(self, y, x):
        for ride in self.ride_list:
            if ride.entrance_pos[:-1] == (y, x) or (y, x) in ride.positions:
                cur_ride_data = self.ride_types[ride.ride_type]
                return cur_ride_data.size
        return -1

    def _get_info(self):
        return {'last_action_num': self.last_action_num}

    def can_place_ride_type(self, action_num, shop_name):
        return True

    def place_ride(self, elmt, y, x, price, apply_in_game, exit_pos, ride_positions, z, entrance_dir, exit_dir, queue_line_length=3, track_dir=None, entry_name="default"):
        assert (np.all(self.rides[y, x, :] <= 0) and ('add' in self.actions or not apply_in_game)) or (np.any(self.rides[y, x, :] > 0) and 'replace' in self.actions)
        assert price > 0
        actual_price = round(normalize(price))
        queue_positions, exit_queue_positions = [], []
        if track_dir is None:
            track_dir = -1

        if elmt >= len(self.ride_types):
            print("No-op: elmt", elmt, "is out of bounds out of the ride types list", len(self.ride_types))
            return -1
        if self.action_space_type == 'normal' or type(elmt) == int:
            assert elmt < len(self.ride_types), f"elmt {elmt} is out of bounds out of the ride types list ({len(self.ride_types)})"
            ride_data = self.ride_types[elmt]
            action_num = ride_data.ride_type
            self.last_action_num = action_num
        
        replacing = np.any(self.rides[y, x, :] > 0)
        #print("replacing", replacing)
        #build_queues = False

        if replacing:
            assert apply_in_game
            assert 'replace' in self.actions
            #assert not self.replaced[y, x]
            
            for ride_list_index, ride in enumerate(self.ride_list):
                if ride.entrance_pos[:-1] == (y, x):
                    break
            else:
                assert False, f"no ride found in ride list at {y} {x}"
            cur_ride = self.ride_list[ride_list_index]
            cur_ride_data = self.ride_types[cur_ride.ride_type]
            cur_action_num = cur_ride_data.ride_type

            if type(elmt) != int and self.action_space_type == 'extended':
                for r, ride_index in enumerate(elmt):
                    # find the first of the N rides of the model's output that can be placed
                    ride_data = self.ride_types[ride_index]
                    action_num = ride_data.ride_type
                    assert self.can_place_ride_type(action_num, ride_data.shop_name)
                    if ride_data.size == cur_ride_data.size and (action_num == cur_action_num or self.can_place_ride_type(action_num, ride_data.shop_name)):
                        elmt = ride_index
                        break
                else:
                    #print(f"No-op ({self.cur_park_filename}): No ride of the model's output can be (re)placed at yx", y+self.map_dy, x+self.map_dx, "; just modifying price instead")
                    action_num = cur_action_num
                self.last_action_num = action_num
                if r > 0:
                    print("Chose ride", r, "of action for replacement")
            
            # check if we can place this ride.
            if ride_data.size != cur_ride_data.size or (cur_ride_data.name == 'RIDE_TYPE_FERRIS_WHEEL' and ride_data.name == 'RIDE_TYPE_SWINGING_SHIP'):
                #print("No-op: Cannot replace", cur_ride_data.name, "with", ride_data.name, "; modifying price instead")
                action_num, ride_data = cur_action_num, cur_ride_data
                self.last_action_num = action_num

            # if same ride type, just adjust price without demolishing
            if action_num == cur_action_num:
                #print('modifying price of', cur_ride_data.name, 'to', price)
                #self.bridge.send_action('unpause')
                message = self.bridge.send_action("modify_ride_price",
                                                    x=x+self.map_dx,
                                                    y=y+self.map_dy,
                                                    z=cur_ride.entrance_pos[2],
                                                    price=price)
                #self.bridge.send_action('pause')
                if message != b'0':
                    print("No-op: error when modifying price:", message)
                    return -1
                #assert message == b'0', f'could not modify ride price at y({y})+{self.map_dy},x({x})+{self.map_dx}: {message}; rct proc output:\n{self.bridge.get_output()}'
                
                self.prices[cur_ride.entrance_pos[0], cur_ride.entrance_pos[1]] = price
                for ry, rx in cur_ride.positions:
                    self.prices[ry, rx] = price
                if ride_data.category != 'shop': # not ride_data.shop:
                    self.prices[cur_ride.exit_pos[0], cur_ride.exit_pos[1]] = price
                #self.replaced[y, x] = 1

                return 0

            #print("Replacing", cur_ride_data.name, "with", ride_data.name)
            exit_pos, entrance_z, ride_positions, entrance_dir, exit_dir, track_dir = cur_ride.exit_pos, cur_ride.entrance_pos[2], cur_ride.positions, cur_ride.entrance_dir, cur_ride.exit_dir, cur_ride.track_dir
            #print("Replacing", cur_ride_data.name, "at", y, x, "with", ride_data.name, "at", exit_pos, "with track dir", track_dir)

            if ride_data.size == '4x4':
                ride_pos = cur_ride.ride_pos

            assert len(ride_positions) > 0 or ride_data.category == 'shop', f"(A) ride positions {ride_positions} is empty for ride {ride_data.name}"

            for ry, rx in cur_ride.positions:
                self.rides[ry, rx, :] = 0
                self.prices[ry, rx] = 0
            self.rides[y, x, :] = 0
            self.entrances[y, x] = 0
            self.prices[y, x] = 0
            #self.replaced[y, x] = 1
            
            #self.bridge.send_action('unpause')
            message = self.bridge.send_action("remove_ride",
                x=x+self.map_dx,
                y=y+self.map_dy,
                z=entrance_z
            )
            #self.bridge.send_action('pause')
            assert message == b'0', f'could not demolish ride at y({y})+{self.map_dy},x({x})+{self.map_dx}: {message}; rct proc output:\n{self.bridge.get_output()}'
        elif apply_in_game: # adding (and not starting map)
            assert 'add' in self.actions
            #assert not self.replaced[y, x]
            assert np.all(self.rides[y, x, :] <= 0), f"Tried to place a ride on {y, x}, but there was a ride there."

            def get_ride_positions(y, x, elmt, queue_line_length):
                for ride_data in self.ride_types:
                    if elmt == ride_data.ride_type:
                        ride_size = ride_data.size
                        break
                else:
                    assert False

                queue_path = (y, x)
                adjacent_path = None
                for dy, dx in XY_DELTAS:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < self.map_size[0] and 0 <= nx < self.map_size[1] and self.paths[ny, nx]:
                        adjacent_path = (ny, nx)
                        break
                else:
                    print("No adjacent path found for ride with queue path at y", y+self.map_dy, "x", x+self.map_dx)
                    return [], [], None, None, [], -1, adjacent_path, None
        
                if ride_size == '1x1' or ride_data.category == 'shop':
                    return [], [], (y, x), None, [], -1, adjacent_path, None
                
                delta_x = sign(queue_path[0] - adjacent_path[0])
                delta_y = sign(queue_path[1] - adjacent_path[1])
                queue_positions = [(queue_path[0] + i*delta_x, queue_path[1] + i*delta_y) for i in range(queue_line_length)]
                #queue_positions = [queue_path, (queue_path[0] + delta_x, queue_path[1] + delta_y), (queue_path[0] + 2 * delta_x, queue_path[1] + 2 * delta_y)]
                entrance = (queue_path[0] + queue_line_length * delta_x, queue_path[1] + queue_line_length * delta_y)
                #entrance = (queue_path[0] + 3 * delta_x, queue_path[1] + 3 * delta_y)

                # there is a vector in direction (delta_x, delta_y). we need a perpendicular vector to this.
                perpendicular_vector = (-delta_y, delta_x)

                #print(f"Delta dx {delta_x}, dy {delta_y} and perpendicular to entrance:", perpendicular_vector)
                track_dir = -1

                if ride_size == '2x2':
                    ride_pos = (entrance[0] + 2 * delta_x, entrance[1] + 2 * delta_y)
                    if (delta_x, delta_y) in [(0, 1), (1, 0)]:
                        ride_pos = (entrance[0] + 1 * delta_x, entrance[1] + 1 * delta_y)
                    if (delta_x, delta_y) not in [(0, -1), (1, 0)]:
                        perpendicular_vector = (-perpendicular_vector[0], -perpendicular_vector[1])
                    CORNER_DELTAS = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    ride_positions = [(ride_pos[0]+dy, ride_pos[1]+dx) for dy, dx in CORNER_DELTAS]
                elif ride_size == '3x3':
                    ride_pos = (entrance[0] + 2 * delta_x, entrance[1] + 2 * delta_y) # center
                    ride_positions = [(ride_pos[0]+dy, ride_pos[1]+dx) for dy, dx in XY_DELTAS_WITH_DIAGONAL]
                elif ride_size == '4x4':
                    if delta_y != 0:
                        track_dir = 1
                    else:
                        track_dir = 2
                    if delta_y == -1:
                        ride_pos = (entrance[0] + 4 * delta_x + perpendicular_vector[0], entrance[1] + 4 * delta_y + perpendicular_vector[1])
                    elif delta_y == 1:
                        ride_pos = (entrance[0] + 1 * delta_x, entrance[1] + 1 * delta_y)
                    elif delta_x == -1:
                        ride_pos = (entrance[0] + 1 * delta_x, entrance[1] + 1 * delta_y)
                    else:
                        assert delta_x == 1
                        ride_pos = (entrance[0] + 4 * delta_x + perpendicular_vector[0], entrance[1] + 4 * delta_y + perpendicular_vector[1])

                    if delta_x == -1:
                        entrance_corner = (entrance[0] + delta_x, entrance[1] + delta_y)
                        other_corner = (entrance_corner[0]-3, entrance_corner[1]-3)
                        ride_positions = []
                        for i in range(4):
                            for j in range(4):
                                ride_positions.append((other_corner[0]+i, other_corner[1]+j))
                    elif delta_y == -1:
                        exit_pos = (entrance[0] + perpendicular_vector[0], entrance[1] + perpendicular_vector[1], 0)
                        exit_corner = (exit_pos[0] + delta_x, exit_pos[1] + delta_y)
                        other_corner = (exit_corner[0]-3, exit_corner[1]-3)
                        ride_positions = []
                        for i in range(4):
                            for j in range(4):
                                ride_positions.append((other_corner[0]+i, other_corner[1]+j))
                    elif delta_y == 1:
                        entrance_corner = (entrance[0] + delta_x, entrance[1] + delta_y)
                        other_corner = (entrance_corner[0]-3, entrance_corner[1]+3)
                        ride_positions = []
                        for y in range(other_corner[0], other_corner[0]+4):
                            for x in range(entrance_corner[1], entrance_corner[1]+4):
                                ride_positions.append((y, x))
                    elif delta_x == 1:
                        exit_pos = (entrance[0] + perpendicular_vector[0], entrance[1] + perpendicular_vector[1], 0)
                        exit_corner = (exit_pos[0] + delta_x, exit_pos[1] + delta_y) # (y=53, x=56)
                        other_corner = (exit_corner[0]+3, exit_corner[1]-3) # (y=56, x=53)
                        ride_positions = []
                        #print("x range from", exit_corner[1]+self.map_dx, "to", exit_corner[1]+4+self.map_dx)
                        #print("y range from", other_corner[0]+self.map_dy, "to", other_corner[0]+4+self.map_dy)
                        for y in range(other_corner[0], other_corner[0]+4): # (48, 52)
                            for x in range(exit_corner[1], exit_corner[1]+4): # (53, 57)
                                ride_positions.append((y, x))
                elif ride_size in ['5x1', '7x1']:
                    if delta_y != 0:
                        track_dir = 1
                    else:
                        track_dir = 2
                    if (delta_x, delta_y) == (0, 1):
                        track_dir = 3
                    if (delta_x, delta_y) == (-1, 0):
                        track_dir = 0
                    #track_dir = 1 if delta_x != 0 else 0
                    ride_pos = (entrance[0] + 1 * delta_x, entrance[1] + 1 * delta_y)
                    if ride_size == '5x1' and track_dir == 2 and delta_x == -1:
                        ride_pos = (entrance[0] + perpendicular_vector[0] + 1 * delta_x, entrance[1] + perpendicular_vector[1] + 1 * delta_y)
                    ride_positions = [
                        (ride_pos[0] - 2 * perpendicular_vector[0], ride_pos[1] - 2 * perpendicular_vector[1]),
                        (ride_pos[0] - 1 * perpendicular_vector[0], ride_pos[1] - 1 * perpendicular_vector[1]),
                        (ride_pos[0], ride_pos[1]),
                        (ride_pos[0] + 1 * perpendicular_vector[0], ride_pos[1] + 1 * perpendicular_vector[1]),
                        (ride_pos[0] + 2 * perpendicular_vector[0], ride_pos[1] + 2 * perpendicular_vector[1]),
                    ]
                    #if ride_size == '7x1':
                    #    ride_positions.extend([
                    #        (ride_pos[0] - 3 * perpendicular_vector[0], ride_pos[1] - 3 * perpendicular_vector[1]),
                    #        (ride_pos[0] + 3 * perpendicular_vector[0], ride_pos[1] + 3 * perpendicular_vector[1]),
                    #    ])
                    #print("Ride positions:", ride_positions)

                exit_queue_positions = [(qy+perpendicular_vector[0], qx+perpendicular_vector[1]) for qy, qx in queue_positions]
                last_path_pos = (adjacent_path[0]+perpendicular_vector[0], adjacent_path[1]+perpendicular_vector[1])
                if not self.paths[last_path_pos[0], last_path_pos[1]]:
                    exit_queue_positions.append(last_path_pos)
                exit_pos = (entrance[0] + perpendicular_vector[0], entrance[1] + perpendicular_vector[1], 0)

                #print("upd Delta:", delta_x, delta_y, "and perpendicular to entrance:", perpendicular_vector)
                #print("For the queue path:", queue_path[0]+self.map_dy, queue_path[1]+self.map_dx, "and entrance:", entrance[0]+self.map_dy, entrance[1]+self.map_dx, "the ride positions are: (x, y)", [(r[1]+self.map_dx, r[0]+self.map_dy) for r in ride_positions], "and exit queue positions:", exit_queue_positions, "and exit pos:", exit_pos[0]+self.map_dy, exit_pos[1]+self.map_dx)
                #print("Track dir:", track_dir)

                return queue_positions, exit_queue_positions, entrance, exit_pos, ride_positions, track_dir, adjacent_path, ride_pos

            def positions_empty(queue_positions, exit_queue_positions, entrance, exit_pos, ride_positions, z, ride_data):
                positions = [('entrance', entrance), *[('queue', q) for q in queue_positions], *[('exit_queue', q) for q in exit_queue_positions], ('exit', exit_pos), *[('ride_pos', q) for q in ride_positions]]
                positions = [(s, p[0], p[1]) for s, p in positions if p is not None]
                if len(set(positions)) != len(positions):
                    print('duplicate positions:', positions)
                    #return False
                #print('positions:', positions)
                #print('actual positions:', [(y+self.map_dy, x+self.map_dx) for y, x in positions])
                #print("bounds:", self.size_y, self.size_x)
                for s, py, px in positions:
                    if py < 0 or py > self.map_size[0]:
                        print(ride_data.name, s, 'out of bounds y', py, self.map_size[0])
                        return False
                    if px < 0 or px > self.map_size[1]:
                        print(ride_data.name, s, 'out of bounds x', px, self.map_size[1])
                        return False
                    if z - self.zs[py, px] > 0.127:
                        print(ride_data.name, s, 'too high for supports y', py+self.map_dy, px+self.map_dx, self.zs[py, px], z)
                        return False
                return True

            if type(elmt) != int and self.action_space_type == 'extended':
                for r, ride_index in enumerate(elmt):
                    ride_data = self.ride_types[ride_index]
                    action_num = ride_data.ride_type
                    queue_positions, exit_queue_positions, entrance, exit_pos, ride_positions, track_dir, adjacent_path = get_ride_positions(y, x, action_num, queue_line_length)
                    if entrance is None:
                        continue

                    temp_z = self.zs[adjacent_path[0], adjacent_path[1]]
                    assert self.can_place_ride_type(action_num, ride_data.shop_name)
                    #print("Trying to place ride", ride_data.name, "at", y, x, "with price", price)
                    if positions_empty(queue_positions, exit_queue_positions, entrance, exit_pos, ride_positions, temp_z, ride_data):
                        print("Chose", ride_data.name, "for placement (r:", r, ")")
                        elmt = ride_index
                        break
                else:
                    #print("can't place ride type (A)")
                    print(f"No-op ({self.cur_park_filename}): No ride of the model's output can be placed at yx", y+self.map_dy, x+self.map_dx)
                    return -1
                if r > 0:
                    print("Chose ride", r, "of action for placement")
            else:
                queue_positions, exit_queue_positions, entrance, exit_pos, ride_positions, track_dir, adjacent_path, ride_pos = get_ride_positions(y, x, action_num, queue_line_length)
                if entrance is None:
                    print("No-op: entrance is None?")
                    #self.plot_valid_actions()
                    #input()
                    return -1
                if ride_data.size not in ['4x4', '5x1']:
                    ride_pos = None
                temp_z = self.zs[adjacent_path[0], adjacent_path[1]]
                assert self.can_place_ride_type(action_num, ride_data.shop_name)
                #print("Trying to place ride", ride_data.name, "at", y, x, "with price", price)
                if not positions_empty(queue_positions, exit_queue_positions, entrance, exit_pos, ride_positions, temp_z, ride_data):
                    print("No-op: can't place at this position")
                    return -1

            self.last_action_num = action_num

            entrance_dir = -1
            exit_dir = -1
            y, x = entrance
            z = self.z_for_path_positions[adjacent_path]
            #self.replaced[y, x] = 1
            self.total_num_rides += 1

            # find the z coordinate for the associated path
            #print("Getting z for adjacent path", adjacent_path)
            entrance_z = z
            if exit_pos is not None:
                exit_pos = (exit_pos[0], exit_pos[1], z)
            if queue_positions is not None:
                queue_positions = [(qy, qx, z) for qy, qx in queue_positions]
                exit_queue_positions = [(qy, qx, z) for qy, qx in exit_queue_positions]

            assert self.can_place_ride_type(action_num, ride_data.shop_name)
            assert ride_data.category == 'shop' or len(ride_positions) > 0, "(B) ride_positions is empty for ride type " + ride_data.name
            #print("Adding new ride", ride_data.name, "at", y, x, "with price", price)
        else:
            #print("not replacing, and not applying in game, so adding initial?")
            entrance_z = z
            assert ride_data.category == 'shop' or len(ride_positions) > 0, "(C) ride_positions is empty for ride type " + ride_data.name

        if ride_data.category == 'shop':
            if apply_in_game:
                adjacent_path = None
                for dy, dx in XY_DELTAS:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < self.map_size[0] and 0 <= nx < self.map_size[1] and self.paths[ny, nx]:
                        adjacent_path = (ny, nx)
                        break
                if x > adjacent_path[1]:
                    track_dir = 0
                elif x < adjacent_path[1]:
                    track_dir = 2 # not 1
                elif y > adjacent_path[0]:
                    track_dir = 3
                else:
                    track_dir = 1 # not 3
                # calculate shop dir (for bathrooms)

                #self.bridge.send_action('unpause')
                result = self.bridge.send_action("place_ride",
                    x=x+self.map_dx,
                    y=y+self.map_dy,
                    z=entrance_z,
                    ride_type=action_num,
                    price=actual_price,
                    name=ride_data.name,
                    track_dir=track_dir,
                    ride_entry_name=ride_data.shop_name,
                ) #, ride_subtype=self.ride_subtypes[ride_name])
                #self.bridge.send_action('pause')
                if result == b'2':
                    # too high for supports
                    print("No-op: Shop placement: too high for supports; actual z:", self.zs[y, x], "and placed:", z/1000, "diff: ", self.zs[y, x] - z/1000)
                    return -1
                elif result == b'4':
                    # not enough money
                    print("No-op: Shop placement: Not enough money")
                    return -2
                elif result != b'0':
                    print(f'No-op: Shop placement: error {result} when placing on y+{self.map_dy}:{y+self.map_dy}, x+{self.map_dx}:{x+self.map_dx}')
                    return -1
                
                if not self.free_park_entry:
                    # park has an entrance fee instead, must set to sum of all prices.
                    sum_prices = 0
                    for price in self.prices[self.prices > 0]:
                        sum_prices += round(normalize(price))
                    #self.bridge.send_action('unpause')
                    self.bridge.send_action("set_park_entry_fee", price=sum_prices)
                    #self.bridge.send_action('pause')
                
            all_positions = [(y, x, z)]

            self.rides[y, x, elmt] = min((len(self.ride_list) + 1) / self.total_num_rides_to_place, 1)
            self.entrances[y, x] = min((len(self.ride_list) + 1) / self.total_num_rides_to_place, 1)
            self.prices[y, x] = price
            self.ride_quantities[elmt] += 1 / MAX_COPIES_PER_RIDE
            
            if replacing:
                self.ride_list[ride_list_index].ride_type = elmt
                self.ride_list[ride_list_index].price = price
            else:
                ride = Ride((y, x, z), None, elmt, price, positions=[], entrance_dir=entrance_dir, exit_dir=exit_dir, track_dir=track_dir, all_positions=all_positions)
                self.ride_list.append(ride)
        else: # ride
            entrance = (y, x, entrance_z)
            adjacent_path = None
            for dy, dx in XY_DELTAS:
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.map_size[0] and 0 <= nx < self.map_size[1] and self.paths[ny, nx]:
                    adjacent_path = (ny, nx)
                    break
            
            if ride_positions is None:
                delta_x = entrance[0] - adjacent_path[0]
                delta_y = entrance[1] - adjacent_path[1]
                ride_pos = (entrance[0] + 2 * delta_x, entrance[1] + 2 * delta_y)
                ride_positions = [(ride_pos[0]+dy, ride_pos[1]+dx) for dy, dx in XY_DELTAS_WITH_DIAGONAL]
                print("Ride pos is none")
            else:
                assert len(ride_positions) > 0
                if ride_data.size != '4x4':
                    ride_pos = centermost_coord(ride_positions)
            
            if exit_pos is None:
                direction_vector = (entrance[0] - adjacent_path[0], entrance[1] - adjacent_path[1])
                perpendicular_vector = (-direction_vector[1], direction_vector[0])
                exit_pos = (y+perpendicular_vector[0], x+perpendicular_vector[1], entrance_z)
            
            if apply_in_game:
                assert entrance_dir is not None, "need to implement for add..."
                #self.bridge.send_action('unpause')
                result = self.bridge.send_action("place_ride",
                    x=ride_pos[1]+self.map_dx,
                    y=ride_pos[0]+self.map_dy,
                    z=entrance_z,
                    ride_type=action_num,
                    entrance_x=entrance[1]+self.map_dx,
                    entrance_y=entrance[0]+self.map_dy,
                    entrance_z=entrance_z, #entrance[2],
                    exit_x=exit_pos[1]+self.map_dx,
                    exit_y=exit_pos[0]+self.map_dy,
                    exit_z=exit_pos[2],
                    price=actual_price,
                    entrance_dir=entrance_dir,
                    exit_dir=exit_dir,
                    name=ride_data.name,
                    track_dir=track_dir,
                    ride_entry_name="default",
                )
                #self.bridge.send_action('pause')
                if result == b'2':
                    if adjacent_path is not None:
                        z = self.zs[adjacent_path[0], adjacent_path[1]]
                        print(f"No-op: Ride placement ({self.cur_park_filename}): too high for supports; actual z:", self.zs[ride_pos[0], ride_pos[1]], "and placed:", z/1000, "diff: ", self.zs[y, x] - z/1000)
                    elif z is not None:
                        print(f"No-op: Ride placement ({self.cur_park_filename}): too high for supports: z coords for ride positions:", [self.zs[ry, rx] for ry, rx in ride_positions], "diff:", [self.zs[ry, rx] - z/1000 for ry, rx in ride_positions])
                    else:
                        print(f"No-op: Ride placement ({self.cur_park_filename}): too high for supports at position y", y+self.map_dy, x+self.map_dx, "with z:", self.zs[ride_pos[0], ride_pos[1]])
                    # too high for supports
                    return -1
                elif result == b'4':
                    # not enough money
                    print(f"No-op: Ride placement ({self.cur_park_filename}): Not enough money")
                    return -2
                elif result != b'0':
                    print(f'No-op: Ride placement ({self.cur_park_filename}): error {result} when placing on y+{self.map_dy}:{y+self.map_dy}, x+{self.map_dx}:{x+self.map_dx}; rct proc output:\n{self.bridge.get_output()}')
                    return -1

                if not self.free_park_entry:
                    # park has an entrance fee instead, must set to sum of all prices.
                    sum_prices = 0
                    for price in self.prices[self.prices > 0]:
                        sum_prices += round(normalize(price))
                    #self.bridge.send_action('unpause')
                    self.bridge.send_action("set_park_entry_fee", price=sum_prices)
                    #self.bridge.send_action('pause')
            
            self.rides[entrance[0], entrance[1], elmt] = min((len(self.ride_list) + 1) / self.total_num_rides_to_place, 1)
            #print("Setting entrance at", entrance[0], entrance[1], "for ride", elmt, "with positions", ride_positions)
            for ry, rx in ride_positions:
                self.rides[ry, rx, elmt] = min((len(self.ride_list) + 1) / self.total_num_rides_to_place, 1)
            self.rides[exit_pos[0], exit_pos[1], elmt] = min((len(self.ride_list) + 1) / self.total_num_rides_to_place, 1)
            self.ride_quantities[elmt] += 1 / MAX_COPIES_PER_RIDE

            self.prices[entrance[0], entrance[1]] = price
            self.entrances[entrance[0], entrance[1]] = min((len(self.ride_list) + 1) / self.total_num_rides_to_place, 1)
            for ry, rx in ride_positions:
                self.prices[ry, rx] = price
            self.prices[exit_pos[0], exit_pos[1]] = price
            
            if replacing:
                self.ride_list[ride_list_index].ride_type = elmt
                self.ride_list[ride_list_index].price = price
            else:
                all_positions = [entrance, exit_pos, *queue_positions, *exit_queue_positions, *ride_positions]
                ride = Ride(entrance, exit_pos, elmt, price, positions=ride_positions, entrance_dir=entrance_dir, exit_dir=exit_dir, track_dir=track_dir, ride_pos=ride_pos, all_positions=all_positions)
                self.ride_list.append(ride)

        if len(queue_positions) > 0 or len(exit_queue_positions) > 0:
            for qy, qx, qz in queue_positions:
                self.apply_action(PATH, qy, qx, z=qz, queue=True)

            for qy, qx, qz in exit_queue_positions:
                self.apply_action(PATH, qy, qx, z=qz, queue=False)

        return 0

    def apply_action(self, elmt, y, x, queue=False, ride=False, price=0, apply_in_game=True, exit_pos=None, ride_positions=None, z=None, entrance_dir=None, exit_dir=None, entrance_z=None, is_sloped=False, queue_line_length=3, track_dir=None, entry_name="default"):
        assert y >= 0 and x >= 0, f"y, x: {y, x}"
        assert y < self.map_size[0] and x < self.map_size[1], f"y, x: {y, x}"
        if ride:
            #print('placing ride', elmt, y, x, price)
            result = self.place_ride(elmt, y, x, price, apply_in_game, exit_pos, ride_positions, z, entrance_dir, exit_dir, queue_line_length, track_dir, entry_name)
            #print(result)
            #input("done placing")
            return result
        elif elmt == PATH:
            if np.any(self.rides[y, x, :] > 0):
                print(f"Tried to place a path on {x+self.map_dx, y+self.map_dy}, but there was a ride there.")
                #input()
                #assert False
                return
            if self.paths[y, x] == 1:
                return
            
            if apply_in_game:
                #self.bridge.send_action('unpause')
                result = self.bridge.send_action("place_path",
                    x=x+self.map_dx,
                    y=y+self.map_dy,
                    z=z if z is not None else 0,
                    queue=int(queue)
                )
                #self.bridge.send_action('pause')
                if result == b'2':
                    # too high for supports
                    if z is not None:
                        print(f"Path placement ({self.cur_park_filename}): too high for supports; actual z:", self.zs[y, x], "and placed:", z/1000, "diff: ", round(self.zs[y, x] - z/1000, 2), "at position", y+self.map_dy, x+self.map_dx)
                    else:
                        print(f"Path placement ({self.cur_park_filename}): too high for supports; z:", z, "at position", y+self.map_dy, x+self.map_dx)
                    return -1
                elif result != b'0':
                    print(f'Path placement: error {result} when placing path on y+{self.map_dy}:{y+self.map_dy}, x+{self.map_dx}:{x+self.map_dx}')
                    return -1

            if queue:
                self.queues[y, x] = 1
            else:
                self.paths[y, x] = 1
            if is_sloped:
                self.sloped_paths[y, x] = 1
            self.path_positions.append((y, x))
            self.z_for_path_positions[(y, x)] = z
        return 0

    def apply_action_with_number(self, action):
        self.last_action_num = None
        # ... (codice existing per is_baseline, lascia invariato) ...

        assert self.action_type == 'wide'

        if self.action_space_type == 'normal':
            yx, ride_type = action['position_and_type']
        elif self.action_space_type == 'extended':
            yx = action['position_and_type'][0]
            ride_type = action['position_and_type'][1:]

        y, x = yx // self.map_size[0], yx % self.map_size[0]

        # Se l'azione è "Skip" (non fare nulla)
        if ride_type == len(self.ride_types):
            # print("[RCT_DECISION] L'agente ha deciso di aspettare (SKIP)") # Decommenta se vuoi vedere anche i tempi morti
            return True

        # ### RECUPERO DATI LEGGIBILI ###
        # Recuperiamo l'oggetto RideDef che contiene i dettagli della giostra
        ride_def = self.ride_types[ride_type]

        # Puliamo il nome: da "RIDE_TYPE_WOODEN_ROLLER_COASTER" a "WOODEN_ROLLER_COASTER"
        # Se ha un nome specifico da negozio (es. "Burger Bar"), usiamo quello.
        readable_name = ride_def.shop_name if ride_def.shop_name else ride_def.name.replace("RIDE_TYPE_", "")

        # Recuperiamo la categoria (Thrill, Gentle, Shop, RollerCoaster)
        category = ride_def.category

        # Recuperiamo il prezzo deciso dall'agente
        price = action['price'][0]

        # ### RECUPERO IL CONTESTO (IL "PERCHÉ") ###
        # Vediamo quanti soldi e ospiti aveva nel momento della decisione.
        # Nota: raw_values potrebbe non essere aggiornatissimo al millisecondo, ma dà l'idea.
        current_cash = self.raw_values.get('cash', 0) if hasattr(self, 'raw_values') else 0
        current_guests = self.raw_values.get('num_guests', 0) if hasattr(self, 'raw_values') else 0

        print(f"\n[RCT_OP] Mese: {self.cur_month} | Ospiti: {current_guests} | Cassa: {current_cash:.0f}€")
        print(f"         >>> SCELTA: {readable_name} ({category})")
        print(f"         >>> POSIZIONE: Y={y}, X={x} | PREZZO: {price:.2f}€")

        queue_line_length = action['queue_line_length'] + 1
        assert 1 <= queue_line_length <= 5, f"queue_line_length: {queue_line_length}"

        self.cur_y = y
        self.cur_x = x

        # Eseguiamo l'azione
        result = self.apply_action(ride_type, y, x, ride=True, price=price, queue_line_length=queue_line_length)

        # ### LOG DEL RISULTATO ###
        if result == 0:
            print(f"         [V] COSTRUZIONE RIUSCITA!")
        elif result == -2:
            print(f"         [X] FALLITO: Non abbastanza soldi (Costo stimato > {current_cash})")
        elif result == -1:
            print(f"         [X] FALLITO: Posizione non valida (Collisione o terreno errato)")

        return result

    def get_cur_values(self):
        cur_values = BasePathEnv.get_cur_values(self)
        
        #self.bridge.send_action('unpause')
        raw_values = {}
        for target in self.target_ranges:
            if target.name in ['ride_diversity', 'won_objective', 'num_added_rides', 'cash'] or 'thoughts' in target.name or '_month' in target.name:
                continue
            message = self.bridge.send_action(f'get_{target.name}')
            if message == b'nan' or float(message) == 0:
                message = 1
            message = float(message)
            
            raw_values[target.name] = message

        cash = float(self.bridge.send_action("get_cash"))
        loan = float(self.bridge.send_action("get_loan"))
        max_loan = float(self.bridge.send_action("get_max_loan"))
        raw_values['cash'] = (max_loan - loan) + cash

        visit_count_json_str = self.bridge.send_action('get_visit_counts')
        num_elapsed_months = int(self.bridge.send_action('get_num_months'))
        # Robust capping for any month overflow (25, 26, etc.)
        num_elapsed_months = min(num_elapsed_months, MAX_NUM_MONTHS - 1)
        assert 0 <= num_elapsed_months < MAX_NUM_MONTHS, f"num_elapsed_months: {num_elapsed_months}"

        num_guests = float(self.bridge.send_action('get_num_guests'))
        self.guests[num_elapsed_months] = num_guests
        for k, guests in self.guests.items():
            raw_values[f'num_guests_month{k}'] = guests

        happiness = float(self.bridge.send_action('get_avg_happiness'))
        self.happinesses[num_elapsed_months] = happiness
        for k, happiness in self.happinesses.items():
            raw_values[f'avg_happiness_month{k}'] = happiness

        self.raw_values = raw_values
                
        #self.bridge.send_action('pause')

        ride_stats = []
        #self.bridge.send_action('unpause')
        for ride in self.ride_list:
            ride_stat = self.bridge.send_action('get_ride_stats', x=ride.entrance_pos[1]+self.map_dx, y=ride.entrance_pos[0]+self.map_dy)
            ride_stats.append((ride, ride_stat))
        #self.bridge.send_action('pause')
        
        awards = self.bridge.send_action('get_awards')
        awards = json.loads(awards)
        self.awards[:] = 0
        if awards is not None:
            for award in awards:
                award_type, award_time = int(award['type']), int(award['time'])
                assert 0 <= award_time <= 5, f"award_time: {award_time}"
                self.awards[award_type] = award_time / 5

        #for i in range(self.num_months):
        #    self.ride_schedule[i] = self.num_rides_per_month[i] / 10

        # FIX: Usa min() per non superare la dimensione massima dell'array (24)
        for i in range(self.num_months):
             self.ride_schedule[i] = self.num_rides_per_month[i] / 10
             
        # calculate ride diversity

        # self.rides is an (x, y, m) array, where m is the number of ride types.
        # we want to sum so that it is of shape (m,)...
        ride_counts = np.sum(self.rides, axis=(0, 1))
        ride_diversity = np.sum(ride_counts > 0)
        raw_values['ride_diversity'] = ride_diversity
        raw_values['num_added_rides'] = len(self.ride_list)

        normalized_values = self.normalize_targets(raw_values)
        cur_values.update(normalized_values)

        # visit counts
        data = np.array(json.loads(visit_count_json_str))
        visit_counts = data
        if np.sum(data) > 0:
            visit_counts = data / np.max(data)

        visit_counts = visit_counts[self.map_dx:, self.map_dy:]

        # adjust visit counts size to be no greater than map size
        visit_counts = visit_counts[:self.map_size[0], :self.map_size[1]]

        # set self.visit_counts to all zeroes
        self.visit_counts[:] = 0

        # place visit counts into the top-left corner of self.visit_counts
        self.visit_counts[:visit_counts.shape[0], :visit_counts.shape[1]] = visit_counts
        #assert np.sum(self.visit_counts) > 0

        assert abs(self.cur_month - num_elapsed_months) <= 1, f"cur_month: {self.cur_month}, num_elapsed_months: {num_elapsed_months}"
        self.cur_month = num_elapsed_months
        self.pct_to_objective_end[:] = self.cur_month / self.num_months
        self.norm_month[:] = self.cur_month / MAX_NUM_MONTHS
        if self.cur_month / self.num_months > 1:
            print("cur_month:", self.cur_month, "num_months:", self.num_months, "timestep:", self.pct_to_objective_end[0])
        #assert 0 <= self.cur_month / self.num_months <= 1, f"cur_month: {self.cur_month}, num_months: {self.num_months}, timestep: {self.pct_to_objective_end[0]}"

        def normalize(stat_name, stat):
            norm_funcs = {
                'excitement': lambda x: min(1, x / 1000) if x > 0 else 0,
                'intensity': lambda x: min(1, x / 1000),
                'nausea': lambda x: min(1, x / 1000),
                'popularity': lambda x: min(1, x / 100) if x < 255 else 0,
                'profit': lambda x: min(1, x / 100000),
            }
            return norm_funcs[stat_name](stat)

        # update stat channels
        for ride, ride_stat in ride_stats:
            if ride_stat is None:
                continue
            try:
                ride_stat = json.loads(ride_stat)
            except:
                print("error with ride", ride.entrance_pos, "ride_stat", ride_stat)
                continue
            for stat_name, stat_value in ride_stat.items():
                matrix = getattr(self, stat_name)
                norm_val = normalize(stat_name, float(stat_value))
                #if not (-1 <= norm_val <= 1):
                #    print("stat_name", stat_name, "stat_value", stat_value, "norm_val", norm_val)
                for pos in ride.all_positions:
                    ry, rx = pos[0], pos[1]
                    matrix[ry, rx] = max(-1, min(1, norm_val))
                #matrix[ride.entrance_pos[0], ride.entrance_pos[1]] = max(-1, min(1, norm_val))
        
        num_guests = raw_values['num_guests']
        guest_thoughts = self.bridge.send_action('get_guest_thoughts')
        guest_thoughts = json.loads(guest_thoughts)
        #print("guest_thoughts", guest_thoughts)
        for thought, value in guest_thoughts.items():
            cur_values[f'num_{thought}_thoughts'] = 1 - float(value) / num_guests

        return cur_values
    
    def get_starting_map(self, map_type, park_filename):
        starting_map = np.zeros(self.map_size)
        occupied_map = np.zeros(self.map_size)
        
        if park_filename is not None:
            # get paths
            #self.bridge.send_action('unpause')
            message = self.bridge.send_action("get_paths")
            #self.bridge.send_action('pause')
            message = json.loads(message)
            #pprint(message)
            xs, ys = [], []
            #for coord in message['paths'] + message['ride_entrances'] + message['shops']:
            for coord in message['positions']:
                xs.append(int(coord['x']))
                ys.append(int(coord['y']))
            min_xs, min_ys, max_xs, max_ys = min(xs), min(ys), max(xs), max(ys)
            size_y, size_x = max_ys-min_ys+1, max_xs-min_xs+1
            dx, dy = min_xs, min_ys
            print("park size from owned positions:", size_y, size_x)

            xs, ys = [], []
            for coord in message['paths']:
                xs.append(int(coord['x']))
                ys.append(int(coord['y']))
            min_xs, min_ys, max_xs, max_ys = min(xs), min(ys), max(xs), max(ys)
            min_xs, min_ys, max_xs, max_ys = min_xs-10, min_ys-10, max_xs+10, max_ys+10
            size_y_, size_x_ = max_ys-min_ys+1, max_xs-min_xs+1
            print("park size from paths with padding:", size_y_, size_x_)
            if size_y_*size_x_ < size_y*size_x:
                size_y, size_x = size_y_, size_x_
                dx, dy = min_xs, min_ys

            #input()
            
            assert size_y <= self.map_size[0] and size_x <= self.map_size[1], f"Map size for {park_filename} ({size_y}, {size_x}) is too large for the current model size ({self.map_size[0]}, {self.map_size[1]})"
            
            #paths = [(int(path['x'])-dx, int(path['y'])-dy) for path in message['paths']]
            #print(message['paths'])
            paths = []
            for path in message['paths']:
                x, y = int(path['x'])-dx, int(path['y'])-dy
                paths.append((y, x, int(path['queue']), int(path['z']), bool(path['is_sloped'])))
            #paths = [(int(path['y'])-dy, int(path['x'])-dx, int(path['queue']), int(path['z'], bool(path['is_sloped']))) for path in message['paths']]
            #print(paths)

            def normalize(value, min_old=1, max_old=200, min_new=0.10, max_new=1):
                normalized_value = ((value - min_old) / (max_old - min_old)) * (max_new - min_new) + min_new
                return normalized_value
            
            def get_elmt_num_for_ride_type(ride_type, ride_entry_name=''):
                for elmt, ride_data in enumerate(self.ride_types):
                    if ride_type == ride_data.ride_type and (ride_data.name not in ['RIDE_TYPE_FOOD_STALL', 'RIDE_TYPE_DRINK_STALL', 'RIDE_TYPE_SHOP'] or ride_entry_name == '' or ride_entry_name == ride_data.shop_name):
                        return elmt
                return -1
            
            rides = defaultdict(list)
            total_num_rides = 0
            for entrance in message.get('ride_entrances', []):
                if entrance['entrance_type'] == 'entrance':
                    total_num_rides += 1
                ride_type = get_elmt_num_for_ride_type(int(entrance['ride_type']))
                #print("finding for type", entrance['ride_type'], get_elmt_num_for_ride_type(int(entrance['ride_type'])))
                if ride_type == -1:
                    continue
                
                x, y, z = int(entrance['x'])-dx, int(entrance['y'])-dy, int(entrance['z'])
                price = normalize(int(entrance['price']))
                #print('get_starting_map::entrance', entrance['price'], price)
                
                ride_index = int(entrance['ride_index'])
                entrance_type = entrance['entrance_type']
                direction = int(entrance['direction'])
                rides[ride_index].append((entrance_type, ride_type, price, direction, y, x, z))

            rides2 = {}
            for ride_index, (pos1, pos2) in rides.items():
                assert pos1[0] in ['entrance', 'exit']
                if pos1[0] == 'entrance':
                    entrance_pos, exit_pos = pos1[-3:], pos2[-3:]
                    entrance_dir, exit_dir = pos1[3], pos2[3]
                else:
                    entrance_pos, exit_pos = pos2[-3:], pos1[-3:]
                    entrance_dir, exit_dir = pos2[3], pos1[3]
                #print(f"Ride (index {ride_index}) with entrance {entrance_pos}, exit {exit_pos}, type {pos1[1]}")
                rides2[ride_index] = Ride(entrance_pos, exit_pos, pos1[1], pos1[2], entrance_dir=entrance_dir, exit_dir=exit_dir, track_dir=-1)
                #print("Ride found at", entrance_pos, entrance_pos[0]+dx, entrance_pos[1]+dy, "with type", self.ride_types[pos1[1]])
            
            shops = []
            for shop in message.get('shops', []):
                total_num_rides += 1

                ride_type = get_elmt_num_for_ride_type(int(shop['ride_type']), shop['entry_name'])
                if ride_type == -1:
                    continue
                
                x, y, z = int(shop['x'])-dx, int(shop['y'])-dy, int(shop['z'])
                price = normalize(int(shop['price']))
                entry_name = shop['entry_name']
                #print('get_starting_map::shop', shop['price'], price)
                shops.append(Ride(entrance_pos=(y, x, z), exit_pos=None, ride_type=ride_type, price=price, entrance_dir=0, exit_dir=0, track_dir=-1, entry_name=entry_name))
                #print("Shop found at", (x, y), x+dx, y+dy, "with type", self.ride_types[ride_type])
            
            #all_ride_positions = set()
            for ride_pos in message.get('ride_positions', []):
                x, y, z = int(ride_pos['x'])-dx, int(ride_pos['y'])-dy, int(ride_pos['z'])
                ride_index = int(ride_pos['ride_index'])
                if y < 0 or x < 0:
                    continue
                #all_ride_positions.add((y, x))
                if 'ride_type' not in ride_pos:
                    print("no ride type in message?", ride_pos)
                ride_type = get_elmt_num_for_ride_type(int(ride_pos['ride_type']))
                if ride_type == -1:
                    occupied_map[y, x] = 1

                if ride_index not in rides2:
                    continue
                rides2[ride_index].positions.append((y, x))
                
                track_dir = int(ride_pos['track_dir'])
                rides2[ride_index].track_dir = track_dir
                #print("Setting track dir for ride", ride_index, "to", track_dir)
                #input()
                #assert y > 0 and x > 0, (y, x)
            
            zs = np.zeros(self.map_size)
            all_owned_positions = set()
            owned_map = np.zeros(self.map_size)
            for pos in message['positions']:
                x, y = int(pos['x'])-dx, int(pos['y'])-dy
                #if y < 0 or x < 0:
                #    continue
                all_owned_positions.add((y, x))
                if 0 <= y < zs.shape[0] and 0 <= x < zs.shape[1]:
                    owned_map[y, x] = 1
                    zs[y, x] = int(pos['z'])

            for path in message['paths']:
                x, y, z = int(path['x'])-dx, int(path['y'])-dy, int(path['z'])
                zs[y, x] = z

            for ride_index, ride in rides2.items():
                assert len(ride.positions) > 0, f"No ride positions for ride index {ride_index} and type {ride.ride_type}"
            
            num_months = int(message['objective_num_months'])
            objective_num_guests = int(message['objective_num_guests'])
            free_park_entry = bool(message['free_entry'])

            #for ride in rides2:
            #    # print ride info
            #    print(vars(rides2[ride]))
            #for shop in shops:
            #    print(vars(shop))
            #rides = rides + shops
            #print(rides)
        elif map_type == 0:
            starting_map[12, :] = PATH
        elif map_type == 1:
            starting_map[1, :] = PATH
            starting_map[22, :] = PATH
            starting_map[:, 1] = PATH
            starting_map[:, 22] = PATH
            starting_map[12, :] = PATH
            starting_map[:, 12] = PATH
        elif map_type == 2:
            i, j = 2, 3
            k, l = 2, 21
            while i < 22:
                starting_map[i:i+2, i+1] = PATH
                i += 1
                starting_map[k:k+2, l] = PATH
                k += 1
                l -= 1
        elif map_type == 3:
            def discretized_circle_outline_coordinates(radius, array_size):
                center = (array_size - 1) / 2
                y, x = np.ogrid[:array_size, :array_size]
                distance_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    
                # Create a mask for the circle outline
                outer_circle = distance_from_center <= radius
                inner_circle = distance_from_center < radius - 1
                outline = outer_circle & ~inner_circle

                # Initialize the final mask for the fully connected outline
                final_mask = np.zeros_like(outline, dtype=bool)
    
                # Check each point and ensure connectivity
                for i in range(1, array_size - 1):
                    for j in range(1, array_size - 1):
                        if outline[i, j]:
                            final_mask[i, j] = True
                            # Check for diagonal-only connections and add points if necessary
                            if outline[i-1, j-1] and not (outline[i-1, j] or outline[i, j-1]):
                                final_mask[i-1, j] = True
                                final_mask[i, j-1] = True
                            if outline[i-1, j+1] and not (outline[i-1, j] or outline[i, j+1]):
                                final_mask[i-1, j] = True
                                final_mask[i, j+1] = True
                            if outline[i+1, j-1] and not (outline[i+1, j] or outline[i, j-1]):
                                final_mask[i+1, j] = True
                                final_mask[i, j-1] = True
                            if outline[i+1, j+1] and not (outline[i+1, j] or outline[i, j+1]):
                                final_mask[i+1, j] = True
                                final_mask[i, j+1] = True
    
                coordinates = np.argwhere(final_mask)
                return coordinates
            # Example usage
            radius = 7
            array_size = 20
            coordinates = discretized_circle_outline_coordinates(radius, array_size)
            for y, x in coordinates:
                starting_map[y, x] = PATH
            starting_map[2, 1:-2] = PATH
            starting_map[2:-2, 1] = PATH
            starting_map[18, 1:-2] = PATH
            starting_map[2:-2:, 18] = PATH
            starting_map[9, 2:-2] = PATH
            starting_map[1:-1:, 9] = PATH
        elif map_type == 4:
            starting_map[1:5, 9] = PATH # in front of entrance
            starting_map[5, 4:-5] = PATH # in front of entrance
            starting_map[5:-5, 4] = PATH # on one side
            starting_map[5:-5, -6] = PATH # other side
            starting_map[-5, 4:-5] = PATH # far side
            dy, dx = 2, 1
        else:
            raise ValueError('bad map type')
        
        return StartingMap(starting_map, dy, dx, paths, list(rides2.values()), shops, size_y, size_x, all_owned_positions, total_num_rides, occupied_map, num_months, objective_num_guests, free_park_entry, zs, owned_map)
    
    def step(self, action):
        #assert self.valid_actions[action['position_and_type'][0]], f"Invalid action sent to env: {action}"
    
        result = self.apply_action_with_number(action)
        if result == -2:
            print("*** tried to build with not enough money ***")
            remaining_rides_to_place = self.total_num_rides_to_place - len(self.ride_list)
            reward = -remaining_rides_to_place
            terminated = True
            truncated = True
            
            self.update_target_arrays()  # <-- Aggiorna comunque lo stato
            self._fix_action_mask()
            
        elif result == -1:
            reward = -1
            terminated = False
            truncated = False
        else:
            assert result > -1
            self.update_target_arrays()
            self.cur_num_steps += 1
            self._fix_action_mask()
            terminated, truncated, reward = self.get_done_and_reward()
        
        observation = self._get_obs()
        info = self._get_info()
        #if terminated and self.verbose: print("End of episode")
        #return self._convert_to_float16(observation), reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def get_done_and_reward(self):
        reward = self.prev_loss - self.loss
        done = False
        truncated = False
        if self.sim_ran: # self.cur_num_steps == self.max_timesteps:
            done = True
        return done, truncated, reward

class MeetObjectiveRCTEnv(RCTEnv):
    def reset(self, **kwargs):
        self.sim_result = None
        observation, info = super().reset(**kwargs)
        self.max_timesteps = self.num_months
        return observation, info

    def get_cur_values(self):
        cur_values = super().get_cur_values()
        if self.sim_result == "lost":
            cur_values['won_objective'] = -1
        elif self.sim_result == "won":
            cur_values['won_objective'] = 1
        else:
            cur_values['won_objective'] = 0
        #cur_values['won_objective'] = 1 if self.sim_result == 'won' else 0
        return cur_values

    def update_target_arrays(self):
        #self.bridge.send_action('unpause')
        self.bridge.send_action("open_park")
        result = self.bridge.send_action("run_sim", num_ticks=-1) # run for one month
        result = result.decode('utf-8')
        self.bridge.send_action("close_park")
        #self.bridge.send_action('pause')

        self.sim_ran = True
        
        # run until objective is lost or won
        if result.startswith('lost'):
            self.sim_result = 'lost'
            print("Objective lost")
        elif result.startswith('won'):
            self.sim_result = 'won'
            print("Objective won")
        else:
            assert result.startswith('done'), f"Unexpected result: {result}"
    
        BasePathEnv.update_target_arrays(self)

    from collections import Counter

    # In rct.py

    def get_done_and_reward(self):
        reward = self.prev_loss - self.loss
        done = False
        truncated = False

        # GUEST BOOST: incentivo progressivo sopra 50% dell'obiettivo
        current_guests = self.raw_values.get('num_guests', 0) if hasattr(self, 'raw_values') else 0
        target_guests = self.objective_num_guests
        if target_guests > 0 and current_guests > target_guests * 0.5:
            # Boost quadratico scalato (max ~0.3 per evitare di dominare il segnale)
            normalized_progress = current_guests / target_guests
            reward += 0.3 * (normalized_progress ** 2)

        if self.sim_result == "lost":
            # Formattazione evidente per la sconfitta
            print("\n" + "!" * 60)
            print(f"[RCT_END] SCONFITTA su {self.cur_park_filename}")
            print(f"          Giostre piazzate: {len(self.ride_list)} / {sum(self.num_rides_per_month)}")
            print(f"          Step totali: {self.cur_num_steps}")
            print("!" * 60 + "\n")

            reward -= 5.0
            done = True

        elif self.sim_result == "won":
            # Formattazione evidente per la vittoria
            print("\n" + "*" * 60)
            print(f"[RCT_END] VITTORIA SU {self.cur_park_filename} !!!")
            print(f"          Obiettivo raggiunto con {len(self.ride_list)} giostre.")
            print("*" * 60 + "\n")

            reward += 10.0
            done = True

        elif self.cur_month >= MAX_NUM_MONTHS:
            done = True

        return done, truncated, reward

class ResearchMeetObjectiveRCTEnv(MeetObjectiveRCTEnv):
    def get_observation_space(self):
        MAX_VALUE = 1
        if self.action_space_type == 'normal':
            return Dict({
                "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=bool),
                "matrix": Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(*self.obs_size, self.num_grid_channels), dtype=np.float32),
                "flat": Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(self.num_flat_channels,), dtype=np.float32),
                'ride_action_mask_0': Box(low=0, high=1, shape=(len(self.ride_types)+1,), dtype=bool),

                # metrics for tensorboard.
                **{f'{target.name}': Box(-1, 10000, shape=(1,), dtype=np.float32) for target in self.target_ranges},
            })

        else:
            return Dict({
                "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=bool),
                "matrix": Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(*self.obs_size, self.num_grid_channels), dtype=np.float32),
                "flat": Box(low=-MAX_VALUE, high=MAX_VALUE, shape=(self.num_flat_channels,), dtype=np.float32),
                **{f'ride_action_mask_{i}': Box(low=0, high=1, shape=(len(self.ride_types),), dtype=bool) for i in range(len(self.extended_ride_types))},

                # metrics for tensorboard.
                **{f'{target.name}': Box(-1, 10000, shape=(1,), dtype=np.float32) for target in self.target_ranges},
            })
    
    def _get_obs(self):
        # bugfix: https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614/2
        obs = OrderedDict()

        kvs = [
            ("action_mask", self.valid_actions),
            ("matrix", self._get_matrix()),
            ("flat", self._get_flat_vector()),
            *[(i, np.array(mask, dtype=bool)) for i, mask in self.ride_action_masks.items()],
            
            # metrics for tensorboard.
            *[(f'{target.name}', np.array([self.cur_values[target.name]], dtype=np.float32)) for target in self.target_ranges],
        ]
        kvs.sort()
        for k, v in kvs:
            obs[k] = v
        #obs.update({OrderedDict(kvs)})
        #self.validate_observation(obs)

        obs['matrix'] = obs['matrix'].astype(np.float32)
        obs['flat'] = obs['flat'].astype(np.float32)
        obs['action_mask'] = obs['action_mask'].astype(np.float32)
            
        return obs
    
    def _get_info(self):
        return { target.name: self.cur_values[target.name] for target in self.target_ranges }

    def _fix_action_mask(self):
        super()._fix_action_mask()

        if np.sum(self.valid_actions) == 0:
            self.valid_actions = np.ones_like(self.valid_actions, dtype=bool)
            self.ride_action_masks = {
                'ride_action_mask_0': [False for _ in self.ride_types] + [True]
            }

    def get_random_starting_rides(self):
        restroom = [(r, ride_data) for r, ride_data in enumerate(self.ride_types) if ride_data.name == 'RIDE_TYPE_TOILETS'][0]
        kiosk = [(r, ride_data) for r, ride_data in enumerate(self.ride_types) if ride_data.name == 'RIDE_TYPE_INFORMATION_KIOSK'][0]
        
        shops = [(r, ride_data) for r, ride_data in enumerate(self.ride_types) if ride_data.category == 'shop' and ride_data.name not in ['RIDE_TYPE_TOILETS', 'RIDE_TYPE_INFORMATION_KIOSK']]
        gentle_rides = [(r, ride_data) for r, ride_data in enumerate(self.ride_types) if ride_data.category == 'gentle']
        thrill_rides = [(r, ride_data) for r, ride_data in enumerate(self.ride_types) if ride_data.category == 'thrill']

        # restroom always present
        selected_rides = [restroom]

        # kiosk usually present
        if random.random() < 0.5:
            selected_rides.append(kiosk)

        # choose between 1 and 4 gentle rides, with the average being 2.8
        selected_rides.extend(random.sample(gentle_rides, random.randint(1, 5)))

        # choose between 1 and 4 thrill rides, with the average being 2.3
        selected_rides.extend(random.sample(thrill_rides, random.randint(1, 4)))

        # choose between 2 and 9 shops, with the average being 6.5
        selected_rides.extend(random.sample(shops, random.randint(2, 9)))

        return selected_rides

    def refresh_available_ride_list(self):
        #self.get_all_possible_rides_in_dataset()
        #self.bridge.send_action('unpause')
        msg = self.bridge.send_action("get_available_rides")
        #self.bridge.send_action("pause")

        rides = json.loads(msg)
        #pprint(rides)

        #print("Available rides for timestep", self.pct_to_objective_end, ":")
        
        if len(self.available_ride_types) == 0:
            #self.rides.fill(-1)
            if self.random_invalid_rides:
                selected_rides = self.get_random_starting_rides()
                for r, ride_data in selected_rides:
                    self.available_ride_types.add((ride_data.ride_type, ride_data.shop_name))
                    self.ride_validities[r] = 1
                    self.ride_research_history[0].append(ride_data)
                    #print("Randomly selected ride:", ride_data.name, "type:", ride_data.ride_type, "shop:", ride_data.shop_name)
                self.last_month_ride_researched = 0
                #print("Randomly selected rides at start:", [ride_data.name for _, ride_data in selected_rides])

        if self.random_invalid_rides:
            # add one new ride every month, starting at month 6 (since research can take a while to kick in)
            if self.cur_month >= 5 and self.cur_month > self.last_month_ride_researched:
                # add a new ride
                researchable_rides = [(r, ride_data) for r, ride_data in enumerate(self.ride_types) if (ride_data.ride_type, ride_data.shop_name) not in self.available_ride_types]
                if len(researchable_rides) > 0:
                    ride_index, ride_to_research = random.choice(researchable_rides)
                    self.available_ride_types.add((ride_to_research.ride_type, ride_to_research.shop_name))
                    self.ride_validities[ride_index] = 1
                    self.last_month_ride_researched = self.cur_month
                    self.ride_research_history[self.cur_month].append(ride_to_research)
                    #print("Researching new ride:", ride_to_research.name, "type:", ride_to_research.ride_type, "shop:", ride_to_research.shop_name)
            
            # take a random sample of the researched rides to make available.
            self.ride_validities[:] = 0
            available_rides = random.sample(self.available_ride_types, len(self.available_ride_types)//2)
            for ride_type, shop_name in available_rides:
                for r, ride_data in enumerate(self.ride_types):
                    if ride_data.ride_type == ride_type and (ride_data.name not in ['RIDE_TYPE_FOOD_STALL', 'RIDE_TYPE_DRINK_STALL', 'RIDE_TYPE_SHOP'] or ride_data.shop_name == '' or ride_data.shop_name == shop_name):
                        self.ride_validities[r] = 1
                        #print("Game research:", self.cur_month, "available ride:", ride_data.name, "type:", ride_data.ride_type, "shop:", ride_data.shop_name)
                        break
                else:
                    assert False, f"Invalid ride type: {ride_type}"
        else:
            for entry in rides:
                ride_type = int(entry['type'])
                name = entry['name']
                for r, ride_data in enumerate(self.ride_types):
                    if (ride_data.ride_type, ride_data.shop_name) in self.available_ride_types:
                        continue
                    if ride_type == ride_data.ride_type and (ride_data.name not in ['RIDE_TYPE_FOOD_STALL', 'RIDE_TYPE_DRINK_STALL', 'RIDE_TYPE_SHOP'] or ride_data.shop_name == '' or ride_data.shop_name == name): # and action_num not in self.available_ride_types:
                        self.available_ride_types.add((ride_data.ride_type, ride_data.shop_name))
                        self.ride_research_history[self.cur_month].append(ride_data)
                        self.ride_validities[r] = 1
                        #print("Game research:", self.cur_park_filename, self.cur_month, "available ride:", ride_data.name, f"type: {entry['type']}, entry_index: {entry['entry_index']}; name: {entry['name']}", ride_data.shop_name)
                        break
                #else:
                #    print(self.cur_park_filename, "Unknown ride type:", entry)
        
        assert len(self.available_ride_types) > 0, "No available rides!"
        #'''
        if self.action_space_type == 'normal':
            validities = [False] * (len(self.ride_types) + 1)
            for r, ride_data in enumerate(self.ride_types):
                if (ride_data.ride_type, ride_data.shop_name) in self.available_ride_types and self.ride_quantities[r] < 1 and self.ride_validities[r] == 1:
                    validities[r] = True
            self.ride_action_masks = {
                'ride_action_mask_0': validities
            }
        else:
            self.ride_action_masks = {}
            for i, (_, _) in enumerate(list(self.extended_ride_types.items())):
                valids = []
                for ride_data in self.ride_types:
                    valids.append(ride_data.ride_type in self.available_ride_types)
                self.ride_action_masks[f'ride_action_mask_{i}'] = valids
        #input("done checking available rides")
        #'''
        #for i, (action_num, _) in enumerate(self.ride_types):
        #    self.ride_validities[:, :, i] = action_num in self.available_ride_types

    def get_all_possible_rides_in_dataset(self):
        all_possible_rides = defaultdict(set)
        for park in self.parks:
            self.bridge.send_park(park.filename)
            self.bridge.send_action('pause')
            self.bridge.send_action('set_speed', speed=8)

            message = ''
            while not message.startswith('lost'):
                message = self.bridge.send_action('run_sim', num_ticks=-1)
                message = message.decode('utf-8')
            
            msg = self.bridge.send_action("get_available_rides")
            rides = json.loads(msg)
            for entry in rides:
                ride_type = int(entry['type'])
                name = entry['name']
                for r, ride_data in enumerate(self.ride_types):
                    if ride_type == ride_data.ride_type: # and action_num not in self.available_ride_types:
                        all_possible_rides[ride_type].add((name)) #, entry["entry_index"]))
                        #self.available_ride_types.add(action_num)
                        #self.rides[:, :, r] = 0
                        #print(self.cur_park_filename, self.cur_month, "available ride:", ride_data.name, f"type: {entry['type']}, entry_index: {entry['entry_index']}; name: {entry['name']}")
                        break
                #else:
                #    print(self.cur_park_filename, "Unknown ride type:", RCT_RIDE_TYPES[ride_type])
        pprint(all_possible_rides)
        input()

    def can_place_ride_type(self, action_num, shop_name):
        for r, ride_data in enumerate(self.ride_types):
            if ride_data.ride_type == action_num and (ride_data.name not in ['RIDE_TYPE_FOOD_STALL', 'RIDE_TYPE_DRINK_STALL', 'RIDE_TYPE_SHOP'] or ride_data.shop_name == '' or ride_data.shop_name == shop_name):
                return self.ride_validities[r]
        else:
            assert False, f"Invalid ride type: {action_num}"
        #return (action_num, shop_name) in self.available_ride_types

    def get_cur_values(self):
        self.refresh_available_ride_list()
        cur_values = super().get_cur_values()
        return cur_values

class DaysResearchMeetObjectiveRCTEnv(ResearchMeetObjectiveRCTEnv):
    def update_target_arrays(self):
        ind = self.cur_month if self.cur_month < len(self.num_rides_per_month) else -1
        num_rides_this_month = self.num_rides_per_month[ind]
        if num_rides_this_month == 0:
            num_ticks_per_ride = NUM_TICKS_PER_MONTH #* 2
        else:
            num_ticks_per_ride = NUM_TICKS_PER_MONTH // num_rides_this_month

        #self.bridge.send_action('unpause')
        self.bridge.send_action('set_speed', speed=8)
        result = self.bridge.send_action("run_sim", num_ticks=num_ticks_per_ride) # run for one month
        #self.bridge.send_action('set_speed', speed=1)
        #self.bridge.send_action('pause')
        result = result.decode('utf-8')

        # run until objective is lost or won
        if result.startswith('lost'):
            self.sim_result = 'lost'
            #print("Objective lost", self.cur_park_filename)
        elif result.startswith('won'):
            self.sim_result = 'won'
            #print("Objective won", self.cur_park_filename)
        elif self.cur_month >= MAX_NUM_MONTHS:
             # Force termination for open-ended scenarios
             self.sim_result = 'lost'
             #print("Forced Time Limit Reached", self.cur_park_filename)
        else:
            assert result.startswith('done'), f"Unexpected result: {result}"
    
        BasePathEnv.update_target_arrays(self)
