from collections import namedtuple

sign = lambda x: x and (1 if x >= 0 else -1)

PATH, QUEUE, RIDE_ENTRANCE, SHOP = 1, 2, 3, 4
ELEMENT_TYPES = [PATH, RIDE_ENTRANCE] #, QUEUE, RIDE_ENTRANCE, SHOP]
XY_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
XY_DELTAS_WITH_DIAGONAL = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

MIN_RIDE_HEIGHT, MAX_RIDE_HEIGHT = 5, 100

NUM_TICKS_PER_MONTH = 16383
NUM_TICKS_PER_WEEK = NUM_TICKS_PER_MONTH // 4

# ride_type is to be sent to the game.
RideDef = namedtuple("RideDef", "ride_type name default_price symbol category size shop_name")

RCT_RIDE_TYPES = [
    "RIDE_TYPE_SPIRAL_ROLLER_COASTER", # 0
    "RIDE_TYPE_STAND_UP_ROLLER_COASTER", # 1
    "RIDE_TYPE_SUSPENDED_SWINGING_COASTER", # 2
    "RIDE_TYPE_INVERTED_ROLLER_COASTER", # 3
    "RIDE_TYPE_JUNIOR_ROLLER_COASTER",
    "RIDE_TYPE_MINIATURE_RAILWAY",
    "RIDE_TYPE_MONORAIL",
    "RIDE_TYPE_MINI_SUSPENDED_COASTER",
    "RIDE_TYPE_BOAT_HIRE",
    "RIDE_TYPE_WOODEN_WILD_MOUSE",
    "RIDE_TYPE_STEEPLECHASE",
    "RIDE_TYPE_CAR_RIDE",
    "RIDE_TYPE_LAUNCHED_FREEFALL",
    "RIDE_TYPE_BOBSLEIGH_COASTER",
    "RIDE_TYPE_OBSERVATION_TOWER",
    "RIDE_TYPE_LOOPING_ROLLER_COASTER",
    "RIDE_TYPE_DINGHY_SLIDE",
    "RIDE_TYPE_MINE_TRAIN_COASTER",
    "RIDE_TYPE_CHAIRLIFT",  # 18
    "RIDE_TYPE_CORKSCREW_ROLLER_COASTER",
    "RIDE_TYPE_MAZE",
    "RIDE_TYPE_SPIRAL_SLIDE", # 21
    "RIDE_TYPE_GO_KARTS",
    "RIDE_TYPE_LOG_FLUME",
    "RIDE_TYPE_RIVER_RAPIDS",
    "RIDE_TYPE_DODGEMS", # 25
    "RIDE_TYPE_SWINGING_SHIP",
    "RIDE_TYPE_SWINGING_INVERTER_SHIP",
    "RIDE_TYPE_FOOD_STALL", # 28
    "RIDE_TYPE_1D",
    "RIDE_TYPE_DRINK_STALL", # 30
    "RIDE_TYPE_1F",
    "RIDE_TYPE_SHOP",
    "RIDE_TYPE_MERRY_GO_ROUND",
    "RIDE_TYPE_22",
    "RIDE_TYPE_INFORMATION_KIOSK",
    "RIDE_TYPE_TOILETS",
    "RIDE_TYPE_FERRIS_WHEEL",
    "RIDE_TYPE_MOTION_SIMULATOR",
    "RIDE_TYPE_3D_CINEMA",   #28
    "RIDE_TYPE_TOP_SPIN",
    "RIDE_TYPE_SPACE_RINGS",
    "RIDE_TYPE_REVERSE_FREEFALL_COASTER",
    "RIDE_TYPE_LIFT",
    "RIDE_TYPE_VERTICAL_DROP_ROLLER_COASTER",
    "RIDE_TYPE_CASH_MACHINE",
    "RIDE_TYPE_TWIST",
    "RIDE_TYPE_HAUNTED_HOUSE",
    "RIDE_TYPE_FIRST_AID",
    "RIDE_TYPE_CIRCUS",
    "RIDE_TYPE_GHOST_TRAIN",
    "RIDE_TYPE_TWISTER_ROLLER_COASTER",
    "RIDE_TYPE_WOODEN_ROLLER_COASTER",
    "RIDE_TYPE_SIDE_FRICTION_ROLLER_COASTER",
    "RIDE_TYPE_STEEL_WILD_MOUSE",
    "RIDE_TYPE_MULTI_DIMENSION_ROLLER_COASTER",
    "RIDE_TYPE_MULTI_DIMENSION_ROLLER_COASTER_ALT",
    "RIDE_TYPE_FLYING_ROLLER_COASTER",
    "RIDE_TYPE_FLYING_ROLLER_COASTER_ALT",
    "RIDE_TYPE_VIRGINIA_REEL",
    "RIDE_TYPE_SPLASH_BOATS",
    "RIDE_TYPE_MINI_HELICOPTERS",
    "RIDE_TYPE_LAY_DOWN_ROLLER_COASTER",
    "RIDE_TYPE_SUSPENDED_MONORAIL",
    "RIDE_TYPE_LAY_DOWN_ROLLER_COASTER_ALT",
    "RIDE_TYPE_REVERSER_ROLLER_COASTER",
    "RIDE_TYPE_HEARTLINE_TWISTER_COASTER",
    "RIDE_TYPE_MINI_GOLF",
    "RIDE_TYPE_GIGA_COASTER",
    "RIDE_TYPE_ROTO_DROP",
    "RIDE_TYPE_FLYING_SAUCERS",
    "RIDE_TYPE_CROOKED_HOUSE",
    "RIDE_TYPE_MONORAIL_CYCLES",
    "RIDE_TYPE_COMPACT_INVERTED_COASTER",
    "RIDE_TYPE_WATER_COASTER",
    "RIDE_TYPE_AIR_POWERED_VERTICAL_COASTER",
    "RIDE_TYPE_INVERTED_HAIRPIN_COASTER",
    "RIDE_TYPE_MAGIC_CARPET",
    "RIDE_TYPE_SUBMARINE_RIDE",
    "RIDE_TYPE_RIVER_RAFTS",
    "RIDE_TYPE_50",
    "RIDE_TYPE_ENTERPRISE",
    "RIDE_TYPE_52",
    "RIDE_TYPE_53",
    "RIDE_TYPE_54",
    "RIDE_TYPE_55",
    "RIDE_TYPE_INVERTED_IMPULSE_COASTER",
    "RIDE_TYPE_MINI_ROLLER_COASTER",
    "RIDE_TYPE_MINE_RIDE",
    "RIDE_TYPE_59",
    "RIDE_TYPE_LIM_LAUNCHED_ROLLER_COASTER",
    "RIDE_TYPE_HYPERCOASTER",
    "RIDE_TYPE_HYPER_TWISTER",
    "RIDE_TYPE_MONSTER_TRUCKS",
    "RIDE_TYPE_SPINNING_WILD_MOUSE",
    "RIDE_TYPE_CLASSIC_MINI_ROLLER_COASTER",
    "RIDE_TYPE_HYBRID_COASTER",
    "RIDE_TYPE_SINGLE_RAIL_ROLLER_COASTER",

    "RIDE_TYPE_COUNT"
]
RCT_RIDE_TYPES = dict(enumerate(RCT_RIDE_TYPES))
RCT_RIDE_TYPE_SYMBOLS = {
    'RIDE_TYPE_FOOD_STALL': 'f',
    'RIDE_TYPE_DRINK_STALL': 'd',
    'RIDE_TYPE_SHOP': 's',
    'RIDE_TYPE_INFORMATION_KIOSK': 'k',
    "RIDE_TYPE_SPIRAL_SLIDE": 'S',
    "RIDE_TYPE_MERRY_GO_ROUND": 'M',
    "RIDE_TYPE_MOTION_SIMULATOR": "O",
    "RIDE_TYPE_3D_CINEMA": "3",
    "RIDE_TYPE_TOP_SPIN": "T",
    "RIDE_TYPE_SPACE_RINGS": "P",
    "RIDE_TYPE_TWIST": "W",
    "RIDE_TYPE_HAUNTED_HOUSE": "H",
    "RIDE_TYPE_CIRCUS": "C",
    "RIDE_TYPE_CROOKED_HOUSE": "R",
    "RIDE_TYPE_TOILETS": 't',
    "RIDE_TYPE_CASH_MACHINE": '$',
    "RIDE_TYPE_FIRST_AID": 'a',
    "RIDE_TYPE_FLYING_SAUCERS": "Y",
    "RIDE_TYPE_ENTERPRISE": "E",
    "RIDE_TYPE_MAGIC_CARPET": '[',
    "RIDE_TYPE_SWINGING_INVERTER_SHIP": "<",
    "RIDE_TYPE_SWINGING_SHIP": ">",
    "RIDE_TYPE_FERRIS_WHEEL": "*",
}
RCT_RIDE_TYPE_DEFAULT_PRICES = {
    'RIDE_TYPE_FOOD_STALL': 1.5,
    'RIDE_TYPE_DRINK_STALL': 1.2,
    'RIDE_TYPE_INFORMATION_KIOSK': 2.5,
    "RIDE_TYPE_SPIRAL_SLIDE": 1.5,
    "RIDE_TYPE_MERRY_GO_ROUND": 1.0,
    "RIDE_TYPE_MOTION_SIMULATOR": 2.0,
    "RIDE_TYPE_3D_CINEMA": 2.0,
    "RIDE_TYPE_TOP_SPIN": 2.0,
    "RIDE_TYPE_SPACE_RINGS": 0.5,
    "RIDE_TYPE_TWIST": 1.0,
    "RIDE_TYPE_HAUNTED_HOUSE": 1.0,
    "RIDE_TYPE_CIRCUS": 1.2,
    "RIDE_TYPE_CROOKED_HOUSE": 0.6,
    "RIDE_TYPE_TOILETS": 0.0,
    "RIDE_TYPE_CASH_MACHINE": 0.0,
    "RIDE_TYPE_FIRST_AID": 0.0,
    "RIDE_TYPE_FLYING_SAUCERS": 1.5,
    "RIDE_TYPE_ENTERPRISE": 2.0,
    "RIDE_TYPE_SHOP": 1.5, # todo: shops have different prices depending on items
    "RIDE_TYPE_MAGIC_CARPET": 1.5,
    "RIDE_TYPE_SWINGING_INVERTER_SHIP": 1.5,
    "RIDE_TYPE_SWINGING_SHIP": 1.5,
    "RIDE_TYPE_FERRIS_WHEEL": 1,
}
RCT_RIDE_TYPE_CATEGORIES = {
    "RIDE_TYPE_FOOD_STALL": "shop",
    "RIDE_TYPE_DRINK_STALL": "shop",
    "RIDE_TYPE_INFORMATION_KIOSK": "shop",
    "RIDE_TYPE_SPIRAL_SLIDE": "gentle",
    "RIDE_TYPE_MERRY_GO_ROUND": "gentle",
    "RIDE_TYPE_MOTION_SIMULATOR": "thrill",
    "RIDE_TYPE_3D_CINEMA": "thrill",
    "RIDE_TYPE_TOP_SPIN": "thrill",
    "RIDE_TYPE_SPACE_RINGS": "gentle",
    "RIDE_TYPE_TWIST": "thrill",
    "RIDE_TYPE_HAUNTED_HOUSE": "gentle",
    "RIDE_TYPE_CIRCUS": "gentle",
    "RIDE_TYPE_CROOKED_HOUSE": "gentle",
    "RIDE_TYPE_TOILETS": "shop",
    "RIDE_TYPE_CASH_MACHINE": "shop",
    "RIDE_TYPE_FIRST_AID": "shop",
    "RIDE_TYPE_FLYING_SAUCERS": "gentle",
    "RIDE_TYPE_ENTERPRISE": "thrill",
    "RIDE_TYPE_SHOP": "shop",
    "RIDE_TYPE_MAGIC_CARPET": "thrill",
    "RIDE_TYPE_SWINGING_INVERTER_SHIP": "thrill",
    "RIDE_TYPE_SWINGING_SHIP": "thrill",
    "RIDE_TYPE_FERRIS_WHEEL": "gentle",
}

RCT_SMALL_RIDE_TYPES = [
    RideDef(ride_type=k,
            name=v,
            default_price=RCT_RIDE_TYPE_DEFAULT_PRICES[v],
            symbol=RCT_RIDE_TYPE_SYMBOLS[v],
            category=RCT_RIDE_TYPE_CATEGORIES[v],
            size='3x3',
            shop_name='') for k, v in RCT_RIDE_TYPES.items() if v in [
    "RIDE_TYPE_MERRY_GO_ROUND",
    "RIDE_TYPE_3D_CINEMA",
    "RIDE_TYPE_TOP_SPIN",
    "RIDE_TYPE_SPACE_RINGS",
    "RIDE_TYPE_TWIST",
    "RIDE_TYPE_HAUNTED_HOUSE",
    "RIDE_TYPE_CIRCUS",
    "RIDE_TYPE_CROOKED_HOUSE",
]]
RCT_VERY_SMALL_RIDE_TYPES = [RideDef(k, v, default_price=RCT_RIDE_TYPE_DEFAULT_PRICES[v], symbol=RCT_RIDE_TYPE_SYMBOLS[v], category=RCT_RIDE_TYPE_CATEGORIES[v], size='2x2', shop_name='') for k, v in RCT_RIDE_TYPES.items() if v in [
    "RIDE_TYPE_MOTION_SIMULATOR",
    "RIDE_TYPE_SPIRAL_SLIDE",
]]
RCT_LARGE_RIDE_TYPES = [RideDef(k, v, default_price=RCT_RIDE_TYPE_DEFAULT_PRICES[v], symbol=RCT_RIDE_TYPE_SYMBOLS[v], category=RCT_RIDE_TYPE_CATEGORIES[v], size='4x4', shop_name='') for k, v in RCT_RIDE_TYPES.items() if v in [
    "RIDE_TYPE_FLYING_SAUCERS",
    "RIDE_TYPE_ENTERPRISE",
]]
RCT_5x1_RIDE_TYPES = [RideDef(k, v, default_price=RCT_RIDE_TYPE_DEFAULT_PRICES[v], symbol=RCT_RIDE_TYPE_SYMBOLS[v], category=RCT_RIDE_TYPE_CATEGORIES[v], size='5x1', shop_name='') for k, v in RCT_RIDE_TYPES.items() if v in [
    "RIDE_TYPE_MAGIC_CARPET",
    "RIDE_TYPE_SWINGING_INVERTER_SHIP",
]]
RCT_7x1_RIDE_TYPES = [RideDef(k, v, default_price=RCT_RIDE_TYPE_DEFAULT_PRICES[v], symbol=RCT_RIDE_TYPE_SYMBOLS[v], category=RCT_RIDE_TYPE_CATEGORIES[v], size='7x1', shop_name='') for k, v in RCT_RIDE_TYPES.items() if v in [
    "RIDE_TYPE_SWINGING_SHIP",
    "RIDE_TYPE_FERRIS_WHEEL",
]]
RCT_SHOP_RIDE_TYPES = [RideDef(k, v, default_price=RCT_RIDE_TYPE_DEFAULT_PRICES[v], symbol=RCT_RIDE_TYPE_SYMBOLS[v], category=RCT_RIDE_TYPE_CATEGORIES[v], size='1x1', shop_name='') for k, v in RCT_RIDE_TYPES.items() if v in [
    'RIDE_TYPE_INFORMATION_KIOSK',
    'RIDE_TYPE_TOILETS',
    'RIDE_TYPE_CASH_MACHINE',
    'RIDE_TYPE_FIRST_AID'
]]

food_stalls = ['Burger Bar', 'Candy Apple Stall', 'Chicken Nuggets Stall', 'Cookie Shop', 'Cotton Candy Stall', 'Donut Shop', 'Fried Chicken Stall', 'Fries Shop', 'Fruity Ices Stall', 'Funnel Cake Shop', 'Hot Dog Stall', 'Ice Cream Cone Stall', 'Pizza Stall', 'Popcorn Stall', 'Sea Food Stall', 'Sub Sandwich Stall']
food_stall_index = [k for k, v in RCT_RIDE_TYPES.items() if v == 'RIDE_TYPE_FOOD_STALL'][0]
for food_stall in food_stalls:
    shop = RideDef(ride_type=food_stall_index,
                   name='RIDE_TYPE_FOOD_STALL',
                   default_price=RCT_RIDE_TYPE_DEFAULT_PRICES['RIDE_TYPE_FOOD_STALL'],
                   symbol=RCT_RIDE_TYPE_SYMBOLS['RIDE_TYPE_FOOD_STALL'],
                   category=RCT_RIDE_TYPE_CATEGORIES['RIDE_TYPE_FOOD_STALL'], size='1x1',
                   shop_name=food_stall)
    RCT_SHOP_RIDE_TYPES.append(shop)

food_stalls = ['Coffee Shop', 'Drinks Stall', 'Hot Chocolate Stall', 'Lemonade Stall']
food_stall_index = [k for k, v in RCT_RIDE_TYPES.items() if v == 'RIDE_TYPE_DRINK_STALL'][0]
for food_stall in food_stalls:
    shop = RideDef(ride_type=food_stall_index,
                   name='RIDE_TYPE_DRINK_STALL',
                   default_price=RCT_RIDE_TYPE_DEFAULT_PRICES['RIDE_TYPE_DRINK_STALL'],
                   symbol=RCT_RIDE_TYPE_SYMBOLS['RIDE_TYPE_DRINK_STALL'],
                   category=RCT_RIDE_TYPE_CATEGORIES['RIDE_TYPE_DRINK_STALL'], size='1x1',
                   shop_name=food_stall)
    RCT_SHOP_RIDE_TYPES.append(shop)

food_stalls = ['Balloon Stall', 'Hat Stall', 'Souvenir Stall', 'T-Shirt Stall', 'Sunglasses Stall']
food_stall_index = [k for k, v in RCT_RIDE_TYPES.items() if v == 'RIDE_TYPE_SHOP'][0]
for food_stall in food_stalls:
    shop = RideDef(ride_type=food_stall_index,
                   name='RIDE_TYPE_SHOP',
                   default_price=RCT_RIDE_TYPE_DEFAULT_PRICES['RIDE_TYPE_SHOP'],
                   symbol=RCT_RIDE_TYPE_SYMBOLS['RIDE_TYPE_SHOP'],
                   category=RCT_RIDE_TYPE_CATEGORIES['RIDE_TYPE_SHOP'], size='1x1',
                   shop_name=food_stall)
    RCT_SHOP_RIDE_TYPES.append(shop)



#"RIDE_TYPE_LAUNCHED_FREEFALL",
#"RIDE_TYPE_ROTO_DROP",
#"RIDE_TYPE_OBSERVATION_TOWER",

ScenarioDifficulty = namedtuple("ScenarioDifficulty", "const name diff game")
base_scenarios = []

base_scenarios.extend([ScenarioDifficulty(*x, 'rct1') for x in [
    [ "SC_FOREST_FRONTIERS",          "Forest Frontiers",     "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_DYNAMITE_DUNES",            "Dynamite Dunes",       "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_LEAFY_LAKE",                "Leafy Lake",           "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_DIAMOND_HEIGHTS",           "Diamond Heights",      "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_EVERGREEN_GARDENS",         "Evergreen Gardens",    "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_BUMBLY_BEACH",              "Bumbly Beach",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_TRINITY_ISLANDS",           "Trinity Islands",      "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_KATIES_DREAMLAND",          "Katie's Dreamland",    "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_POKEY_PARK",                "Pokey Park",           "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_WHITE_WATER_PARK",          "White Water Park",     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_MILLENNIUM_MINES",          "Millennium Mines",     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_KARTS_COASTERS",            "Karts & Coasters",     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_MELS_WORLD",                "Mel's World",          "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_MYSTIC_MOUNTAIN",           "Mystic Mountain",      "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_PACIFIC_PYRAMIDS",          "Pacific Pyramids",     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_CRUMBLY_WOODS",             "Crumbly Woods",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_PARADISE_PIER",             "Paradise Pier",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_LIGHTNING_PEAKS",           "Lightning Peaks",      "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_IVORY_TOWERS",              "Ivory Towers",         "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_RAINBOW_VALLEY",            "Rainbow Valley",       "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_THUNDER_ROCK",              "Thunder Rock",         "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_MEGA_PARK",                 "Mega Park",            "SCENARIO_CATEGORY_OTHER"         ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'rct1aa') for x in [
    [ "SC_WHISPERING_CLIFFS",         "Whispering Cliffs",    "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_THREE_MONKEYS_PARK",        "Three Monkeys Park",   "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_CANARY_MINES",              "Canary Mines",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_BARONY_BRIDGE",             "Barony Bridge",        "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_FUNTOPIA",                  "Funtopia",             "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_HAUNTED_HARBOUR",           "Haunted Harbour",      "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_FUN_FORTRESS",              "Fun Fortress",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_FUTURE_WORLD",              "Future World",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_GENTLE_GLEN",               "Gentle Glen",          "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_JOLLY_JUNGLE",              "Jolly Jungle",         "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_HYDRO_HILLS",               "Hydro Hills",          "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_SPRIGHTLY_PARK",            "Sprightly Park",       "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_MAGIC_QUARTERS",            "Magic Quarters",       "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_FRUIT_FARM",                "Fruit Farm",           "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_BUTTERFLY_DAM",             "Butterfly Dam",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_COASTER_CANYON",            "Coaster Canyon",       "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_THUNDERSTORM_PARK",         "Thunderstorm Park",    "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_HARMONIC_HILLS",            "Harmonic Hills",       "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_ROMAN_VILLAGE",             "Roman Village",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_SWAMP_COVE",                "Swamp Cove",           "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_ADRENALINE_HEIGHTS",        "Adrenaline Heights",   "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UTOPIA_PARK",               "Utopia Park",          "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_ROTTING_HEIGHTS",           "Rotting Heights",      "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_FIASCO_FOREST",             "Fiasco Forest",        "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_PICKLE_PARK",               "Pickle Park",          "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_GIGGLE_DOWNS",              "Giggle Downs",         "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_MINERAL_PARK",              "Mineral Park",         "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_COASTER_CRAZY",             "Coaster Crazy",        "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_URBAN_PARK",                "Urban Park",           "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_GEOFFREY_GARDENS",          "Geoffrey Gardens",     "SCENARIO_CATEGORY_EXPERT"        ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'rct1ll') for x in [
    [   "SC_ICEBERG_ISLANDS",         "Iceberg Islands",      "SCENARIO_CATEGORY_BEGINNER"      ],
    [   "SC_VOLCANIA",                "Volcania",             "SCENARIO_CATEGORY_BEGINNER"      ],
    [   "SC_ARID_HEIGHTS",            "Arid Heights",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [   "SC_RAZOR_ROCKS",             "Razor Rocks",          "SCENARIO_CATEGORY_BEGINNER"      ],
    [   "SC_CRATER_LAKE",             "Crater Lake",          "SCENARIO_CATEGORY_BEGINNER"      ],
    [   "SC_VERTIGO_VIEWS",           "Vertigo Views",        "SCENARIO_CATEGORY_BEGINNER"      ],
    [   "SC_PARADISE_PIER_2",         "Paradise Pier 2",      "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_DRAGONS_COVE",            "Dragon's Cove",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_GOOD_KNIGHT_PARK",        "Good Knight Park",     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_WACKY_WARREN",            "Wacky Warren",         "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_GRAND_GLACIER",           "Grand Glacier",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_CRAZY_CRATERS",           "Crazy Craters",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_DUSTY_DESERT",            "Dusty Desert",         "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_WOODWORM_PARK",           "Woodworm Park",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_ICARUS_PARK",             "Icarus Park",          "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_SUNNY_SWAMPS",            "Sunny Swamps",         "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_FRIGHTMARE_HILLS",        "Frightmare Hills",     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_THUNDER_ROCKS",           "Thunder Rocks",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_OCTAGON_PARK",            "Octagon Park",         "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_PLEASURE_ISLAND",         "Pleasure Island",      "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_ICICLE_WORLDS",           "Icicle Worlds",        "SCENARIO_CATEGORY_CHALLENGING"   ],
    [   "SC_SOUTHERN_SANDS",          "Southern Sands",       "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_TINY_TOWERS",             "Tiny Towers",          "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_NEVERMORE_PARK",          "Nevermore Park",       "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_PACIFICA",                "Pacifica",             "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_URBAN_JUNGLE",            "Urban Jungle",         "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_TERROR_TOWN",             "Terror Town",          "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_MEGAWORLD_PARK",          "Megaworld Park",       "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_VENUS_PONDS",             "Venus Ponds",          "SCENARIO_CATEGORY_EXPERT"        ],
    [   "SC_MICRO_PARK",              "Micro Park",           "SCENARIO_CATEGORY_EXPERT"        ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'rct2') for x in [
    [ "SC_UNIDENTIFIED",              "Electric Fields",      "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Factory Capers",       "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Crazy Castle",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Dusty Greens",         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Bumbly Bazaar",        "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Infernal Views",       "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Lucky Lake",           "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Botany Breakers",      "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Alpine Adventures",    "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Gravity Gardens",      "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Extreme Heights",      "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Amity Airfield",       "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Ghost Town",           "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Fungus Woods",         "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Rainbow Summit",       "SCENARIO_CATEGORY_EXPERT"        ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'rct2ww') for x in [
    [ "SC_UNIDENTIFIED",              "North America - Grand Canyon",                     "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Asia - Great Wall of China Tourism Enhancement",   "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Africa - African Diamond Mine",                    "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Australasia - Ayers Rock",                         "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "North America - Rollercoaster Heaven",             "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Africa - Oasis",                                   "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "South America - Rio Carnival",                     "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Asia - Maharaja Palace",                           "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Africa - Victoria Falls",                          "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "N. America - Extreme Hawaiian Island",             "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "South America - Rain Forest Plateau",              "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Europe - Renovation",                              "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Antarctic - Ecological Salvage",                   "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Europe - European Cultural Festival",              "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Australasia - Fun at the Beach",                   "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "South America - Inca Lost City",                   "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Asia - Japanese Coastal Reclaim",                  "SCENARIO_CATEGORY_EXPERT"        ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'rct2tt') for x in [
    [ "SC_UNIDENTIFIED",              "Dark Age - Robin Hood",                            "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Prehistoric - After the Asteroid",                 "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Roaring Twenties - Prison Island",                 "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Future - First Encounters",                        "SCENARIO_CATEGORY_BEGINNER"      ],
    [ "SC_UNIDENTIFIED",              "Roaring Twenties - Schneider Cup",                 "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Prehistoric - Stone Age",                          "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Mythological - Cradle of Civilisation",            "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Rock 'n' Roll - Rock 'n' Roll",                    "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Future - Future World",                            "SCENARIO_CATEGORY_CHALLENGING"   ],
    [ "SC_UNIDENTIFIED",              "Roaring Twenties - Skyscrapers",                   "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Rock 'n' Roll - Flower Power",                     "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Dark Age - Castle",                                "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Mythological - Animatronic Film Set",              "SCENARIO_CATEGORY_EXPERT"        ],
    [ "SC_UNIDENTIFIED",              "Prehistoric - Jurassic Safari",                    "SCENARIO_CATEGORY_EXPERT"        ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'uces') for x in [
    [ "SC_UNIDENTIFIED",              "Lighthouse of Alexandria by Katatude for UCES",    "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Cleveland's Luna Park",                            "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Mount Vesuvius 1700 A.D. by Katatude for UCES",    "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "The Sandbox by Katatude for UCES",                 "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Niagara Falls & Gorge by Katatude for UCES",       "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Rocky Mountain Miners",                            "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "The Time Machine by Katatude for UCES",            "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Tower of Babel",                                   "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Transformation",                                   "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Urbis Incognitus",                                 "SCENARIO_CATEGORY_TIME_MACHINE"     ],
    [ "SC_UNIDENTIFIED",              "Beneath the Christmas Tree by Katatude for UCES",  "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "Bigrock Blast",                                    "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "Camp Mockingbird for UCES by Katatude",            "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "Choo Choo Town",                                   "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "Dragon Islands",                                   "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "Kiddy Karnival II",                                "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "Sand Dune",                                        "SCENARIO_CATEGORY_KATYS_DREAMWORLD" ],
    [ "SC_UNIDENTIFIED",              "UCES Halloween",                                   "SCENARIO_CATEGORY_OTHER"            ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'real') for x in [
    [ "SC_ALTON_TOWERS",              "Alton Towers",                                     "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_HEIDE_PARK",                "Heide-Park",                                       "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_BLACKPOOL_PLEASURE_BEACH",  "Blackpool Pleasure Beach",                         "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_UNIDENTIFIED",              "Six Flags Belgium",                                "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_UNIDENTIFIED",              "Six Flags Great Adventure",                        "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_UNIDENTIFIED",              "Six Flags Holland",                                "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_UNIDENTIFIED",              "Six Flags Magic Mountain",                         "SCENARIO_CATEGORY_REAL"  ],
    [ "SC_UNIDENTIFIED",              "Six Flags over Texas",                             "SCENARIO_CATEGORY_REAL"  ],
]])

base_scenarios.extend([ScenarioDifficulty(*x, 'other') for x in [
    [ "SC_FORT_ANACHRONISM",                          "Fort Anachronism",                                 "SCENARIO_CATEGORY_DLC"            ],
    [ "SC_PCPLAYER",                                  "PC Player",                                        "SCENARIO_CATEGORY_DLC"            ],
    [ "SC_PCGW",                                      "PC Gaming World",                                  "SCENARIO_CATEGORY_DLC"            ],
    [ "SC_GAMEPLAY",                                  "gameplay",                                         "SCENARIO_CATEGORY_DLC"            ],
    [ "SC_UNIDENTIFIED",                              "Panda World",                                      "SCENARIO_CATEGORY_DLC"            ],
    [ "SC_UNIDENTIFIED",                              "Build your own Six Flags Belgium",                 "SCENARIO_CATEGORY_BUILD_YOUR_OWN" ],
    [ "SC_UNIDENTIFIED",                              "Build your own Six Flags Great Adventure",         "SCENARIO_CATEGORY_BUILD_YOUR_OWN" ],
    [ "SC_UNIDENTIFIED",                              "Build your own Six Flags Holland",                 "SCENARIO_CATEGORY_BUILD_YOUR_OWN" ],
    [ "SC_UNIDENTIFIED",                              "Build your own Six Flags Magic Mountain",          "SCENARIO_CATEGORY_BUILD_YOUR_OWN" ],
    [ "SC_UNIDENTIFIED",                              "Build your own Six Flags Park",                    "SCENARIO_CATEGORY_BUILD_YOUR_OWN" ],
    [ "SC_UNIDENTIFIED",                              "Build your own Six Flags over Texas",              "SCENARIO_CATEGORY_BUILD_YOUR_OWN" ],
    [ "SC_UNIDENTIFIED",                              "Competition Land 1",                               "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_UNIDENTIFIED",                              "Competition Land 2",                               "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_BOBSLED_COMPETITION",                       "Bobsled Roller Coaster Competition",               "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_GO_KARTS_COMPETITION",                      "Go Karts Competition",                             "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_INVERTED_ROLLER_COASTER_COMPETITION",       "Inverted Roller Coaster Competition",              "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_MINE_TRAIN_COMPETITION",                    "Mine Train Roller Coaster Competition",            "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_STAND_UP_STEEL_ROLLER_COASTER_COMPETITION", "Stand-Up Steel Roller Coaster Competition",        "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_STEEL_CORKSCREW_COMPETITION",               "Steel Corkscrew Roller Coaster Competition",       "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_STEEL_MINI_ROLLER_COASTER_COMPETITION",     "Steel Mini Roller Coaster Competition",            "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_STEEL_ROLLER_COASTER_COMPETITION",          "Steel Roller Coaster Competition",                 "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_STEEL_TWISTER_COMPETITION",                 "Steel Twister Roller Coaster Competition",         "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_SUSPENDED_ROLLER_COASTER_COMPETITION",      "Suspended Roller Coaster Competition",             "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_WOODEN_ROLLER_COASTER_COMPETITION",         "Wooden Roller Coaster Competition",                "SCENARIO_CATEGORY_COMPETITIONS"   ],
    [ "SC_UNIDENTIFIED",                              "Tycoon Park",                                      "SCENARIO_CATEGORY_OTHER"          ],
]])

objectives = [
    "OBJECTIVE_NONE",
    "OBJECTIVE_GUESTS_BY",
    "OBJECTIVE_PARK_VALUE_BY",
    "OBJECTIVE_HAVE_FUN",
    "OBJECTIVE_BUILD_THE_BEST",
    "OBJECTIVE_10_ROLLERCOASTERS",
    "OBJECTIVE_GUESTS_AND_RATING",
    "OBJECTIVE_MONTHLY_RIDE_INCOME",
    "OBJECTIVE_10_ROLLERCOASTERS_LENGTH",
    "OBJECTIVE_FINISH_5_ROLLERCOASTERS",
    "OBJECTIVE_REPAY_LOAN_AND_PARK_VALUE",
    "OBJECTIVE_MONTHLY_FOOD_INCOME",
]

PARK_FLAGS_PARK_OPEN = (1 << 0)
PARK_FLAGS_SCENARIO_COMPLETE_NAME_INPUT = (1 << 1)
PARK_FLAGS_FORBID_LANDSCAPE_CHANGES = (1 << 2)
PARK_FLAGS_FORBID_TREE_REMOVAL = (1 << 3)
PARK_FLAGS_SHOW_REAL_GUEST_NAMES = (1 << 4)
PARK_FLAGS_FORBID_HIGH_CONSTRUCTION = (1 << 5) # below tree height
PARK_FLAGS_PREF_LESS_INTENSE_RIDES = (1 << 6)
PARK_FLAGS_FORBID_MARKETING_CAMPAIGN = (1 << 7)
PARK_FLAGS_ANTI_CHEAT_DEPRECATED = (1 << 8) # Not used anymore, used for cheat detection
PARK_FLAGS_PREF_MORE_INTENSE_RIDES = (1 << 9)
PARK_FLAGS_NO_MONEY = (1 << 11)
PARK_FLAGS_DIFFICULT_GUEST_GENERATION = (1 << 12)
PARK_FLAGS_PARK_FREE_ENTRY = (1 << 13)
PARK_FLAGS_DIFFICULT_PARK_RATING = (1 << 14)
PARK_FLAGS_LOCK_REAL_NAMES_OPTION_DEPRECATED = (1 << 15) # Deprecated now we use a persistent 'real names' setting
PARK_FLAGS_NO_MONEY_SCENARIO = (1 << 17) # Deprecated, originally used in scenario editor
PARK_FLAGS_SPRITES_INITIALISED = (1 << 18) # After a scenario is loaded this prevents edits in the scenario editor
PARK_FLAGS_SIX_FLAGS_DEPRECATED = (1 << 19) # Not used anymore

flags = [PARK_FLAGS_PARK_OPEN, PARK_FLAGS_SCENARIO_COMPLETE_NAME_INPUT, PARK_FLAGS_FORBID_LANDSCAPE_CHANGES, PARK_FLAGS_FORBID_TREE_REMOVAL, PARK_FLAGS_SHOW_REAL_GUEST_NAMES, PARK_FLAGS_FORBID_HIGH_CONSTRUCTION, PARK_FLAGS_PREF_LESS_INTENSE_RIDES, PARK_FLAGS_FORBID_MARKETING_CAMPAIGN, PARK_FLAGS_ANTI_CHEAT_DEPRECATED, PARK_FLAGS_PREF_MORE_INTENSE_RIDES, PARK_FLAGS_NO_MONEY, PARK_FLAGS_DIFFICULT_GUEST_GENERATION, PARK_FLAGS_PARK_FREE_ENTRY, PARK_FLAGS_DIFFICULT_PARK_RATING, PARK_FLAGS_LOCK_REAL_NAMES_OPTION_DEPRECATED, PARK_FLAGS_NO_MONEY_SCENARIO, PARK_FLAGS_SPRITES_INITIALISED, PARK_FLAGS_SIX_FLAGS_DEPRECATED]

# dictionary with flag as key, and the name of the flag as a string value
PARK_FLAGS = {flag: name for flag, name in zip(flags, [
    "PARK_FLAGS_PARK_OPEN",
    "PARK_FLAGS_SCENARIO_COMPLETE_NAME_INPUT",
    "PARK_FLAGS_FORBID_LANDSCAPE_CHANGES",
    "PARK_FLAGS_FORBID_TREE_REMOVAL",
    "PARK_FLAGS_SHOW_REAL_GUEST_NAMES",
    "PARK_FLAGS_FORBID_HIGH_CONSTRUCTION",
    "PARK_FLAGS_PREF_LESS_INTENSE_RIDES",
    "PARK_FLAGS_FORBID_MARKETING_CAMPAIGN",
    "PARK_FLAGS_ANTI_CHEAT_DEPRECATED",
    "PARK_FLAGS_PREF_MORE_INTENSE_RIDES",
    "PARK_FLAGS_NO_MONEY",
    "PARK_FLAGS_DIFFICULT_GUEST_GENERATION",
    "PARK_FLAGS_PARK_FREE_ENTRY",
    "PARK_FLAGS_DIFFICULT_PARK_RATING",
    "PARK_FLAGS_LOCK_REAL_NAMES_OPTION_DEPRECATED",
    "PARK_FLAGS_NO_MONEY_SCENARIO",
    "PARK_FLAGS_SPRITES_INITIALISED",
    "PARK_FLAGS_SIX_FLAGS_DEPRECATED",
])}

#PARK_FLAGS_RCT1_INTEREST = (1u << 30) # OpenRCT2 only
#PARK_FLAGS_UNLOCK_ALL_PRICES = (1u << 31) # OpenRCT2 only
