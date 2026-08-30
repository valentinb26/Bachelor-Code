# config.py
# Rein deklarative Datei für alle Parameter des Systems. Keine Logik.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent #src ordner 

# Directories
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model
GROUNDING_DINO_CONFIG = WEIGHTS_DIR / "config_swint.py"
GROUNDING_DINO_CHECKPOINT = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"

# Calibration
HOMOGRAPHY_PATH = DATA_DIR / "homography_matrix.npy"
CALIBRATION_PATH = DATA_DIR / "WIP.py"

# Planning
PLAN_PATH = DATA_DIR / "plan.json"

USE_CALIBRATION = False

# Roboter-Höhen (in mm)
Z_GROUND = -34 # wo ist der Boden
Z_SAFE = 100 # abstand vom Arm am Target
Z_DIP = -20 # Nick-Tiefe in mm
Z_LIFT = -(Z_GROUND) + Z_SAFE 
DOBOT_RETRIES = 3
DOBOT_TIMEOUT = 5 #sek
SCREW_ROTATIONS = 1

# Vision-Schwellwerte
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.4


# TODO: Vorrübergehende Lösung!
# Mapping für Slots zu Positionen 
SLOT_POSITIONS = {
    "slot_master_":        (200, 0, 10),
    "slot_front_":         (200, 0, 0),
    "slot_back_":          (200, 0, 0),
    "slot_top_1":          (200, 0, 0),
    "slot_top_2":          (200, 0, 0),
    "slot_top_3":          (200, 0, 0),
    "slot_left_1":         (70, -192, 53),
    "slot_left_2":         (154, -188, 53),
    "slot_right_1":        (70, -192, 53),
    "slot_right_2":        (154, -188, 53),
    "slot_side_cover_1":   (200, 0, 0),
    "slot_side_cover_2":   (200, 0, 0),
    "slot_side_cover_3":   (200, 0, 0),
    "slot_side_cover_4":   (200, 0, 0),
}

GRASP_STRATEGIES = {
    "screw": {
        "name": "screw",
        "use_orientation": False,  # Symmetrisch
        "gripper_rotation": 0,     # Egal
        "grasp_height_offset": 0,  # Greif oben
        "description": "Greife Schraube von oben, Z aus Höhe"
    },
    "wheel": {
        "name": "wheel",
        "use_orientation": False,
        "gripper_rotation": 0,
        "grasp_height_offset": 0,
        "description": "Greife Rad von oben"
    },
    "bumper": {
        "name": "bumper",
        "use_orientation": True,   # Orientierung wichtig!
        "gripper_rotation": "from_mask",  # Rotiere Gripper nach Maske
        "grasp_height_offset": 5,  # Etwas höher greifen
        "description": "Greife Bumper von oben, Orientierung nach längster Achse"
    },
    "side_cover": {
        "name": "side_cover",
        "use_orientation": True,
        "gripper_rotation": "from_mask",
        "grasp_height_offset": 3,
        "description": "Greife Side Cover nach Orientierung"
    },
    "top_addon": {
        "name": "top_addon",
        "use_orientation": True,
        "gripper_rotation": "from_mask",
        "grasp_height_offset": 2,
        "description": "Greife Top Addon nach Orientierung"
    }
}


# Map für Objekte mit deren prompt und optinalen festen Position
OBJECT_CONFIG = {
    "blue_cover": {
        "prompt": "blue large cover",
        "height": 0,
        "pos": None #(200, 0, 10),
        
    },
    "side_cover": {
        #"prompt": "long grey metalic side piece",,
        "height": 0,
        "pos": None
    },
    "screw_long": {
        "prompt": "long red screw bolt",
        "height": 32, 
        "pos": (-25.5, 201, 18, 95)
    },
    "screw_short": {
        #"prompt": "small short red bolt",,
        "height": 0,
        "pos": None #(200, 0, 0)
    },
    "wheel_big": {
        #"prompt": "orange tires",,
        "height": 0,
        "pos": None
    },
    "wheel_small": {
        "prompt": "small black wheel",
        "height": 0,
        "pos": None
    },
    "addon": {
        #"prompt": "yellow long . yellow addon",
        "height": 0,
        "pos": None
    },
    "bumper_front": {
        "prompt": "",
        "height": 0,
        "pos": None
    },
    "bumper_back": {
        "prompt": "",
        "height": 0,
        "pos": None
    },
}