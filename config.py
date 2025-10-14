# Diffusion Model Hyperparameters (from Table I in paper)
T = 3000  # Number of diffusion steps
BETA_START = 1e-4  # β₀
BETA_END = 0.02   # βT-1
NOISE_SCHEDULE = "linear"  # Linear noise schedule

# Data Parameters
SEGMENT_LENGTH = 3000  # 0.25 s at 12 kHz sampling rate
SEGMENT_STEP = 750     # 75% overlap
SEVERITY_CODES = ["007"]  # Fault diameter 0.007 inches
INCLUDE_NORMAL = True
SAMPLING_RATE = 12000  # 12 kHz

# Training Parameters
NUM_EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OPTIMIZER = "Adam"
OPTIMIZER_BETAS = (0.9, 0.999)

# Model Architecture Parameters
IN_CHANNELS = 1
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4, 8)
NUM_ATTENTION_HEADS = 4
TIME_EMB_DIM = 256
ATTENTION_RESOLUTIONS = [256, 512]  # Apply attention at 256 and 512 channels

# Data paths
DATA_DIR = "data/CWRU-dataset"
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = "output/trained_model.pth"

# Device settings
DEVICE = "cuda"  # Will fallback to CPU if CUDA unavailable
NUM_WORKERS = 4
PIN_MEMORY = True

# Evaluation parameters
NUM_SYNTHETIC_SAMPLES = 100
EVALUATION_METRICS = ["time_domain", "frequency_domain", "statistical"]