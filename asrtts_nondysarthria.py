import os
from pathlib import Path
import string
import tempfile
import subprocess
import sys
import wandb

#MUST INSTALL
#pip install nemo_toolkit['asr']
#pip install nemo_text_processing
#pip install nemo_toolkit['tts']
#pip install wandb
#python asrtts_zh.py


## Install dependencies
#!pip install wget
#!apt-get install sox libsndfile1 ffmpeg
#!pip install text-unidecode
### Install NeMo
#!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
#pip install omegaconf
#pip install pytorch_lightning
#pip install hydra-core
#pip install ffmpeg-python

# Run the pip command to install the wget module
#try:
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "wget"])
    #subprocess.run(['apt-get', 'install', '-y', 'sox'], check=True)
    #subprocess.run(['apt-get', 'install', '-y', 'libsndfile1'], check=True)
    ##subprocess.run(['apt-get', 'install', '-y', 'imageio[ffmpeg]'], check=True)
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg"])
    ##subprocess.run(['apt-get', 'install', '-y', 'text-unidecode'], check=True)
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
#    print("wandb package installed successfully.")
#except subprocess.CalledProcessError as e:
#    print(f"Error installing wget: {e}")

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from tqdm.auto import tqdm
import wget

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel, EncDecCTCModel, EncDecRNNTModel
from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel, HifiGanModel
#from nemo.utils.notebook_utils import download_an4

# Define the URL and directorypython asrtts
BRANCH = 'main' # Set the branch you want (e.g., "main", "dev", etc.)
config_dir = "configs/"

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this system.")

# Check if CUDA is available
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Using GPU: {torch.cuda.get_device_name(current_device)}")
else:
    print("No GPU is being used.")

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    print("NEMO TEXT PROCESSING package installed successfully.")
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

DATASETS_DIR = Path("./datasets")  # directory for data
CHECKPOINTS_DIR = Path("./checkpoints/")  # directory for checkpoints

# create directories if necessary
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

#download_an4(data_dir=f"{DATASETS_DIR}")
#AN4_DATASET = DATASETS_DIR / "an4"
LRDWWS_CONTROL_DATASET = DATASETS_DIR / "lrdwws/train/Control" 
#LRDWWS_DYSARTHRIC_DATASET = DATASETS_DIR / "lrdwws/train/Uncontrol"

#CONSTRUCT TEXT-ONLY TRAINING DATA
# read original training data
#an4_train_data = read_manifest(AN4_DATASET / "train_manifest.json")
lrdwws_control_train_data = read_manifest(LRDWWS_CONTROL_DATASET / "manifest_train/train.json")
#lrdwws_uncontrol_train_data = read_manifest(LRDWWS_DYSARTHRIC_DATASET / "manifest_train/train.json")

# fill `text` and `tts_text` fields with the source data
textonly_data = []
#for record in an4_train_data:
for record in lrdwws_control_train_data:
    text = record["text"]
    textonly_data.append({"text": text, "tts_text": text})

WHITELIST_URL = (
    "https://raw.githubusercontent.com/NVIDIA/NeMo-text-processing/main/nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv",
    "https://raw.githubusercontent.com/NVIDIA/NeMo-text-processing/main/nemo_text_processing/text_normalization/zh/data/char/charset_national_standard_2013_8105.tsv",
    "https://raw.githubusercontent.com/NVIDIA/NeMo-text-processing/main/nemo_text_processing/text_normalization/zh/data/whitelist.tsv",
    "https://raw.githubusercontent.com/NVIDIA/NeMo-text-processing/main/nemo_text_processing/text_normalization/zh/data/char/fullwidth_to_halfwidth.tsv",
)

def get_normalizer() -> Normalizer:
    with tempfile.TemporaryDirectory() as data_dir:
        whitelist_path = Path(data_dir) / "lj_speech.tsv"
        if not whitelist_path.exists():
            for url in WHITELIST_URL:
                try:
                    wget.download(url, out=str(data_dir))
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

        '''normalizer = Normalizer(
            lang="en",
            input_case="cased",
            whitelist=str(whitelist_path),
            overwrite_cache=True,
            cache_dir=None,
        )'''

        normalizer = Normalizer(
            lang="zh",
            input_case="cased",
            whitelist=str(whitelist_path),
            overwrite_cache=True,
            cache_dir=None,
        )
    return normalizer

normalizer = get_normalizer()

for record in tqdm(textonly_data):
    record["tts_text_normalized"] = normalizer.normalize(
        record["tts_text"], verbose=False, punct_pre_process=True, punct_post_process=True
    )

#save manifest
#write_manifest(AN4_DATASET / "train_text_manifest.json", textonly_data)
write_manifest(LRDWWS_CONTROL_DATASET / "train_text_manifest.json", textonly_data)
#write_manifest(LRDWWS_DYSARTHRIC_DATASET / "train_text_manifest.json", textonly_data)

#Save pretrained checkpoints
#ASR_MODEL_PATH = CHECKPOINTS_DIR / "stt_en_conformer_ctc_small_ls.nemo"
ASR_MODEL_PATH = CHECKPOINTS_DIR / "stt_zh_conformer_ctc.nemo"
TTS_MODEL_PATH = CHECKPOINTS_DIR / "fastpitch.nemo"
ENHANCER_MODEL_PATH = CHECKPOINTS_DIR / "enhancer.nemo"


# asr model: stt_en_conformer_ctc_small_ls
#asr_model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small_ls")
model_path = Path("./checkpoints/Conformer-CTC-L_spe7000_zh-CN_2.0.nemo")
asr_model = EncDecCTCModelBPE.restore_from(str(model_path))
#asr_model = EncDecCTCModelBPE.load_state_dict(torch.load(model_path))

#asr_model = EncDecCTCModelBPE.from_pretrained(Path("./checkpoints/nvidia/Conformer-CTC-L_spe7000_zh-CN_2.0"))

#EncDecCTCModelBPE, EncDecRNNTBPEModel
#asr_model = EncDecRNNTModel.from_pretrained(Path("./checkpoints/stt_zh_conformer_transducer_large.nemo") #later then try this
#asr_model = EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/stt_zh_conformer_transducer_large")
#asr_model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_enzh_fastconformer_transducer_large_codesw")
#asr_model = EncDecRNNTBPEModel.restore_from(Path("./checkpoints/train_hybrid/finetuned_asr_model.nemo"))
asr_model.save_to(f"{ASR_MODEL_PATH}")

# tts model: tts_en_fastpitch_for_asr_finetuning
#tts_model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch_for_asr_finetuning")
tts_model = FastPitchModel.from_pretrained(model_name="tts_zh_fastpitch_sfspeech")
tts_model.save_to(f"{TTS_MODEL_PATH}")

#ADD VOCODER HERE

# enhancer model: tts_en_spectrogram_enhancer_for_asr_finetuning
enhancer_model = SpectrogramEnhancerModel.from_pretrained(model_name="tts_en_spectrogram_enhancer_for_asr_finetuning")
enhancer_model.save_to(f"{ENHANCER_MODEL_PATH}")

#Construct hybrid ASR-TTS model
#Config Parameters
#Hybrid ASR-TTS model consists of three parts:

#ASR model (EncDecCTCModelBPE, EncDecRNNTBPEModel or EncDecHybridRNNTCTCBPEModel)
#TTS Mel Spectrogram Generator (currently, only FastPitch model is supported)
#Enhancer model (optional)

# load config
#!wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/examples/asr/conf/asr_tts/hybrid_asr_tts.yaml

# Create the directory if it doesn't exist
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
# Define the file URL and local path
filename = "hybrid_asr_tts.yaml"  # Path to save the file locally
# Construct the full path
file_path = os.path.join(config_dir, filename)
# Check if the file exists locally
if not os.path.exists(file_path):
    try:
        subprocess.run(["wget", "-P", "configs/", f"https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/asr/conf/asr_tts/hybrid_asr_tts.yaml"], check=True)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the file: {e}")
else:
    print(f"{filename} already exists. Skipping download.")


config = OmegaConf.load("./configs/hybrid_asr_tts.yaml")
NUM_EPOCHS = 1

#We will use all available speakers (sampled uniformly).

TTS_SPEAKERS_PATH = Path("./checkpoints/speakers.txt")

with open(TTS_SPEAKERS_PATH, "w", encoding="utf-8") as f:
    for speaker_id in range(tts_model.cfg.n_speakers):
        print(speaker_id, file=f)

config.model.asr_model_path = ASR_MODEL_PATH
config.model.tts_model_path = TTS_MODEL_PATH
config.model.enhancer_model_path = ENHANCER_MODEL_PATH

# fuse BathNorm automatically in Conformer for better performance
config.model.asr_model_fuse_bn = True

# training data
# constructed dataset
#config.model.train_ds.text_data.manifest_filepath = str(AN4_DATASET / "train_text_manifest.json")
config.model.train_ds.text_data.manifest_filepath = str(LRDWWS_CONTROL_DATASET / "train_text_manifest.json")
#config.model.train_ds.text_data.manifest_filepath = str(LRDWWS_DYSARTHRIC_DATASET / "train_text_manifest.json")

# speakers for TTS model
config.model.train_ds.text_data.speakers_filepath = f"{TTS_SPEAKERS_PATH}"
config.model.train_ds.manifest_filepath = None  # audio-text pairs - we don't use them here
config.model.train_ds.batch_size = 8

# validation data
#config.model.validation_ds.manifest_filepath = str(AN4_DATASET / "test_manifest.json")
config.model.validation_ds.manifest_filepath = str(LRDWWS_CONTROL_DATASET / "manifest_dev/dev.json")
#config.model.validation_ds.manifest_filepath = str(LRDWWS_DYSARTHRIC_DATASET / "manifest_dev/dev.json")
config.model.validation_ds.batch_size = 8

# test data
config.model.test_ds.manifest_filepath = str(LRDWWS_CONTROL_DATASET / "manifest_test/test.json")
#config.model.validation_ds.manifest_filepath = str(LRDWWS_DYSARTHRIC_DATASET / "manifest_dev/test.json")
config.model.test_ds.batch_size = 8

config.trainer.max_epochs = NUM_EPOCHS

config.trainer.devices = 1
config.trainer.strategy = 'auto'  # use 1 device, no need for ddp strategy
config.trainer.accelerator = 'gpu'
OmegaConf.resolve(config)

#WANDB LOG
from pytorch_lightning.loggers.wandb import WandbLogger
# Paste your WANDB API key here
wandb_api_key = "4fa27f11cc974bfa6a002ca72f73a0bf1e95bc3f"
# Login to wandb
wandb.login(key=wandb_api_key)

# Initialize WandB (if not already done)
#wandb.init(project="ASRTTS", name="Original")

# Enable WandB logger in config
config.exp_manager.create_wandb_logger = True
config.exp_manager.wandb_logger_kwargs = {
    "project": "hybrid_asrtts",
    "name": "lrdwws_control",
}

# Create the WandB logger
wandb_logger = WandbLogger(**config.exp_manager.wandb_logger_kwargs)
# Remove `logger` from `config.trainer` if it exists
if "logger" in config.trainer:
    del config.trainer["logger"]

# #Construct trainer and ASRWithTTSModel
# Initialize the trainer
trainer = pl.Trainer(logger=wandb_logger, **config.trainer)
# Load the model
hybrid_model = ASRWithTTSModel(config.model)

'''
#VALIDATE THE MODEL BEFORE TRAINING
#Expect ~17.7% WER on the AN4 test data.
trainer.validate(hybrid_model)
# Extract WER
val_wer= trainer.callback_metrics.get("val_wer")
if val_wer is not None:
    print(f"Validation WER (Before Training): {val_wer}")
    wandb.log({"val_wer_before_training": float(val_wer)})  # Log as "val_wer_before_training"
else:
    print("WER not found in callback metrics after the first validation.")
'''

#Train the model
trainer.fit(hybrid_model)

#Validate the model after training
#Expect ~2% WER on the AN4 test data.
trainer.validate(hybrid_model)
# Extract WER
val_wer = trainer.callback_metrics.get("val_wer")
if val_wer is not None:
    print(f"Validation WER (After Training): {val_wer}")
    wandb.log({"val_wer_after_training": float(val_wer)})  # Log as "val_wer_after_training"
else:
    print("WER not found in callback metrics after the second validation.")
    
#Save final model. Extract pure ASR model
# save full model: the model can be further used for finetuning
hybrid_model.save_to("checkpoints/finetuned_hybrid_model.nemo")
# extract the resulting ASR model from the hybrid model
hybrid_model.save_asr_model_to("checkpoints/finetuned_asr_model.nemo") #checkpoints/train_hybrid/finetuned_asr_model.nemo


#hybrid_model.setup_test_data(test_data_config=config.model.test_ds)

#inference
# Assuming you want to transcribe an audio file
audio_file = './datasets/lrdwws/train/Control/wav/test/CM0025/CM0025_0015.wav'
# Perform inference (transcription)
transcription = hybrid_model.transcribe([audio_file])
print("Transcription:", transcription)

#hybrid_model.set_trainer(trainer)
#trainer.test(hybrid_model, ckpt_path=None)
#trainer.test(hybrid_model)
#TEST MODEL
#if hasattr(config.model, 'test_ds') and config.model.test_ds.manifest_filepath is not None:
#    if hybrid_model.prepare_test(trainer):
#        trainer.test(hybrid_model, )