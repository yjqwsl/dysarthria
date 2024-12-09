import os
from pathlib import Path
import string
import tempfile
import subprocess
import sys

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel, EncDecCTCModel, EncDecRNNTModel
from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel, HifiGanModel

###ACOUSTIC SPEECH RECOGNITION
# Load ASR model trained from the hybrid model
model_path = Path("./checkpoints/finetuned_asr_model.nemo")
asr_model = EncDecCTCModelBPE.restore_from(str(model_path))
'''to restore your weights, you can rebuild the model using the config'''
#first_asr_model_continued.restore_from(<checkpoint_path>).

data_dir = './datasets/lrdwws/train/'
#/home/jinqiy/Dev/Dysarthria/NeMo/datasets/lrdwws/train/Control/wav/test/CF0005/CF0005_0009.wav #control
#/home/jinqiy/Dev/Dysarthria/NeMo/datasets/lrdwws/train/Uncontrol/wav/test/DM0012/DM0012_0012.wav #dysarthric
audio = [#os.path.join(data_dir, 'Control/wav/test/CF0005/CF0005_0009.wav'),
        os.path.join(data_dir, 'Uncontrol/wav/test/DM0012/DM0012_0012.wav'),]
transcribed_text=str(asr_model.transcribe(audio=audio, batch_size=4))
print(transcribed_text)


###TEXT TO SPEECH
# Load spectrogram generator
spec_generator = FastPitchModel.from_pretrained(model_name="tts_zh_fastpitch_sfspeech")

# Load Vocoder
from nemo.collections.tts.models import HifiGanModel
model = HifiGanModel.from_pretrained(model_name="tts_zh_hifigan_sfspeech")

# Generate audio
import soundfile as sf
import torch
with torch.no_grad():
    parsed = spec_generator.parse(transcribed_text) #parsed = spec_generator.parse("这些新一代的CPU不只效能惊人。")#王明是中国人王明是学生
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = model.convert_spectrogram_to_audio(spec=spectrogram)
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()

# Save the audio to disk in a file called speech.wav
generated_audio='speech.wav'
sf.write(generated_audio, audio.T, 22050, format='WAV')
