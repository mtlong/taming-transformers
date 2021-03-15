
# This script is used to prepare the data and pre-trained models needed for the reconstruction test

# download a VQGAN with a codebook with 1024 entries
mkdir -p logs/vqgan_imagenet_f16_1024/checkpoints
mkdir -p logs/vqgan_imagenet_f16_1024/configs
wget 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1' -O 'logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt' 
wget 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1' -O 'logs/vqgan_imagenet_f16_1024/configs/model.yaml' 

# download a VQGAN with a larger codebook (16384 entries)
mkdir -p logs/vqgan_imagenet_f16_16384/checkpoints
mkdir -p logs/vqgan_imagenet_f16_16384/configs
wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt' 
wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/configs/model.yaml'

# Prepare for DALL-E
mkdir -p logs/DALLE/checkpoints
wget 'https://cdn.openai.com/dall-e/encoder.pkl' -O 'logs/DALLE/checkpoints/encoder.pkl'
wget 'https://cdn.openai.com/dall-e/decoder.pkl' -O 'logs/DALLE/checkpoints/decoder.pkl'

pip install omegaconf==2.0.0
pip install moviepy

pip install git+https://github.com/openai/DALL-E.git &> /dev/null