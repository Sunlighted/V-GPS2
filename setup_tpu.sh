# #!/bin/bash
# set -e  # Exit on error

# echo "=== Installing Miniconda ==="
# cd ~
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
# rm Miniconda3-latest-Linux-x86_64.sh

# # Initialize conda
# eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
# conda init bash
# source ~/.bashrc

# echo "=== Accepting Conda Terms of Service ==="
# conda config --set channel_priority strict
# conda tos accept

# echo "=== Creating conda environment ==="
# conda create -n vgps python=3.10 -y

# echo "=== Activating conda environment ==="
# source $HOME/miniconda3/bin/activate vgps

# echo "=== Installing git (if not present) ==="
# sudo apt-get update
# sudo apt-get install -y git

# echo "=== Installing Vulkan SDK ==="
# sudo apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools

# echo "=== Cloning V-GPS repo with submodules ==="
# cd ~
# # git clone https://github.com/nakamotoo/V-GPS --recurse-submodules
# # cd V-GPS

echo "=== Installing packages ==="
conda install -c conda-forge libgl
pip install -e .
pip install -e octo
pip install -e SimplerEnv
pip install -e SimplerEnv/ManiSkill2_real2sim
pip install -r requirements.txt
pip install sentencepiece

echo "=== Installing JAX for TPU ==="
pip install --upgrade "jax[tpu]==0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Verifying JAX installation ==="
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

echo "=== Setup complete! ==="
echo "To activate environment: conda activate vgps"
echo "V-GPS repo location: ~/V-GPS"