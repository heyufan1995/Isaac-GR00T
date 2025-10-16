
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
apt update
apt install -y libgl1
pip install simplejpeg
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e .

