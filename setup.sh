# create enverments
python -m venv .env
source ./.env/bin/activate

# install package
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet


# download model and config
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .