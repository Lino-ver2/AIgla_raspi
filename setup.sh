# create enverments
# python -m venv .env
# source ./.env/bin/activate

sudo apt update
sudo apt upgrade

# install package
pip install picamera
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install termcolor
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet


# download model and config
# mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .