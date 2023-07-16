apt update
apt install -y kakoune tmux
pip install -t . https://r2-public-worker.drysys.workers.dev/nyacomp-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --force-reinstall --upgrade
cp nyacomp/__init__.py nya.pb
cp nyacomp/partition.py .

pip install nvtx

mkdir data

cd /root
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb

dpkg -i cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-12-1-local/cuda-cuobjdump-12-1_12.1.55-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-12-1-local/cuda-nvdisasm-12-1_12.1.55-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-12-1-local/cuda-gdb-12-1_12.1.55-1_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-debian11-11-6-local_11.6.0-510.39.01-1_amd64.deb
dpkg -i cuda-repo-debian11-11-6-local_11.6.0-510.39.01-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-11-6-local/cuda-cuobjdump-11-6_11.6.55-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-11-6-local/cuda-nvdisasm-11-6_11.6.55-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-11-6-local/cuda-gdb-11-6_11.6.55-1_amd64.deb
dpkg -i /var/cuda-repo-debian11-11-6-local/nsight-systems-2021.5.2_2021.5.2.53-1_amd64.deb



COMPRESS=1 python3 nya.py
COMPRESS=1 /usr/local/cuda-12.1/bin/cuda-gdb -ex run -arg python3 nya.py

import torch, from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
model = StableDiffusionAITPipeline.from_pretrained("stabilityai/stable-diffusion-2-base",revision="fp16",torch_dtype=torch.float16,safety_checker=None,local_files_only=True)
#model = StableDiffusionAITPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",revision="fp16",torch_dtype=torch.float16,safety_checker=None,local_files_only=True)
import torch
torch.save(model, "/tmp/model.pth")
import nyacomp
for param in nyacomp.get_pipeline_params(model):
    param.data = torch.empty([])
torch.save(model, "/tmp/deboned_model.pth")
``
