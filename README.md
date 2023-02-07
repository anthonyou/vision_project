# vision_project
## Clonining
Clone recursively to include submodules with the command:
```
git clone --recurse-submodules -j8 git@github.com:anthonyou/vision_project.git
```
Or if you have already cloned it without recursive, you can run the following command
```
git submodule update --init --recursive
```

# Requirements

Install main requirements from stable-diffusion
```
cd dependencies/stable-diffusion
conda env create -f environment.yaml
conda activate vision_project
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

Also install the requirements for my_python_utils with:
```
pip install -r my_python_utils/requirements.txt 
conda install -c conda-forge opencv 
```
