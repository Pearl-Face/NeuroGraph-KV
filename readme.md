项目文件
我已经 clone好了

复原moba环境方法：
注释掉
# - moba==1.0.0
# - torch==2.5.1+cu124
# - torchaudio==2.5.1+cu124
# - torchvision==0.20.1+cu124
创建环境
conda env create -f moba_environment.yml -n moba_env_fixed
下载pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124