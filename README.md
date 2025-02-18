## Installation
创建 Python 3.10+ 的虚拟环境, 安装 requirements.txt 中的依赖:

```bash
conda create -n xxx python=3.10
conda activate xxx
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
安装特征光栅化模块 diff-feature-gaussian-rasterization 和 simple-knn:

```bash
git clone https://github.com/ywyue/FiT3D.git
cd FiT3D/submodules/diff-feature-gaussian-rasterization
python setup.py install
cd ../simple-knn
python setup.py install
```