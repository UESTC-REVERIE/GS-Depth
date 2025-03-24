## ⚙️ Setup
使用 PyTorch 2.1.2, CUDA 11.8 和 Ubuntu 20.04 \
推荐使用 conda 创建 Python 3.10+ 的虚拟环境, 首先安装 requirements.txt 中的依赖:

```shell
conda create -n xxx python=3.10
conda activate xxx
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
并安装特征光栅化模块 diff-feature-gaussian-rasterization  和 simple-knn (from [FiT3D](https://github.com/ywyue/FiT3D.git)):

```shell
git clone https://github.com/ywyue/FiT3D.git
cd FiT3D/submodules/diff-feature-gaussian-rasterization
python setup.py install
cd ../simple-knn
python setup.py install
```

## 💾 KITTI training data

下载 [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) :
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
解压：
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
解压后的数据集文件结构应该如 [kitti_overview](./kitti_overview.txt) 所示。

**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

## ⏳ Training
### GPUs

代码仅支持单卡训练，使用 `CUDA_VISIBLE_DEVICES` 以指定显卡：
```shell
CUDA_VISIBLE_DEVICES=2 python train.py --model_name ./your_model_name
```

我们使用一张 NVIDIA Titan RTX 显卡进行训练，预计的消耗如下：
| Training modality | Approximate GPU memory  | Approximate training time   |
|-------------------|-------------------------|-----------------------------|
| Mono              | -                     | - hours                    |

### 💽 pretrained model
| model | params | link|
|-------|------|------|
|[HRNet-W18-C](https://github.com/HRNet/HRNet-Image-Classification.git) | 21.3M | [LINK](https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar)|

## 📊 KITTI evaluation
| `--eval_split`        | Test set size | For models trained with... | Description  |
|-----------------------|---------------|----------------------------|--------------|
| **`eigen`**           | 697           | `--split eigen_zhou` | The standard Eigen test files |

评估模型：
```shell
python evaluate_depth_gs_bestmodel.py \
    --data_path ~/dataset/KITTI_dataset/raw_data \
    --load_weights_path ./xxx.pth \
    --eval_split eigen \
    --eval_mono \
    --use_gs \
```
可以使用 `--eval_output_dir ./your_eval_output_dir` 和 `--save_pred_disps` 选项来保存评估中预测的深度图进行分析。

### 🎉 eval results
|abs_rel |sq_rel |rmse | rmse_log |a1 |a2 |a3 | Input size  | `.pth` filesize | Eigen disparities|
|-------|---------|-------|-------|----|----|----|----|-----|------|
|   0.101  |   0.707  |   4.448  |   0.179  |   0.894  |   0.965 |   0.983 | 640 x 192   | - MB          |     [Download 🔗](-)    |
