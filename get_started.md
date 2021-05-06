# 安装和使用教程

## 环境需求

```yaml
python>=3.6
numpy>=1.16.0
torch>=1.0  # effdet需要torch>=1.5，如果不使用effdet，在network/__init__.py下将其注释掉
tensorboardX>=1.6
utils-misc>=0.0.5
mscv>=0.0.3
matplotlib>=3.1.1
opencv-python>=4.2.0.34  # opencv>=4.4版本需要编译，耗时较长，建议安装4.2版本
opencv-python-headless>=4.2.0.34
albumentations>=0.5.1  # 需要opencv>=4.2
scikit-image>=0.17.2
easydict>=1.9
timm==0.1.30  # timm >= 0.2.0 不兼容 
typing_extensions==3.7.2
tqdm>=4.49.0
PyYAML>=5.3.1
Cython>=0.29.16
pycocotools>=2.0  # 需要Cython
omegaconf>=2.0.0  # effdet依赖

```

## 训练和验证模型voc数据集

### 准备voc数据集

1. 下载voc数据集，这里提供一个VOC0712的网盘下载链接：<https://pan.baidu.com/s/1AYao-vYtHbTRN-gQajfHCw>，密码7yyp。

2. 在项目目录下新建`datasets`目录：

   ```bash
   mkdir datasets
   ```

3. 将voc数据集的`VOC2007`或者`VOC2017`目录移动`datasets/voc`目录。（推荐使用软链接）

   ```bash
   ln -s <VOC的下载路径>/VOCdevkit/VOC2017 datasets/voc
   ```

4. 数据准备好后，数据的目录结构看起来应该是这样的：

   ```yml
   code
       └── datasets
             ├── voc           
             │    ├── Annotations
             │    ├── JPEGImages
             │    └── ImageSets/Main
             │            ├── train.txt
             │            └── test.txt
             └── <其他数据集>
   ```

### 使用预训练模型验证


1. 新建`pretrained`文件夹：

   ```bash
   mkdir pretrained
   ```

2. 以Faster-RCNN为例，下载[[预训练模型]](https://github.com/misads/detection_template#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)，并将其放在`pretrained`目录下：

   ```yml
   code
       └── pretrained
             └── 0_voc_FasterRCNN.pt
   ```

3. 运行以下命令来验证模型的`mAP`指标：

**Faster RCNN**

   ```bash
   python3 eval.py --model Faster_RCNN --dataset voc --load pretrained/0_voc_FasterRCNN.pt -b1
   ```

**YOLOv2**

   ```bash
   python3 eval.py --model Yolo2 --load pretrained/0_voc_Yolo2.pt -b24 
   ```


4. 如果需要使用`Tensorboard`可视化预测结果，可以在上面的命令最后加上`--vis`参数。然后运行`tensorboard --logdir results/cache`查看检测的可视化结果。

   

### 训练模型

#### Faster RCNN

```bash
python3 train.py --tag frcnn_voc --model Faster_RCNN -b1 --optimizer sgd --val_freq 1 --save_freq 1 --lr 0.001
```

#### YOLOv2

```bash
python3 train.py --tag yolo2_voc --model Yolo2  -b24 --val_freq 5 --save_freq 5 --optimizer sgd --lr 0.00005 --scheduler 10x --weights pretrained/darknet19_448.conv.23 --scale 544
```

`darknet19_448.conv.23`是Yolo2在`ImageNet`上的预训练模型，可以在yolo官网下载。[[下载地址]](https://pjreddie.com/media/files/darknet19_448.conv.23)。



### 参数说明

| 作用                        | 参数                       | 示例                         | 说明                                                         |
| --------------------------- | -------------------------- | ---------------------------- | ------------------------------------------------------------ |
| 指定训练标签                | `--tag`                    | `--tag yolo2_voc`            | 日志会保存在`logs/标签`目录下，模型会保存在`checkpoints/标签`目录下。 |
| 选择模型                    | `--model`                  | `--model Yolo2`              | **必须明确给定**。                                           |
| 选择backbone                | `--backbone`               | `--backbone res50`           | 目前仅Faster RCNN支持选择backbone。                          |
| 选择数据集                  | `--dataset`                | `--dataset voc`              | 支持voc、coco等数据集，也支持`dataloader/custom`中自定义的数据集。 |
| 指定batch_size              | `-b`                       | `-b24`                       | 设置batch_size为24。                                         |
| 指定学习率                  | `--lr`                     | `--lr 0.001`                 | 设置初始学习率为0.001。                                      |
| 设置训练总代数和周期        | `--scheduler`              | `--checuler 10x`             | `1x`总共训练`12`个epoch，`10x`总共训练`120`个epoch。         |
| 指定优化器                  | `--optimizer`             | `--optimizer sgd`            | 指定优化器为sgd。                                            |
| 指定dataloader的进程数      | `-w`                       | `-w4`                        | 如果需要用pdb调试，须设为`-w0`。                             |
| 加载之前的模型/恢复训练     | `--load`       | `--load pretrained/yolo2.pt` | `--resume`配合`--load`使用，会恢复上次训练的`epoch`和优化器。 |
| 指定每几代验证/保存一次模型 | `--val_freq`、`--save_freq` | `--val_freq 5`               | 每5代验证一次模型。    |
| 调试模式                    | `--debug`                  | `--debug`                    | 调试模式下只会训练几个batch就会开始验证。      |

