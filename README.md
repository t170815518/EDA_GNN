# Graph Neural Based End-to-end Data Association Framework for Online Multiple-Object Tracking
A PyTorch implementation combines with Siamese Network and Graph Neural Network for Online Multiple-Object Tracking.

Dataset available at [https://motchallenge.net/]

According paper can be found at [https://arxiv.org/abs/1907.05315]

模型原本用来提取人物图片特征的卷积网络模块修改成了提取数字特征的多层感知机模块。其余的模型训练代码和损失函数都和源代码相同。
位置历史记录考虑了t-1, t-2, ..., t-5.

## How to run
Edit the hyperparameters in `config.yml`.

运行`pip3 install -r requirements.txt`安装用到的库。

先运行`notebook/data_analyze.ipynb`做数据处理。

Use `python main.py` to train a model from scratch. Settings for training is in `config.yml`.  
Use `python tracking.py` to track a test video, meanwhile you need to provide the detected objects & tracking results for the first five frames. Setting for tracking is in `setting/`.

**Use `python test_on_ship.py` to run the test for evaluation metrics.** 这个是我自己写的测试程序，主要输出Multiple Object 
Tracking Accuracy (MOTA) 和ID F1 score (IDF1)这两种指标。

Generator.py定义了训练数据处理流程。

### 修改训练集和测试集
找到下列代码修改文件名即可。

训练集定义代码：
```python
class Generator(object):
    """
    For data loading.
    """
    def __init__(self, entirety=True, is_ship: bool = False):
        """

        :param entirety: bool, if to use all the videos in MOT17
        :param is_ship: bool, if to load ship dataset instead of videos
        """
        self.sequence = []

        if is_ship:
            self.SequenceID = ['2020new_processed.csv', '1010new_processed.csv']
```

测试集定义代码:
```python
# prepare the test Generator
test_generator = TestGenerator('', '1010new_processed.csv', is_ship=True,
                               net_path=TRAINED_MODEL_PATH)
```


## Requirements
 - Python 2.7.12
 - numpy 1.11.0
 - scipy 1.1.0
 - torchvision 0.2.1
 - opencv_python 3.3.0.10
 - easydict 1.7
 - torch 0.4.1
 - Pillow 6.2.0
 - PyYAML 5.1
