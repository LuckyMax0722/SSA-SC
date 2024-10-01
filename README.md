# SSA-SC

## Semantic Segmentation-assisted Scene Completion for LiDAR Point Clouds

This repository implement the inference script for SSA-SC. For original code, please refer to [SSA-SC](https://github.com/jokester-zzz/SSA-SC)


## Getting Start
Clone the repository:
```
git clone https://github.com/LuckyMax0722/SSA-SC.git
```

### Inference

You can use the `networks/test_DSEC.py` file to infer the sematic voxel output. 

Also, you need to set the path to the pre-trained model and the dataset root directory.


```
$ cd <root dir of this repo>
$ python networks/test_DSEC.py
```

### Pretrained Model

You can download the models with the scores below from this [Google drive link](https://drive.google.com/file/d/1pzvtuk3A9V_M-8a0rTAh5_E-ZjqTpPWN/view?usp=sharing), 

| Model  | Segmentation | Completion |
|--|--|--|
| SSA-SC | 23.51 | 58.79 | 
