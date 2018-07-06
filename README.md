**GeoMAN**
======
[GeoMAN](https://github.com/yoshall/GeoMAN): Multi-level Attention Networks for Geo-sensory Time Series Prediction.

*An easy implement of GeoMAN using TensorFlow, tested on `CentOS 7` and `Windows Server 2012 R2`.*

## Paper 
[Yuxuan Liang](http://yuxuanliang.com), Songyu Ke, [Junbo Zhang](http://zhangjunbo.org/), Xiuwen Yi, [Yu Zheng](http://urban-computing.com/yuzheng), "[GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction](http://yuxuanliang.com/assets/pdf/ijcai-18.pdf)", *IJCAI*, 2018.

If you find this code and dataset useful for your research, please cite our paper:
```
@article{liang2018geoman,
  title={GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction},
  author={Liang, Yuxuan and Ke, Songyu and Zhang, Junbo and Yi, Xiuwen and Zheng, Yu},
  booktitle={IJCAI},
  year={2018}
}
```

## Dataset [[Click here](http://urban-computing.com/data/Data-1.zip)]
The datasets we use for model training is detailed in Section 4.1 of our paper, which are still under processing for release. You can use the sampled data with 100 instances under "sample_data" folder. Besides, you can also test the code by using the air quality data from 2014/5/1 to 2015/4/30 in our previous research, i.e., KDD-13 and KDD-15. For more datasets, you can visit the [homepage](http://urban-computing.com/) of Urban Computing Lab in JD Group.



## Code Usage
### Preliminary
GeoMAN uses the following dependencies: 
* [TensorFlow](https://github.com/tensorflow/tensorflow#download-and-setup) > 1.2.0
* numpy and scipy.
* CUDA 8.0 or latest version. And **cuDNN** is highly recommended. 

### Code Framework
![](images/framework.png)

### Model Input
The model has the following inputs:
* local_inputs: the input of local spatial attention, shape->[batch_size, n_steps_encoder, n_input_encoder]
* global_inputs: the input of global spatial attention, shape->[batch_size, n_steps_encoder, n_sensors]
* external_inputs: the input of external factors, shape->[batch_size, n_steps_decoder, n_external_input]
* local_attn_states: shape->[batch_size, n_input_encoder, n_steps_encoder]
* global_attn_states: shape->[batch_size, n_sensors, n_input_encoder, n_steps_encoder]
* labels: ground truths, shape->[batch_size, n_steps_decoder, n_output_decoder]

### Guide
The model implement mainly lies in "GeoMAN.py" and "base_model.py" and both of them are well commented. To train or test our model, please follow the presented notebooks.

### License
GeoMAN is released under the MIT License (refer to the LICENSE file for details).
