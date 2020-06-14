# 3D-GAE

This is the [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) implementation of the volumetric [generalized autoencoder (GAE)](http://openaccess.thecvf.com/content_cvpr_workshops_2014/W15/html/Wang_Generalized_Autoencoder_A_2014_CVPR_paper.html) described in the paper ["Generalized Autoencoder for Volumetric Shape Generation"](http://openaccess.thecvf.com/content_CVPRW_2020/html/w17/Guan_Generalized_Autoencoder_for_Volumetric_Shape_Generation_CVPRW_2020_paper.html).

## Preparing the Data

Some experimental shapes from the [COSEG](http://www.yunhaiwang.org/public_html/ssl/ssd.htm) and [auto-aligned ModelNet40](https://github.com/lmb-freiburg/orion#modelnet40-aligned-objects) datasets are saved in the `datasets` folder. The model consumes volumetric shapes compressed in the [TAR](https://www.gnu.org/software/tar/) file format. For details about the structure and preparation of the TAR files, please refer to [voxnet](https://github.com/dimatura/voxnet).

## Training

```bash
python train.py
```

## Testing

```bash
python test.py
```
