# FALLEN PEOPLE DETECTION

This work is about recognizing fallen people through a Computer Vision approach. Falling is one of the most common causes of injury in all ages, especially in the elderly, where it is most frequent and severe. For this reason, a method that can detect a fall in real-time can be helpful to avoid more severe damage. Some methods use sensors, wearable devices, or video cameras with particular features such as infrared or depth cameras. However, in this work, we used a vision-based approach to exploit classic video cameras that are more accessible and widespread. A limitation of this method is the lack of generalization to unseen environments. This is due to the error generated during the object detection and, overall, for the unavailability of a large-scale dataset specialized in fallen detection problems with different environments and fallen types. In this work, we try to solve these problems by an object detector trained using a virtual-world dataset in addition to real-world images. We tested our models, and we noted that using synthetic images improves generalization from these results.

## Virtual Dataset
The virtual dataset used in training phase is available folowing this [link](https://zenodo.org/record/6394684#.YkmokChBzDc).

## Training
There are two ways to train your models. With `train_loss.py` the best model obtained is the one with the lower loss. With `train_map.py` the best model obtained is the one with the higher mAP. In our work, we have employed the latest strategy. In training are empoyed config files which are under `conf_files` folder. So, for example, to train your model you could digit:

```bash
python train_map.py conf_files/train_virtual.yaml
```

You could modify this config file as you want (by selecting the folder in which store trainlogs, plots, number of epochs, and so on).

## Inference
To get some inference from some test dataset by using your model, you could digit the following command:

```bash
python inference.py --model checkpoint_train_real_over_virtual.pth --dataset fpds --filename real_over_virtual_finetuned_fpds.txt
```

In `--model` you have to put the model you are using (in this case, for the moment they are not present in this repository). In `--dataset` you have to specify the dataset from which get inference. You can select between: fpds, ufd and elderly. At the end, in `--filename` you have to specify a custom filename in which store the results.

## Dataset creation
To create a `.txt` file from which you can create a Pandas Data Frame, you could perform the following command:

```bash
python script_dataset.py --dt_type train_real --filename train_real.txt
```

The `--dt_type` is used to specify the dataset to build. You can select one among: train_real, train_virtual, test, test_ufd and test_elderly. The filename is defined by the user.

## Create video
You can create a video with bounding boxes and prediction starting from your videos. For example, you can perform this command:

```bash
python create_video.py --video video_1.avi --model checkpoint_train_real_over_virtual.pth --output video_v_th_r.avi --threshold 0.99
```

Here we have `--video` that specify the video's name. In `--model` you have to put the model used to predict the video. In `--output` you specify the output video name and `--threshold` is the threshold used for that particular model.

## Reference
**Learning to detect fallen people in virtual worlds**. *Fabio Carrara, Lorenzo Pasco, Claudio Gennaro, Fabrizio Falchi.* In 2022 Proceedings of the 19th International Conference on Content-based Multimedia Indexing (CBMI). (pp. 126-130). ACM. [[DOI](https://doi.org/10.1145/3549555.3549573)]

```
@inproceedings{carrara2022learning,
  title={Learning to detect fallen people in virtual worlds},
  author={Carrara, Fabio and Pasco, Lorenzo and Gennaro, Claudio and Falchi, Fabrizio},
  booktitle={Proceedings of the 19th International Conference on Content-based Multimedia Indexing},
  pages={126--130},
  year={2022}
}
```
