# Learning Heuristic Functions for Path Planning Using Neural Networks

This work shows how, with only 30 maps at the start, it is possible to increase the efficiency of a path planner even on maps that have nothing in common with the original ones. The key idea<sup id="a1">[1](#f1)</sup> of the approach is to learn the perfect heuristics using DNNs, and then to use the predicted values as a heuristic function of a path planner.

## Dataset

The initial data consist of 30 [city maps](https://movingai.com/benchmarks/street/index.html) of size 1024×1024 taken from [Moving AI 2D Pathfinding Benchmarks](https://movingai.com/benchmarks/grids.html).
The data are converted from `.map` format to `numpy.array` and resized to the size of 64×64 for convenience. Each of the 30 maps can be splited to 4 big tiles 32×32 and 16 small tiles 16×16. There are 8 possible ways to pose a map tile using flips and 90 degree rotations, so in total there are 960 big tiles and 3840 small tiles. A new generated map for the dataset is put together of four big tiles, and each of them can be replaced by four small tiles with a given probability `p`. A random map cell is selected as a goal pose. If an available cell could not be found in `max_iter` iterations, or there is too much or too little empty space on the map, this map is excluded from consideration. The process of creating maps continues until `dataset_size` maps are generated. Then each of the generated maps is stacked together with their goal pose tensor and saved as a `torch.Tensor` of size (1, 2, 64, 64).

For each of the maps and goal pose tensors of the generated dataset the target perfect heuristic tensor can be evalueted by running [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) from the goal pose in the case of an 8-connected map or [BFS](https://en.wikipedia.org/wiki/Breadth-first_search) in the case of a 4-connected map. The positions which are unreachable from the goal pose have a perfect heuristic value of -1.

![](https://user-images.githubusercontent.com/48711287/162804050-590e0adf-5a29-426d-a4a1-9974eddf0c44.jpg)

## Model

MAE is used as a main loss function. However, it counts only those positions which are reachable from the goal pose:

```
def MAELoss(output, target):
    mask = target > -.5
    return torch.mean(torch.abs(output[mask] - target[mask]))
```

[UnetPlusPlus](https://smp.readthedocs.io/en/latest/models.html#id2) from [smp](https://github.com/qubvel/segmentation_models.pytorch) is used as a network architechture:

```
self.net = UnetPlusPlus(
    encoder_name='efficientnet-b7',
    encoder_weights=None,
    in_channels=2,
    activation='tanh'
).to(device)
```

## Experiments (for 8-connected maps)

Testing with a path planner ([A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)) is performed on a tenth share of the generated dataset, as well as on non-city maps, also taken from Moving AI 2D Pathfinding Benchmarks: game maps, room maps, street maps. A* with 1.5-scaled diagonal heuristic is used as the baseline. The main quality metric of a heuristic function is the ratio of the number of nodes expanded during a run of the algorithm with this heuristic to the number of nodes expanded during a run of the algorithm with diagonal heuristic. Another important metric is a deviation from the length of the shortest path, since both the learned heuristic and the 1.5-scaled diagonal heuristic could be inadmissible.

### Easy and difficult tasks

The difficulty<sup id="a1">[1](#f1)</sup> of the task on an 8-connected map is the ratio of the values of the diagonal and perfect heuristics at the start pose. The more it is, the more difficult the task is. It turns out that learned heuristics work much worser on very easy tasks. However, it is not that bad for two reasons. The first reason is that if a task has difficulty close to 1, this means that even a basic version of A* will up to the task as fast as possible. Another reason is that we can automatically decide whether we want to apply the learned heuristic to a task or not by estimating its difficulty in advance. The ratio of the learned heuristic to the perfect heuristic at the start pose can serve as a difficulty estimator. Given this, we can simply not use the learned heuristic for the tasks with dufficulty estimation less than a given threshold, but apply diagonal heuristic instead.

### Results

| **Map type** | **The number of tasks** | **MRNE\* (Learned)** | **MRNE (Baseline)** | **MDSP**** (Learned), % | **MDSP (Baseline)**, % |
|:------------:|:-----------------------:|:------------------:|:------------------:|:----------------------:|:---------------------:|
|     city     |          1,180          |        0.48        |        0.68        |          0.33          |          1.47         |
|     game     |          1,355          |        0.49        |        0.72        |          0.68          |          1.45         |
|     maze     |          1,895          |        0.41        |        0.69        |          0.68          |          2.02         |
|     room     |          1,656          |        0.51        |        0.77        |          1.22          |          1.10         |

\* - the mean ratio of nodes expanded (the main quality metric)

\*\* - the mean deviation from the shortest path

Visualized paths for the learned heuristics and for the baseline heuristics can be found [here](https://drive.google.com/drive/folders/1qt6a9kownxs1xAnG1xf2W1nJaHGHPb-F?usp=sharing).

![](https://user-images.githubusercontent.com/48711287/162804658-70bdbde6-13fe-43d6-ac20-5ec5ee2bf6d1.jpg)

## How to

### Create a dataset

To create a train dataset run `prepare_dataset.py` script. In this script you should specify the path to the original data (several maps from which the dataset is generated) and the paths to the directories where the results should be saved. You can also specify the size of the maps of the dataset, the size of the dataset, and some other parameters. By default the size of the maps of the dataset is 64. It can be changed, but the process of maps generation is based on the fact that their size must by a multiple of four. You can find an example of the original data in `train maps` directory in `data.zip` [here](https://drive.google.com/drive/folders/1EqTBn5k79eaj0DH5fPFeejEeCwsOSg0i?usp=sharing).

### Train a model

To train a model you can run `learning heuristics.ipynb` notebook. There you should specify the paths to the train dataset (inputs and targets) and the path to the directory where the model parameters should be saved. You can find the train dataset obtained by running `prepare_dataset.py` on the original data from `data.zip` in `train data` directory in the same place. You can also change the size of the maps of the dataset in the notebook if needed.

## Cases

### Case 0: Reproduce the result.

* Download `data.zip` and `model` directory from [here](https://drive.google.com/drive/folders/1EqTBn5k79eaj0DH5fPFeejEeCwsOSg0i?usp=sharing).
* Take `train data` folder from the archive and put it with `model` directory and `learning heuristics.ipynb` in the same directory.
* Run `learning heuristics.ipynb`.
* To reproduce train data run `prepare_dataset.py` on `train maps` from `data.zip`.

### Case 1: Predict heuritics on other maps with goals using the pretrained model.

* The model takes as an input a pytorch tensor of size (N, 2, 64, 64), where N is the size of the test data. An element of the test data is a tensor of size (2, 64, 64) which is a concatenation of two tensors: the map tensor and the goal pose tensor in exactly that order. Both of them have size (1, 64, 64). Obstacles in the map tensor are marked with 0 and free cells are marked with 1. The goal pose tensor contains only one non-zero element - a 1 at the goal cell position.
* How to get model predictions on the test data one can find at `Load a trained model` section in `learning heuristics.ipynb`.
* The predictions is a pytorch tensor of the size (N, 1, 64, 64) and contain the values of predicted heuristics for each element of the test data.
* Before testing the predicted heuristics with a path planner you should slightly modify the code of `run_a_test` function -- remove an outer cycle from it.

### Case 2: Create your own dataset and train a model on it.

* If the size of your maps is distinct from 64 times 64 or the pretrained model is inapplicable to your data, follow these instructions.
* First of all, you need to create your own dataset. For this purpose you can use `prepare_dataset.py` script (see `Create a dataset` in `How to` section of this README). Change `SIZE` and other parameters in the code of the script if needed. As a base for the dataset you can use maps from `train maps` from `data.zip` as well as your own maps in .map format.
* The next step is to train the model on the created dataset. For this purpose you can use `learning heuristics.ipynb` notebook (see `Train a model` in `How to`). Do not forget to change the paths to train data and `SIZE` parameter in the notebook. Uncomment one of the cells in the `Model` section of the notebook and comment the cell in `Load a trained model` section.
* Run the notebook.

## References
<b id="f1">1</b> Takahashi, T.; Sun, H.; Tian, D.; Wang, Y. Learning Heuristic Functions for Mobile Robot Path Planning Using Deep Neural Networks. _ICAPS_ **2019**, _29_, 764-772. [](#a1)
