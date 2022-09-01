# Baseline

|  Model   | Average Loss | Accuracy | Time consumed each iter (s) | 200 Epochs Duration |                Command                 |
|:--------:|:------------:|:--------:|:---------------------------:|---------------------|:--------------------------------------:|
|  vgg16   |    0.0130    |  0.7217  |            1.33             | 2h8m51s             |  `python .\train.py -net vgg16 -gpu`   |
| xception |    0.0067    |  0.7783  |            6.16             | 5h47m19s            | `python .\train.py -net xception -gpu` |


# Adversarial Training

|  Model   | Average Loss | Accuracy | Epsilon | Retrain % | Time consumed each iter (s) | +100 Epochs Duration |                Command                 |
|:--------:|:------------:|:--------:|:-------:|:---------:|:---------------------------:|----------------------|:--------------------------------------:|
