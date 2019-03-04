# Renal Scintigraphy Image Segmentation

Algorithm for renal scintigraphy image segmentation using [opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/).  

This algorithm is an implementation based on the [monography](https://github.com/RobertoDebarba/renal-scintigraphy-image-segmentation/blob/master/docs/original-monograph.pdf) by Monica Marcuzzo *(Instituto de Informática, Universidade Federal do Rio Grande do Sul)*.  

* [Slide show](https://github.com/RobertoDebarba/renal-scintigraphy-image-segmentation/blob/master/docs/presentation.pdf)
* [Paper](https://github.com/RobertoDebarba/renal-scintigraphy-image-segmentation/blob/master/docs/paper.pdf)

<img src="https://github.com/RobertoDebarba/renal-scintigraphy-image-segmentation/blob/master/results/result1.png" width="800" width="auto">
<img src="https://github.com/RobertoDebarba/renal-scintigraphy-image-segmentation/blob/master/results/result2.png" width="800" width="auto">
<img src="https://github.com/RobertoDebarba/renal-scintigraphy-image-segmentation/blob/master/results/result3.png" width="800" width="auto">

## Goals

* Segmentation of the kidney for extraction of features.
* One of the most important characteristics of the kidney are the dimensions and formats, therefore it is necessary that the segmentation is accurate.

## Conclusion

Applying the solution proposed by Marcuzzo's Dissertation in a sample of the public base of images, considering the first 25 images without being selective, we can observe that the method proposed by the author not always works, since only 20% of the cases had the expected result, it should also be noted that there were several images that were considered of a bad quality, because do not present the second kidney.  
We conclude that the method has a good result only in controlled environments.

## How to run

### Prerequisites

* Python
* Pip

### Install dependencies

1. `pip install -r requirements.txt`

### Run

1. `python main.py`

## Authors

* [Roberto Luiz Debarba](https://github.com/RobertoDebarba)
* [Matheus Adriano Pereira](https://github.com/matheusPereiraKrumm)

## License

The codebase is licensed under [GPL v3.0](http://www.gnu.org/licenses/gpl-3.0.html).
