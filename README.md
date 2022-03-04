# rho-nlr
**Neural Lumigraph Rendering with proof-of-concept controllable illumination in TensorFlow 2.** <br />
<sub>Undergraduate project by Laura Ann Perkins, [New College of Florida](https://ncf.edu/).</sub>

The idea behind `rho-nlr` is to have the appearance model output coefficients in a spherical harmonic basis. This allows for simultaneous novel view synthesis and relighting from a sparse model, given supervision of light direction.

**COMING SOON**: pretrained models and more details.

## Requirements

 - This project requires Python 3.6+ with TensorFlow 2.5.0 (among other dependencies). 
 - For the [SIREN](https://www.vincentsitzmann.com/siren/) models, it builds upon [tf_siren](https://github.com/titu1994/tf_SIREN).

## Usage

You can train a model from scratch with `train.py`.

```
python train.py --dataset_path path/to/data/folder --img_size max_dimension --out_folder path/to/output/folder
```

To evaluate a fitted model, run `test.py`.

```
python test.py --model path/to/fitted/model --dataset_path path/to/data/folder --img_size max_dimension
```

For more options, try running either script with the help flag, e.g.  `python train.py -h`, or edit the [config file](conf/config.py).
