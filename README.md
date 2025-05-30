![alt text](video/ionosphere.png)


# Data Science Lab: Deep Learning for Ionosphere Modeling

Welcome to the code repository for the ETHZ Data Science Lab 2025, challenge 16! This codebase comprises of code to train and evaluate neural networks for Ionospheric Modeling.

## Contents

- [About](#about)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Inferences](#inferences)
- [Evaluation](#evaluation)
- [Video](#video)
- [License](#license)


## About
The codebase allows one to:
- Train neural networks from scratch in order to build global STEC maps for a short period of time (typically 1 day).
- Pretrain neural networks on subsampled data corresponding to a longer period of time (January 2022 to June 2024 in our setup)
- Subsample STEC daily datasets in order to generate the pretraining data.
- Fine tune pretrained neural networks in order to build global STEC maps for a short period of time.
- Use a trained model to run inferences on datasets comprising a short period of time.
- Run data analysis and visualizations on inference results.
- Make videos that show the predicted STEC of our models overlaid to an Earth map as a function of time.

The code is designed to run in the ETHZ Euler cluster. The Python environment has been tested on Euler. Pretrainings on Euler take 2-6 hours depending on the amount of data. While the testing coverage for the code outside of Euler has been less extensive, the following instructions can also be followed outside Euler and everything should work. Training and inference times will depend on the user's available resources.

The main dataset used for this project has not yet been published. In order to illustrate how the training code runs end-to-end, the folder `example_code` contains a very small subset of the dataset. Check the section [Example training code](#example-training-code) to find out how to run the example code without needing to have access to the full dataset. 

## Installation
- Clone the repository with `git clone git@github.com:eboesch/IonosphereModel.git`
- If working on the Euler cluster, run `module load stack/2024-06 python/3.12.8`. If running on your local machine, this step can be skipped.
- Create a virtual environment with `python3 -m venv ./venv` and activate it with `source venv/bin/activate`.
- Install the required Python packages with `pip install -r requirements.txt`.

## Datasets

### Dataset description
The dataset is private and can be found in the cluster location `/cluster/work/igp_psr/arrueegg/GNSS_STEC_DB/`. For each year and day of the year, an h5 file contains the STEC measurements for that day. All the code related to dataset manipulation can be found under the folder `datasets` in this repository.

### Dataset splits
We divided the dataset into training, validation and test stations. Training stations are used to optimize the training loss. Validation stations are used for model selection. Test stations are used to estimate our model's generalization error. 

The lists of stations within each split are contained in the files `train.list`, `val.list` and `test.list`. We use the same splits as in [our project supervisor's repository](https://github.com/arrueegg/STEC_pretrained/tree/main). The script `dataset/split.py` can also be used to generate new random splits.

### Data Reorganization
The dataset format is suitable for training models from scratch or for fine-tuning pretrained models on few days worth of data in a short period of time. However, the structure of the dataset is not suitable for pretraining models, given that the dataset is very large and cannot fit into memory. Furthermore train, val and test data are mixed within the files for each day.

We reorganized the data by separating it into train, val and test files and subsampling it. The subsampled dataset fits in memory, which is a key factor for speeding up the training. Furthermore, the reorganized dataset can be manipulated more easily since the training, validation and test data are separated.

On the Euler cluster, the script to run data reorganization can be run with
```bash
 srun --ntasks=1 --cpus-per-task=6 --mem-per-cpu=4096 -t 600 -o file.out -e file.err python dataset/reorganize_data.py &
```
Outside the Euler cluster, provided that the user has a copy of the dataset in their machine, it could also be run with
```bash
python dataset/reorganize_data.py
```
The parameters for the data reorganization (time period for which to subsample and reorganize the data, subsampling ratio, etc.) are contained in the config file `config/reorganize_data_config.yaml`.

### Solar Indices

We further augmented our dataset with solar index data. We used both daily and hourly data of the Kp index, the sunspot number R, the Dst index and the F10.7 index. The data can be directly downloaded from [NASA/GSFC's Space Physics Data Facility's OMNIWeb Service](https://omniweb.gsfc.nasa.gov/form/dx1.html). For the example code, the relevant solar index data is already provided in the folder `solar_indices_data`.



## Training

Models can be trained on the Euler cluster with
```bash
srun --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8192 -G 1 -t 600 -o file.out -e file.err python run_training.py &
```
If training on a local machine, simply run
```bash
python run_training.py
```
Every time a model is trained, a folder is created to store the model weights, training log, and config file used to launch the training. The timestamp of the training start is used as a model id and determines the folder name. 

### Training config

The training script is parametrized by a config file, which is specified in the `config_path` variable at the beginning of the file. We now describe some relevant variables in the training config files:
- `num_workers` is the number of cpus that will be used in the DataLoader. We recommend 8-32. In the `srun` arguments, set `--cpus-per-task` to the same number as `num_workers`. If the number of cpus is large, the `--mem-per-cpu` argument can be set to a lower number to keep the total shared memory constant.
- `use_reorganized_data`: Whether to use the original dataset for training (`false`) or the reorganized dataset (`true`). Set to `true` for pretraining and to `false` for fine-tuning.
- `pretrained_model_path`: Path to a pretrained model. If specified, the weights of such model will be loaded for training and the architecture related config parameters will be ignored. If set to `null`, a new model is initialized with random weights using the architecture related config parameters. For pretraining or training a model from scratch, we set `pretrained_model_path` to `null`. For fine-tuning, `pretrained_model_path` needs to point to the desired pretrained model folder.
- `optional_features`: Determines which optional features are incorporated in model training. All available features are listed in the current config files.. If a feature should be used, uncomment it, and vice versa. When training from scratch, we recommend *not* using doy and year as optional features. Note that when using a TwoStage architecture all optional features are only provided to the model in the second stage.

Check the config files under `config/` for additional variable descriptions.

### Available configs: Pretraining, fine-tuning and mock trainings

We provide multiple config files for different functionalities:

- `config/pretraining_config.yaml` can be used to pretrain models on data corresponding to a large period of time (see the section [Data Reorganization](#data-reorganization)).
- `config/training_config.yaml` can be used to fine-tune models or to train models initialized with random weights on data corresponding to a short time period. Whether a pretrained model is used or not is determined by the `pretrained_model_path` in the config file, see the section above.
- `config/training_mock_config.yaml` can be used to launch a mock training that shows the training code running end-to-end. See the subsection [Example training code](#example-training-code).


### Example training code

Make sure that the `config_path` variable at the beginning of the `run_training.py` file points to `config/training_mock_config.yaml` (should be set as default) and run the training script as described at the beginning of this section. This launches a training with randomly initialized weights on a subset of 20000 random rows of the dataset for day 183 in year 2024. The training uses daily solar indices as additional input features for the model. All the necessary data for the example training is contained in the `example_code` folder. The generated model folder for the training will be stored also under `example_code`.

The purpose of this example code is to illustrate how models are trained and how the code runs end-to-end without the need of sharing the full dataset. However, due to the very small dataset size considered, models trained on this subset are not expected to perform well.

## Inferences
The script `run_inferences.py` is used to run inferences for a given model. It is parametrized by the `config/inferences_config.yaml` file. In order to run inferences for a given model, specify the `model_path` variable within the inferences config file, and run (in the Euler cluster)
```bash
srun --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8192 -G 1 -t 600 -o file.out -e file.err python run_inferences.py &
```
If running on a local machine, simply run
```bash
python run_inferences.py
```
Inferences can be run for the training, validation or test split for the data corresponding to a short period of time (typically one day). They can also be run on an additional satellite altimetry test dataset (check the variable `evaluation_data` within `config/inferences_config.yaml`).

The inferences script generates an inferences subfolder within the model folder of the model used to run inferences. The Mean Absolute Error and Mean Squared Error are saved in a csv file `metrics.csv`. The model predictions, labels, and features for each entry in the dataset are stored in `inferences.csv`.

## Evaluation
`plotting/plot_loss.py` retrieves the validation losses after each training epoch from the model logs for several models and plots them. The models that should be plotted can be specified. Both MAE and MSE validation losses can be plotted.

`plotting/plot_loss_side_by_side.py` does the same thing, but creates two plots side by side, for easier comparison.

`plotting/plot_extrapolation.py` can be used to visualize the results of our extrapolation experiments.



`analysis.ipynb` takes in an inference file and calcules the MAE. It then breaks down the MAE by region, local time and elevation angle.


![image](https://github.com/user-attachments/assets/dfeb281d-dcd0-401d-b592-e227ac6bcfdf)

## Video

`video/global_inference.py` calculates model inferences for every part of the world for an entire day. `video/make_video.ipynb` creates visualizations of those inferences and stiches them together to make a video.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
