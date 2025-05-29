![alt text](video/ionosphere.png)




# Data Science Lab: Deep Learning for Ionosphere Modeling

Welcome to the code repository for the ETHZ Data Science Lab 2025! The codebase comprises code to train and evaluate neural networks for Ionospheric Modeling.

## Contents

- [About](#about)
- [Installation](#installation)
- [Getting the data ready](#installation)
- [Training](#usage)
- [Inferences](#screenshots)
- [Evaluation](#code-example)
- [Other](#other)
- [License](#license)


## About
The codebase allows to:
- Train neural networks from scratch in order to build global STEC maps for a short period of time (typically 1 day).
- Pretrain neural networks on subsampled data corresponding to a longer period of time (January 2022 to June 2024 in our setup)
- Subsample STEC daily datasets in order to generate the pretraining data.
- Fine tune pretrained neural networks in order to build global STEC maps for a short period of time.
- Use a trained model to run inferences on datasets comprising a short period of time.
- Run data analysis and visualizations on inferences results.
- Make videos that show the predicted STEC of our models overlaid to an Earth map as a function of time.

The code is designed to run in the ETHZ Euler cluster. The Python environment has been tested on Euler. Pretrainings on Euler take 2-6 hours depending on the amount of data. While the testing coverage for the code outside has been less extense, the following instructions can also be followed outside Euler and everything should work. Training and inference times will depend on the user's available resources.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.