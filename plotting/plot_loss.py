"""Script used to plot the validation losses of different models, in a single plot."""

from matplotlib import pyplot as plt
from pprint import pp

# select whether you want to plot pretrained or finetuned models
# MODEL_TYPE = "pretrained"
MODEL_TYPE = "finetuned"
assert (MODEL_TYPE == "finetuned") or (MODEL_TYPE == "pretrained"), "Unknown Model Type"

dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"


# pretrained models

pretrained_model_dict = {  # dictionary of pretrained model locations
    "model_2025-05-19-17-11-14/": "6 MSE",
    "model_2025-04-30-23-22-21/": "4+2 MSE",
    "model_2025-05-04-12-10-18/": "4+2 MAE",
    "model_2025-05-19-15-24-01/": "hourly solar indices, 4+2, MAE",
    "model_2025-05-10-17-39-31/": "daily solar indices, 4+2, MAE",
}

# flip dictionary
pretrained_model_dict = {value: key for key, value in pretrained_model_dict.items()}


if MODEL_TYPE == "pretrained":
    models_path = dslab_path + "pretrained_models/"
    outpath = "outputs/val_loss_pt_single.png"

    # select which models you want to plot
    model_labels = ["6 MSE", "4+2 MSE", "hourly solar indices, 4+2, MAE", "daily solar indices, 4+2, MAE"]

    models = [pretrained_model_dict[model] for model in model_labels]


# fine tuned models

finetuned_model_dict = {  # dictionary of finetuned model locations
    "model_2025-05-19-16-24-04/": "From scratch MSE",
    "model_2025-05-20-09-58-46/": "6 MSE",
    "model_2025-05-19-15-22-50/": "4+2 MSE",
    "model_2025-05-04-21-40-49/": "4+2 MAE",
    "model_2025-05-11-16-53-38/": "daily solar indices, 4+2, MAE",  # same as "Day 1, finetuned"
    "model_2025-05-20-10-29-44/": "hourly solar indices, 4+2, MAE",
    "model_2025-05-20-11-23-28/": "From scratch with daily solar indices",
    "model_2025-05-20-11-37-14/": "From scratch with hourly solar indices",  # same as "Day 1, from scratch"
    "model_2025-05-21-11-00-30/": "Day 8, finetuned",
    "model_2025-05-21-12-39-41/": "Day 8, from scratch",
    "model_2025-05-21-10-45-05/": "Day 16, finetuned",
    "model_2025-05-21-12-38-44/": "Day 16, from scratch",
    "model_2025-05-21-10-45-41/": "Day 32, finetuned",
    "model_2025-05-21-11-40-35/": "Day 32, from scratch",
    "model_2025-05-21-10-46-14/": "Day 64, finetuned",
    "model_2025-05-21-11-41-03/": "Day 64, from scratch",
}

# flip dictionary
finetuned_model_dict = {value: key for key, value in finetuned_model_dict.items()}


if MODEL_TYPE == "finetuned":
    models_path = dslab_path + "models/"
    outpath = "outputs/val_loss_ft_single.png"

    model_labels = ["From scratch MSE", "6 MSE", "4+2 MSE"] + [
        "From scratch with daily solar indices",
        "daily solar indices, 4+2, MAE",
    ]
    # model_labels = ["daily solar indices, 4+2, MAE", "Day 8, finetuned", "Day 16, finetuned", "Day 32, finetuned", "Day 64, finetuned"]

    models = [finetuned_model_dict[model] for model in model_labels]


fig, ax = plt.subplots(figsize=(9, 6))

test_losses = {}


# if you want you can specify which colors to use in what order

# default colors
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# custom colors
# colors = ["tab:blue", "tab:green", "tab:orange"]

j = 0
for model in models:
    log_path = models_path + model + "logs.log"
    mse_val_losses = []
    mae_val_losses = []
    with open(log_path, "r") as file:
        for line in file:
            if "MSE Val" in line:
                mse_val_losses.append(float(line.split(" ")[-1]))

            if "MAE Val" in line:
                mae_val_losses.append(float(line.split(" ")[-1]))

            if "Evaluation" in line:
                test_losses[model] = float(line.split(" ")[-1])

    """MSE validation loss -> can compare all models"""
    # doesn't include zero-shot loss -> note that zero-shot error is much bigger -> need to limit y axis
    # ax.plot(range(1, len(mse_val_losses)), mse_val_losses[1:], label = model, c=colors[j])

    # includes zero-shot loss
    # ax.plot(range(0, len(mse_val_losses)), mse_val_losses, label = model, c=colors[j])

    """MAE validation loss -> some very old models don't have this"""
    # doesn't include zero-shot loss -> note that zero-shot error is much bigger -> need to limit y axis
    # ax.plot(range(1, len(mae_val_losses)), mae_val_losses[1:], label = model, c=colors[j])

    # includes zero-shot loss
    ax.plot(range(0, len(mae_val_losses)), mae_val_losses, label=model, zorder=3, c=colors[j])
    j += 1


# if you want to change the labels you can do it here
# model_labels = ["FCN with no Pretraining, \nMSE and no Solar Indices", "FCN with Pretraining, \nMSE and no Solar Indices", "TwoStage with Pretraining, \nMSE and no Solar Indices"] + ["FCN with no Pretraining, \nMAE and Daily Solar Indices", "TwoStage with Pretraining, \nMAE and Daily Solar Indices", "Final Val Loss of FCN Model \nwith no Pretraining but \nDaily Solar Indices"]


pp(test_losses)
ax.set_ylim(top=10)
# ax.set_ylim([0,15])
# ax.axhline(3.16, c="darkgrey", zorder=1) # for final val loss when training from scratch
ax.tick_params(axis="both", which="major", labelsize=14)
# fig.legend(labels=model_labels, loc="center right", fontsize = 14, bbox_to_anchor=(0.93, 0.5))
ax.legend(labels=model_labels, loc="upper right", fontsize=14)
# fig.subplots_adjust(right=0.7)
ax.set_xlabel("Epoch", fontsize=18, labelpad=8)
ax.set_ylabel("MAE Validation Loss", fontsize=18, labelpad=8)
ax.set_title("Validation Loss During Training", fontsize=24, pad=14)


fig.savefig(outpath, dpi=300)
