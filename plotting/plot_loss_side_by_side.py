"""Script used to plot the validation losses of different models, in two plots side by side."""
from matplotlib import pyplot as plt
from pprint import pp

# select whether you want to plot pretrained or finetuned models
# MODEL_TYPE = "pretrained"
MODEL_TYPE = "finetuned"
assert (MODEL_TYPE == "finetuned") or (MODEL_TYPE == "pretrained"), "Unknown Model Type"

dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"


# pretrained models

pretrained_model_dict = { # dictionary of pretrained model locations
    "model_2025-04-25-19-09-00/": "Hector 1", 
    "model_2025-04-25-15-03-51/": "Hector 2", 
    "model_2025-04-28-09-08-52/": "Hector 3", 
    "model_2025-04-30-14-05-28/": "4+1",
    "model_2025-04-30-21-02-28/": "6 old", 
    "model_2025-04-30-23-22-21/": "4+2", 
    "model_2025-05-19-17-11-14/": "6", 
    # "model_2025-04-30-23-22-21/": "4+2", 
    "model_2025-05-01-16-14-32/": "6+3", 
    "model_2025-05-01-23-02-56/": "4+2, with smaller batch size",
    "model_2025-05-04-12-10-18/": "4+2, with MAE training loss",    
    "model_2025-05-10-21-34-55/": "hourly solar indices, 4+2, MAE",
    "model_2025-05-10-17-39-31/": "daily solar indices, 4+2, MAE",    
}

# flip dictionary
pretrained_model_dict = {value: key for key, value in pretrained_model_dict.items()}



if MODEL_TYPE == "pretrained":
    models_path = dslab_path + "pretrained_models/"
    outpath = 'outputs/val_loss_pt_double.png'

    # select which models you want to plot
    model_labels1 = ["6", "4+2"]
    model_labels2 = ["hourly solar indices, 4+2, MAE", "daily solar indices, 4+2, MAE"]

    models1 = [pretrained_model_dict[model] for model in model_labels1]
    models2 = [pretrained_model_dict[model] for model in model_labels2]



# fine tuned models

finetuned_model_dict = { # dictionary of finetuned model locations
    "model_2025-04-28-12-21-37/": "Hector 1",
    "model_2025-04-28-12-37-02/": "Hector 2",
    "model_2025-04-28-12-50-35/": "Hector 3",
    "model_2025-04-28-13-02-05/": "Hector 4",
    "model_2025-04-28-13-13-17/": "Hector 5",
    "model_2025-05-01-13-51-30/": "4+1",
    "model_2025-05-01-14-03-22/": "6",
    "model_2025-05-01-14-35-02/": "4+2",
    "model_2025-05-01-14-48-57/": "4+2 0.5 FT batch size", 
    "model_2025-05-01-15-09-01/": "4+2 0.5 FT LR", 
    "model_2025-05-01-15-55-55/": "from scratch, 4", 
    "model_2025-05-02-09-41-24/": "6+3", 
    "model_2025-05-02-09-50-42/": "4+2 0.5 batch size", 
    "model_2025-05-02-10-03-18/": "6+3 0.5 FT LR", 
    "model_2025-05-02-10-23-23/": "4+2 0.1 FT LR", 
    "model_2025-05-02-10-36-41/": "4+2, day 7", 
    "model_2025-05-02-10-52-15/": "4+2, day 14", 
    "model_2025-05-02-11-14-16/": "4+2, day 28",
    "model_2025-05-04-21-40-49/": "4+2, MAE bad formatting",
    "model_2025-05-05-11-15-10/": "4+2, MAE 0.5 FT LR",
    "model_2025-05-05-11-31-39/": "4+2, MAE",
    "model_2025-05-05-13-23-28/": "from scratch, 4, MAE, with doy+year", 
    "model_2025-05-05-13-37-53/": "from scratch, 4, MAE", 
    "model_2025-05-11-15-32-59/": "hourly solar indices, 4+2, MAE",
    "model_2025-05-11-16-53-38/": "daily solar indices, 4+2, MAE",
    "model_2025-05-19-15-22-50/": "4+2 MSE",
    "model_2025-05-20-09-58-46/": "6 MSE",
    "model_2025-05-19-15-51-39/": "6 MSE bad run",
    "model_2025-05-19-16-24-04/": "From scratch MSE",
    "model_2025-05-19-16-44-58/": "From scratch MSE 7",
    "model_2025-05-20-11-23-28/": "From scratch with daily solar indices",
    "model_2025-05-11-16-53-38/": "Day 1, finetuned",
    "model_2025-05-20-11-23-28/": "Day 1, from scratch",
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
    outpath = 'outputs/val_loss_ft_double.png'

    # select which models you want to plot
    model_labels1 = ["From scratch MSE", "6 MSE", "4+2 MSE"]
    model_labels2 = ["From scratch with daily solar indices", "daily solar indices, 4+2, MAE"]


    models1 = [finetuned_model_dict[model] for model in model_labels1]
    models2 = [finetuned_model_dict[model] for model in model_labels2]



fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,8))

test_losses = {}


# if you want you can specify which colors to use in what order

# default colors
colors1 = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors2 = plt.rcParams['axes.prop_cycle'].by_key()['color']

# custom colors
colors1 = ["tab:blue", "tab:green", "tab:orange"]
colors2 = ["tab:blue", "tab:orange"]


i = 0
for model in models1:
    log_path = models_path + model + "logs.log"
    mse_val_losses = []
    mae_val_losses = []
    with open(log_path, "r") as file:
        for line in file:
            if 'MSE Val' in line:
                mse_val_losses.append(float(line.split(' ')[-1]))

            if 'MAE Val' in line:
                mae_val_losses.append(float(line.split(' ')[-1]))
    
            if "Evaluation" in line:
                test_losses[model] = float(line.split(' ')[-1])
    
    
    """MSE validation loss -> can compare all models"""
    # doesn't include zero-shot loss -> note that zero-shot error is much bigger -> need to limit y axis
    # ax1.plot(range(1, len(mse_val_losses)), mse_val_losses[1:], label = model, c=colors1[i])

    # includes zero-shot loss
    # ax1.plot(range(0, len(mse_val_losses)), mse_val_losses, label = model, c=colors1[i])


    """MAE validation loss -> some very old models don't have this"""
    # doesn't include zero-shot loss -> note that zero-shot error is much bigger -> need to limit y axis
    # ax1.plot(range(1, len(mae_val_losses)), mae_val_losses[1:], label = model, c=colors1[i])

    # includes zero-shot loss
    ax1.plot(range(0, len(mae_val_losses)), mae_val_losses, label = model, zorder=3, c=colors1[i])
    i+=1

j=0
for model in models2:
    log_path = models_path + model + "logs.log"
    mse_val_losses = []
    mae_val_losses = []
    with open(log_path, "r") as file:
        for line in file:
            if 'MSE Val' in line:
                mse_val_losses.append(float(line.split(' ')[-1]))

            if 'MAE Val' in line:
                mae_val_losses.append(float(line.split(' ')[-1]))
    
            if "Evaluation" in line:
                test_losses[model] = float(line.split(' ')[-1])
    
    """MSE validation loss -> can compare all models"""
    # doesn't include zero-shot loss -> note that zero-shot error is much bigger -> need to limit y axis
    # ax2.plot(range(1, len(mse_val_losses)), mse_val_losses[1:], label = model, c=colors2[j])

    # includes zero-shot loss
    # ax2.plot(range(0, len(mse_val_losses)), mse_val_losses, label = model, c=colors2[j])


    """MAE validation loss -> some very old models don't have this"""
    # doesn't include zero-shot loss -> note that zero-shot error is much bigger -> need to limit y axis
    # ax2.plot(range(1, len(mae_val_losses)), mae_val_losses[1:], label = model, c=colors2[j])

    # includes zero-shot loss
    ax2.plot(range(0, len(mae_val_losses)), mae_val_losses, label = model, zorder=3, c=colors2[j])
    j+=1


# if you want to change the labels you can do it here
# model_labels1 = ["FCN with no Pretraining, \nMSE and no Solar Indices", "FCN with Pretraining, \nMSE and no Solar Indices", "TwoStage with Pretraining, \nMSE and no Solar Indices"] 
# model_labels2 = ["FCN with no Pretraining, \nMAE and Daily Solar Indices", "TwoStage with Pretraining, \nMAE and Daily Solar Indices", "Final Val Loss of FCN Model \nwith no Pretraining but \nDaily Solar Indices"]


pp(test_losses)
ax1.set_ylim([1.5,15])
ax2.set_ylim([1.5,15])
ax2.axhline(3.16, c="darkgrey", zorder=1) # for final val loss when training from scratch
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax1.legend(labels=model_labels1, loc="upper right", fontsize=16)
ax2.legend(labels=model_labels2, loc="upper right", fontsize=16)
# ax1.set_xlabel("Epoch", fontsize=20, labelpad=8)
# ax2.set_xlabel("Epoch", fontsize=20, labelpad=8)
fig.supxlabel("Epoch", fontsize=22)
# ax1.set_ylabel("MAE Validation Loss", fontsize=20, labelpad=8)
# ax2.set_ylabel("MAE Validation Loss", fontsize=20, labelpad=8)
fig.supylabel("MAE Validation Loss", fontsize=22)
fig.suptitle("Validation Loss During Training", fontsize=28)
fig.tight_layout(rect=[0.01, 0, 0.95, 0.99]) 

fig.savefig(outpath)