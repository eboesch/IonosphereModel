from matplotlib import pyplot as plt
from pprint import pp


# MODEL_TYPE = "pretrained"
MODEL_TYPE = "finetuned"
assert (MODEL_TYPE == "finetuned") or (MODEL_TYPE == "pretrained"), "Unknown Model Type"

dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
# pretrained models

pretrained_model_dict = {
    "model_2025-04-25-19-09-00/": "Hector 1", 
    "model_2025-04-25-15-03-51/": "Hector 2", 
    "model_2025-04-28-09-08-52/": "Hector 3", 
    "model_2025-04-30-14-05-28/": "4+1",
    "model_2025-04-30-21-02-28/": "6", 
    "model_2025-04-30-23-22-21/": "4+2", 
    "model_2025-05-01-16-14-32/": "6+3", 
    "model_2025-05-01-23-02-56/": "4+2, with smaller batch size",
    "model_2025-05-04-12-10-18/": "4+2, with MAE training loss",    
    "model_2025-05-10-21-34-55/": "hourly solar indices, 4+2, MAE",
    "model_2025-05-10-17-39-31/": "daily solar indices, 4+2, MAE",    
}


if MODEL_TYPE == "pretrained":
    models_path = dslab_path + "pretrained_models/"
    outpath = 'outputs/val_loss_pt.png'

    # model_labels = ["4+1", "6", "4+2", "6+3", "4+2, with smaller batch size", "4+2, with MAE training loss"]
    # model_labels = ["4+2", "4+2, with MAE training loss", "hourly solar indices, 4+2, MAE", "daily solar indices, 4+2, MAE"]
    model_labels = ["hourly solar indices, 4+2, MAE", "daily solar indices, 4+2, MAE"]

    models = [key for key, value in pretrained_model_dict.items() if value in model_labels]



# fine tuned models

finetuned_model_dict = {
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
}

# flip dictionary
finetuned_model_dict = {value: key for key, value in finetuned_model_dict.items()}


if MODEL_TYPE == "finetuned":
    models_path = dslab_path + "models/"
    outpath = 'outputs/val_loss_ft.png'

    # model_labels = ["4+1", "4+2", "4+2 0.5 FT batch size", "4+2 0.5 batch size", 
                # "4+2 0.5 FT LR", "4+2 0.1 FT LR", 
                # "6+3", "6+3 0.5 FT LR",
                # "4+2, MAE"]
    # model_labels = ["from scratch, 4", "from scratch, 4, MAE", "4+2", "4+2, MAE"]
    # model_labels = ["from scratch, 4, MAE", "4+2, MAE"]
    # model_labels = ["4+2, MAE", "4+2, MAE 0.5 FT LR", "4+2, MAE bad formatting"]
    # model_labels = ["4+2", "4+2, MAE", "from scratch, 4, MAE", "hourly solar indices, 4+2, MAE", "daily solar indices, 4+2, MAE"]
    model_labels = ["hourly solar indices, 4+2, MAE", "daily solar indices, 4+2, MAE"]

    models = [finetuned_model_dict[model] for model in model_labels]



fig, ax = plt.subplots(figsize=(12,8))

test_losses = {}

for model in models:
    log_path = models_path + model + "logs.log"
    mse_val_losses = []
    val_losses = []
    with open(log_path, "r") as file:
        for line in file:
            if 'Validation' in line:
                mse_val_losses.append(float(line.split(' ')[-1]))

            if 'Val ' in line:
                val_losses.append(float(line.split(' ')[-1]))
    
            if "Evaluation" in line:
                test_losses[model] = float(line.split(' ')[-1])
    
    
    """MSE validation loss -> can compare to older models"""
    # doesn't include zero-shot loss -> important for MSE loss, since zero-shot error is much bigger and makes plot unreadable
    # ax.plot(range(1, len(mse_val_losses)), mse_val_losses[1:], label = model)

    # includes zero-shot loss
    # ax.plot(range(0, len(mse_val_losses)), mse_val_losses, label = model)


    """validation loss same as chosen training loss -> only works on newer models"""
    # doesn't include zero-shot loss
    ax.plot(range(1, len(val_losses)), val_losses[1:], label = model)

    # includes zero-shot loss
    # ax.plot(range(0, len(val_losses)), val_losses, label = model)

pp(test_losses)
# ax.set_ylim([0,300])
# ax.axhline(40.94, c="k") # for final val loss when training from scratch
# ax.axhline(3.98, c="k") # for final val loss when training from scratch
fig.legend(labels=model_labels, loc="center right", bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right=0.75)
fig.supxlabel("Epoch")
fig.supylabel("Validation Loss")
fig.suptitle("Validation Loss During Training", fontsize=20)

fig.savefig(outpath)