from matplotlib import pyplot as plt
from pprint import pp

dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
# pretrained models
# models_path = dslab_path + "pretrained_models/"
# models = ["model_2025-04-25-19-09-00/", "model_2025-04-25-15-03-51/", "model_2025-04-28-09-08-52/"]

# fine tuned models
models = ["model_2025-04-28-12-21-37/", "model_2025-04-28-12-37-02/", "model_2025-04-28-12-50-35/", "model_2025-04-28-13-02-05/", "model_2025-04-28-13-13-17/"]
models_path = dslab_path + "models/"

fig, ax = plt.subplots(figsize=(10,10))

test_losses = {}

for model in models:
    log_path = models_path + model + "logs.log"
    val_losses = []
    with open(log_path, "r") as file:
        for line in file:
            if 'Validation' in line:
                val_losses.append(float(line.split(' ')[-1]))
    
            if "Evaluation" in line:
                test_losses[model] = float(line.split(' ')[-1])
    
    ax.scatter(range(0, len(val_losses)), val_losses, label = model)
pp(test_losses)
ax.set_ylim([0,1000])
fig.legend()
fig.savefig('outputs/val_loss_ft.png')