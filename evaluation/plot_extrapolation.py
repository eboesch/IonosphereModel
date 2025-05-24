from matplotlib import pyplot as plt
from pprint import pp
import numpy as np


# MODEL_TYPE = "pretrained"
MODEL_TYPE = "finetuned"
assert (MODEL_TYPE == "finetuned") or (MODEL_TYPE == "pretrained"), "Unknown Model Type"

dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"

outpath = 'outputs/extrapolation.png'

days = np.array([1,8,16,32,64])
FCN_val_loss = np.array([3.16, 2.98, 3.33, 3.04, 3.91])
FCN_test_loss = np.array([3.7, 2.99, 3.73, 4.01, 5.32])
TS_val_loss = np.array([2.56, 2.49, 2.98, 2.62, 3.42])
TS_test_loss = np.array([3.04, 2.75, 3.24, 3.56, 5.14])
TS_zeroshot_loss = np.array([8.69554, 5.684754, 13.844563, 13.761117, 38.440984])

loss_dict = {
    "FCN_val_loss": FCN_val_loss,
    "FCN_test_loss": FCN_test_loss,
    "TS_val_loss": TS_val_loss,
    "TS_test_loss": TS_test_loss,
    "TS_zeroshot_loss": TS_zeroshot_loss,
}



fig, ax = plt.subplots(figsize=(12,8))

# selection = [""]

# ax.plot(days, FCN_val_loss, label="FCN_val_loss")
# ax.plot(days, FCN_test_loss, label="FCN_test_loss")
# ax.plot(days, TS_val_loss, label="TS_val_loss")
# ax.plot(days, TS_test_loss, label="TS_test_loss")
ax.plot(days, TS_zeroshot_loss, label="TS_zeroshot_loss")
# ax.plot(days, FCN_val_loss-TS_val_loss, label="val diff")
# ax.plot(days, FCN_test_loss-TS_test_loss, label="test diff")
# ax.plot(days, TS_zeroshot_loss-TS_val_loss, label="zeroshot - val")
ax.plot(days, TS_zeroshot_loss-TS_test_loss, label="zeroshot - test")

# model_labels = []



# ax.set_ylim([2,20])
# ax.set_ylim([0,15])
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.axhline(40.94, c="k") # for final val loss when training from scratch
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xticks(days)
# fig.legend(labels=model_labels, loc="center right", fontsize = 14, bbox_to_anchor=(0.93, 0.5))
# ax.legend(labels=model_labels, loc="upper right", fontsize=16)
ax.legend(fontsize=16)
# fig.subplots_adjust(right=0.7)
# ax.set_xlabel("Epoch", fontsize=18, labelpad=8)
# ax.set_ylabel("MAE Validation Loss", fontsize=18, labelpad=8)
# ax2.set_xlabel("Epoch", fontsize=18, labelpad=8)
# ax1.set_ylabel("MAE Validation Loss", fontsize=18, labelpad=8)
fig.suptitle("Loss when finetuning further into the future", fontsize=28)
# ax.set_tight_layout()
fig.supxlabel("Days After July 1st", fontsize=22)
fig.supylabel("MAE Loss", fontsize=22)
# fig.suptitle("MAE Validation Loss During Training", fontsize=24)
fig.tight_layout(rect=[0.01, 0, 0.95, 0.99]) 

fig.savefig(outpath)