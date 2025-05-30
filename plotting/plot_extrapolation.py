"""Script used to plot the results of our extrapolation experiment."""
from matplotlib import pyplot as plt
import numpy as np


dslab_path = "/cluster/work/igp_psr/dslab_FS25_data_and_weights/"
outpath = 'outputs/extrapolation.png'


days = np.array([1,8,16,32,64])
FCN_val_loss = np.array([3.16, 2.98, 3.33, 3.04, 3.91])
FCN_test_loss = np.array([3.7, 2.99, 3.73, 4.01, 5.32])
TS_val_loss = np.array([2.56, 2.49, 2.98, 2.62, 3.42])
TS_test_loss = np.array([3.04, 2.75, 3.24, 3.56, 5.14])
TS_zeroshot_loss = np.array([8.69554, 5.684754, 13.844563, 13.761117, 38.440984])

val_diff = FCN_val_loss-TS_val_loss
test_diff =  FCN_test_loss-TS_test_loss

fig, ax = plt.subplots(figsize=(12,8))

# plotting the different curves
ax.plot(days, FCN_val_loss, label="FCN_val_loss")
ax.plot(days, FCN_test_loss, label="FCN_test_loss")
ax.plot(days, TS_val_loss, label="TS_val_loss")
ax.plot(days, TS_test_loss, label="TS_test_loss")
ax.plot(days, TS_zeroshot_loss, label="TS_zeroshot_loss")

# plotting pretrained vs finetuned difference
# ax.plot(days, val_diff, label="val diff")
# ax.plot(days, test_diff, label="test diff")

# plotting zeroshot vs final losses
# ax.plot(days, TS_zeroshot_loss-TS_val_loss, label="zeroshot - val")
# ax.plot(days, TS_zeroshot_loss-TS_test_loss, label="zeroshot - test")
# ax.plot(days, TS_zeroshot_loss/FCN_test_loss, label="zeroshot / test")


print("val diff:", val_diff)
print("test diff:", test_diff)


ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xticks(days)
ax.legend(fontsize=16)
fig.suptitle("Loss when finetuning further into the future", fontsize=28)
fig.supxlabel("#Days After Pretraining", fontsize=22)
fig.supylabel("MAE Loss", fontsize=22)

fig.savefig(outpath)