import os

import torch

from backbones.iresnet import iresnet50
from eval.verification import load_bin, test


def main(
    chkpt_fp: str = "checkpoints/backbone_casia_cosface_iresnet50.pth",
    eval_dir: str = os.environ["DATASET_DIR"] + "/EvalDatasets/binfiles",
    datasets: list[str] = ["lfw.bin"],
):
    # Assign device where code is executed
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Neural Engine (MPS)
    else:
        device = torch.device("cpu")  # Default to CPU
    print("Device:", device)

    backbone = iresnet50(num_features=512)
    backbone.load_state_dict(torch.load(chkpt_fp, weights_only=True))
    backbone.to(device)

    data_set = load_bin(os.path.join(eval_dir, datasets[0]), image_size=(112, 112))
    backbone.eval()
    # TODO: Currently test() is throwing an error
    acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set, backbone, 10, 10)
    print("XNorm: ", xnorm)


if __name__ == "__main__":
    main()