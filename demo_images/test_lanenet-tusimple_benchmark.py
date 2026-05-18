import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

import model.lanenet as lanenet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--arch", type=str, default="fcn")
    parser.add_argument("--dual_decoder", action="store_true")
    return parser.parse_args()


def preprocess(img_path):
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = cv2.resize(img, (512, 256))
    img_float = img.astype(np.float32)

    # BGR image, VGG mean subtract
    img_float -= VGG_MEAN

    # HWC -> CHW, scale
    tensor = img_float.transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(tensor).float().unsqueeze(0)

    return img, tensor


def build_model(args, device):
    arch = args.arch

    if "fcn" in arch.lower():
        model_name = "lanenet.LaneNet_FCN_Res"
    elif "enet" in arch.lower():
        model_name = "lanenet.LaneNet_ENet"
    elif "icnet" in arch.lower():
        model_name = "lanenet.LaneNet_ICNet"
    else:
        raise ValueError("arch must be fcn, enet, or icnet")

    model_name = model_name + "_1E2D" if args.dual_decoder else model_name + "_1E1D"

    print("Architecture:", model_name)

    net = eval(model_name)()
    net = nn.DataParallel(net)
    net.to(device)
    net.eval()

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return net


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs("demo_output", exist_ok=True)

    net = build_model(args, device)

    image_files = [
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with torch.no_grad():
        for file in image_files:
            img_path = os.path.join(args.img_dir, file)
            original, input_tensor = preprocess(img_path)
            input_tensor = input_tensor.to(device)

            embeddings, logit = net(input_tensor)
            pred_bin = torch.argmax(logit, dim=1, keepdim=True)

            mask = pred_bin[0, 0].cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))

            overlay = original.copy()
            overlay[mask_resized > 0] = [0, 0, 255]

            result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

            out_path = os.path.join("demo_output", f"result_{file}")
            cv2.imwrite(out_path, result)

            print("Saved:", out_path)


if __name__ == "__main__":
    main()