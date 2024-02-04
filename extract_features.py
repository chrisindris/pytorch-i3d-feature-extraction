# time python extract_features.py --mode flow --load_model models/flow_imagenet.pt --input_dir /data/i5O/THUMOS14/actionformer_subset_i3d_flows_all/ --output_dir ./out_flow_imagenet_fps30_oversample_freq4 --sample_mode oversample --frequency 4 --usezip

import os
import pathlib
import shutil

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import io
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse
import torchvision
from PIL import Image
from pathlib import Path

import numpy as np
import mmcv

from pytorch_i3d import InceptionI3d

import subprocess
import time

import pdb


def build_video_names_list(input_dir, output_dir, start_index, end_index):
    video_names = sorted(
        [
            i
            for i in os.listdir(input_dir)[start_index:end_index]
            if (i.startswith("video") or i.startswith("out"))
            and (
                os.path.exists(os.path.join(input_dir, i, "img.zip"))
                or os.path.exists(os.path.join(input_dir, i, "flow_0.flo"))
            )
            and i + "-flow.npz" not in os.listdir(output_dir)
        ]
    )
    return video_names


def load_frame(frame_file, resize=False, flow=False, direction="y"):
    if flow:
        data = mmcv.flowread(frame_file)
        data = data[:, :, int(direction == "y")]
        data = Image.fromarray(data)
    else:
        data = Image.open(frame_file)

    data = data.resize((340, 256), Image.ANTIALIAS)

    assert data.size[1] == 256
    assert data.size[0] == 340

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)

    if flow:
        data = data - data.min()
        data = data / data.max()
        data = (2 * data) - 1
    else:
        data = (data * 2 / 255) - 1

    assert data.max() <= 1.0
    assert data.min() >= -1.0

    return data


def load_zipframe(zipdata, name, resize=False, flow=False, direction="x"):
    stream = zipdata.read(name)

    # print("load frame")

    if flow:
        data = mmcv.flow_from_bytes(stream)
        data = data[:, :, int(direction == "y")]
        data = Image.fromarray(data)
    else:
        data = Image.open(io.BytesIO(stream))

    # print("data.size=", data.size)
    data = data.resize((340, 256), Image.ANTIALIAS)

    assert data.size[1] == 256
    assert data.size[0] == 340

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)

    if flow:
        data = data - data.min()
        data = data / data.max()
        data = (2 * data) - 1
    else:
        data = (data * 2 / 255) - 1

    assert data.max() <= 1.0
    assert data.min() >= -1.0

    return data


def oversample_data(data):  # (39, 16, 224, 224, 2)  # Check twice
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])  # ,:,16:240,58:282,:
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_f_1,
        data_f_2,
        data_f_3,
        data_f_4,
        data_f_5,
    ]


def load_rgb_batch(frames_dir, rgb_files, frame_indices, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]]), resize
            )

    return batch_data


def load_ziprgb_batch(rgb_zipdata, rgb_files, frame_indices, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_zipframe(
                rgb_zipdata, rgb_files[frame_indices[i][j]], resize
            )

    return batch_data


def load_flow_batch(frames_dir, flow_files, frame_indices, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, 0] = load_frame(
                os.path.join(frames_dir, flow_files[frame_indices[i][j]]),
                resize,
                flow=True,
                direction="x",
            )

            batch_data[i, j, :, :, 1] = load_frame(
                os.path.join(frames_dir, flow_files[frame_indices[i][j]]),
                resize,
                flow=True,
                direction="y",
            )

    return batch_data


def load_zipflow_batch(
    flow_zipdata,
    flow_files,
    frame_indices,
    resize=False,
):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, 0] = load_zipframe(
                flow_zipdata,
                flow_files[frame_indices[i][j]],
                resize,
                flow=True,
                direction="x",
            )

            batch_data[i, j, :, :, 1] = load_zipframe(
                flow_zipdata,
                flow_files[frame_indices[i][j]],
                resize,
                flow=True,
                direction="y",
            )

    return batch_data


def run(
    args,
    mode="rgb",
    load_model="",
    sample_mode="oversample",
    frequency=16,
    input_dir="",
    output_dir="",
    batch_size=40,
    usezip=False,
    start_index=0,
    end_index=412,
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(args)
    # args.batch_size = 500
    # print(args)

    chunk_size = 16

    assert mode in ["rgb", "flow"]
    assert sample_mode in ["oversample", "center_crop", "resize"]

    # setup the model
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224

        # b_data = Variable(b_data.cuda(), volatile=True).float()
        with torch.no_grad():
            b_data = b_data.cuda().float()

        b_features = i3d.extract_features(b_data)

        b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
        return b_features

    # sorted to fix the order, and not in to specify that we don't want the same ones twice
    # video_names = sorted([i for i in os.listdir(input_dir) if i[0] == 'v'])
    # video_names = sorted(
    #    [
    #        i
    #        for i in os.listdir(input_dir)[start_index:end_index]
    #        if (i.startswith("video") or i.startswith("out"))
    #        and (
    #            os.path.exists(os.path.join(input_dir, i, "img.zip"))
    #            or os.path.exists(os.path.join(input_dir, i, "flow_0.flo"))
    #        )
    #        and i + "-flow.npz" not in os.listdir(output_dir)
    #    ]
    # )
    counter = 0
    while (
        True
    ):  # len(build_video_names_list(input_dir, output_dir, start_index, end_index)):
        video_names = build_video_names_list(
            input_dir, output_dir, start_index, end_index
        )

        if video_names == []:
            print("waiting, counter =", counter)
            if counter >= 45:
                break
            else:
                counter += 1
                time.sleep(60)
                continue
        else:
            counter = 0

        video_name = video_names[0]
        usezip = os.path.exists(os.path.join(input_dir, video_name, "img.zip"))

        print(video_names)
        print(video_name)

        save_file = "{}-{}.npz".format(video_name, mode)
        if save_file in os.listdir(output_dir):
            continue

        frames_dir = os.path.join(input_dir, video_name)

        if mode == "rgb":
            if usezip:
                print(frames_dir)
                rgb_zipdata = zipfile.ZipFile(os.path.join(frames_dir, "img.zip"), "r")
                rgb_files = [i for i in rgb_zipdata.namelist() if i.endswith("jpg")]
            else:
                rgb_files = [i for i in os.listdir(frames_dir) if i.startswith("img")]

            rgb_files.sort()
            frame_cnt = len(rgb_files)

        else:
            if usezip:
                print(os.path.join(frames_dir, "img.zip"))
                flow_zipdata = zipfile.ZipFile(os.path.join(frames_dir, "img.zip"), "r")
                flow_files = [i for i in flow_zipdata.namelist() if i.endswith(".flo")]

            else:
                flow_files = [i for i in os.listdir(frames_dir) if i.endswith(".flo")]

            flow_files.sort()
            frame_cnt = len(flow_files)

        # clipped_length = (frame_cnt // chunk_size) * chunk_size   # Cut frames

        # Cut frames
        print("frame_cnt=", frame_cnt)
        print("chunk_size=", chunk_size)
        assert frame_cnt > chunk_size
        clipped_length = frame_cnt - chunk_size
        clipped_length = (
            clipped_length // frequency
        ) * frequency  # The start of last chunk

        frame_indices = []  # Frames to chunks
        for i in range(clipped_length // frequency + 1):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)]
            )

        frame_indices = np.array(frame_indices)

        # frame_indices = np.reshape(frame_indices, (-1, 16)) # Frames to chunks
        chunk_num = frame_indices.shape[0]

        batch_num = int(np.ceil(chunk_num / batch_size))  # Chunks to batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        if sample_mode == "oversample":
            full_features = [[] for i in range(10)]
        else:
            full_features = [[]]

        for batch_id in range(batch_num):
            require_resize = sample_mode == "resize"

            if mode == "rgb":
                if usezip:
                    batch_data = load_ziprgb_batch(
                        rgb_zipdata, rgb_files, frame_indices[batch_id], require_resize
                    )
                else:
                    batch_data = load_rgb_batch(
                        frames_dir, rgb_files, frame_indices[batch_id], require_resize
                    )
            else:
                if usezip:
                    batch_data = load_zipflow_batch(
                        flow_zipdata,
                        flow_files,
                        frame_indices[batch_id],
                        require_resize,
                    )
                else:
                    batch_data = load_flow_batch(
                        frames_dir,
                        flow_files,
                        frame_indices[batch_id],
                        require_resize,
                    )

            if sample_mode == "oversample":
                batch_data_ten_crop = oversample_data(batch_data)

                for i in range(10):
                    # pdb.set_trace()
                    assert batch_data_ten_crop[i].shape[-2] == 224
                    assert batch_data_ten_crop[i].shape[-3] == 224
                    full_features[i].append(forward_batch(batch_data_ten_crop[i]))

            else:
                if sample_mode == "center_crop":
                    batch_data = batch_data[
                        :, :, 16:240, 58:282, :
                    ]  # Centrer Crop  (39, 16, 224, 224, 2)

                assert batch_data.shape[-2] == 224
                assert batch_data.shape[-3] == 224
                full_features[0].append(forward_batch(batch_data))

        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)

        np.savez(
            os.path.join(output_dir, save_file),
            feature=full_features,
            frame_cnt=frame_cnt,
            video_name=video_name,
        )

        print(
            "{} done: {} / {}, {}".format(
                video_name, frame_cnt, clipped_length, full_features.shape
            )
        )

        if usezip:
            os.remove(os.path.join(input_dir, video_name, "img.zip"))
        else:
            shutil.rmtree(os.path.join(input_dir, video_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--sample_mode", type=str)
    parser.add_argument("--frequency", type=int, default=16)

    parser.add_argument("--usezip", dest="usezip", action="store_true")
    parser.add_argument("--no-usezip", dest="usezip", action="store_false")

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=412)

    parser.set_defaults(usezip=True)

    args = parser.parse_args()

    run(
        args,
        mode=args.mode,
        load_model=args.load_model,
        sample_mode=args.sample_mode,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        frequency=args.frequency,
        usezip=args.usezip,
        start_index=args.start_index,
        end_index=args.end_index,
    )
