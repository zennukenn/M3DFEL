import os
import torch
import glob
import os
import numpy as np
import csv
import PIL.Image as Image
import torchvision
from torch.utils import data

from .video_transform import *


class DaiseeDataset(data.Dataset):
    def __init__(self, args, mode):
        """Dataset for Daisee

        Args:
            args
            mode: String("train" or "test")

            num_frames: the number of sampled frames from every video, default: 16
            image_size: crop images to 112*112

        """
        self.args = args
        self.path = self.args.train_dataset if mode == "train" else self.args.test_dataset
        self.num_frames = self.args.num_frames
        self.image_size = self.args.crop_size
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()

        pass

    def get_data(self):
        """get data path, label from the csv file

        Returns:
            data_list: List of dictionaries with keys "path", "labels", "num_frames"
        """
        full_data = []

        print("loading data")

        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                pathname = row[0].split('.')[0]
                path_person = pathname[:6]

                labels = [int(row[i]) for i in range(1, 5)]



                # Combine the path
                if self.mode == "train":
                    path = os.path.join("D:\dataset\DAiSEE_notest\DataSet\Train",path_person, pathname, "face")
                elif self.mode == "test":
                    path = os.path.join("D:\dataset\DAiSEE_notest\DataSet\Validation",path_person, pathname, "face")
                full_num_frames = len(os.listdir(path))

                # Get the paths of the frames of a video and sort them
                full_video_frames_paths = glob.glob(os.path.join(path, '*.jpg'))
                full_video_frames_paths.sort()

                full_data.append({"path": full_video_frames_paths,
                                  "labels": labels,
                                  "num_frames": full_num_frames})

        print("daisee loaded")

        return full_data

    def get_transform(self):
        """get trasform accorging to train/test mode and args including: crop, flip, color jitter

        Returns:
            transform
        """
        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(
                                                            self.args.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        elif self.mode == "test":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        Stack(),
                                                        ToTorchFormatTensor()])

        return transform

    def __getitem__(self, index):

        # get the data according to index
        data = self.data[index]
        full_video_frames_paths = data['path']
        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)

        # when getting the frames, randomly choose the neighbour to augment
        for i in range(28):
            # frame = int(full_num_frames * i / self.num_frames)
            # if self.args.random_sample:
            #     frame += int(random.random() * self.num_frames)
            #     frame = min(full_num_frames - 1, frame)
            video_frames_paths.append(full_video_frames_paths[i])

        # get the images and transform
        images = []
        for video_frames_path in video_frames_paths:
            images.append(Image.open(video_frames_path).convert('RGB'))
        images = self.transform(images)
        images = torch.reshape(
            images, (-1, 3, self.image_size, self.image_size))
        labels = torch.tensor(data["labels"])

        return images, labels

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        """Return the labels of the dataset

        Returns:
            List of labels
        """
        return [data["labels"] for data in self.data]