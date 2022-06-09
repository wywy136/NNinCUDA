import random

import numpy as np


class MNISTDataset():
    def __init__(self, split, batch_size):
        if split == "train":
            self.image_path = "/home/ywang27/HPC/project-4-wywy136/data/train-images-idx3-ubyte"
            self.label_path = "/home/ywang27/HPC/project-4-wywy136/data/train-labels-idx1-ubyte"
        else:
            self.image_path = "/home/ywang27/HPC/project-4-wywy136/data/t10k-images-idx3-ubyte"
            self.label_path = "/home/ywang27/HPC/project-4-wywy136/data/t10k-labels-idx1-ubyte"
        self.num_figs = 0
        self.fig_data = []
        self.fig_label = []
        self.batch_size = batch_size
        
        # Read image
        print("Reading images into memory...")
        f_image = open(self.image_path, "rb")
        self.offset = 0
        while True:
            if self.offset == 0:
                self.offset += 4
            if self.offset == 4:
                f_image.seek(self.offset)
                self.num_figs = int.from_bytes(f_image.read(4), "big")
                self.offset += 12
            if self.offset == 16:
                for i in range(self.num_figs):
                     
                    fig_data = []
                    for j in range(28 * 28):
                        f_image.seek(self.offset)
                        fig_data.append(int.from_bytes(f_image.read(1), "big"))
                        self.offset += 1
                    self.fig_data.append(fig_data)

                    # if i == 35662:
                    #    self.save_img(fig_data, split)
            else:
                break
                
        # Read label
        f_label = open(self.label_path, "rb")
        # Read from the first label
        self.offset = 8
        for i in range(self.num_figs):
            f_label.seek(self.offset)
            self.fig_label.append(int.from_bytes(f_label.read(1), "big"))
            self.offset += 1

            # if i == 35662:
            #     print(f"Label for img: {self.fig_label[-1]}")
        
        # Indices of image
        self.indices = [i for i in range(self.num_figs)]
        # Number of sampled images
        self.sampled_num = 0
    
    def save_img(self, fig_data, split):
        with open(f"temp_{split}", 'w') as f:
            for i in range(28):
                for j in range(28):
                    f.write(str(fig_data[i * 28 + j]))
                    f.write('\t')
                f.write('\n')

    def get_batch_data(self):
        images, labels = [], []
        sampled_indices = []
        for i in range(self.batch_size):
            # Generate random int in range [0, num_figs-1 - sampled_num]
            index = random.randint(0, self.num_figs - 1 - self.sampled_num)
            sampled_indices.append(index)
            images.append(self.fig_data[self.indices[index]])
            labels.append(self.fig_label[self.indices[index]])
            # Swap the index with the last unsampled index
            self.indices[index], self.indices[self.num_figs - 1 - self.sampled_num] = \
                self.indices[self.num_figs - 1 - self.sampled_num], self.indices[index]
            self.sampled_num += 1

            # Sampled all images
            if self.sampled_num == self.num_figs:
                print("All training data is exhausted. Starting a new iteration.")
                self.sampled_num = 0
                self.indices = [i for i in range(self.num_figs)]
                break
        
        # print(labels)
        # Make one-hot labels
        for i in range(len(labels)):
            category = labels[i]
            labels[i] = [0 for _ in range(10)]
            labels[i][category] = 1

        return np.array(images), np.array(labels)
    
    def __len__(self):
        return self.num_figs // self.batch_size + 1