import torch
import torch.utils.data.sampler as torch_sampler
from random import randint, choice


class TrainSampler(torch_sampler.BatchSampler):

    def __init__(self, subdivision, batch_size, max_iterations, num_samples, image_scales, scale_interval):

        self.subdivision = subdivision
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.image_scales = image_scales
        self.scale_interval = scale_interval

    def __iter__(self):
        batch = []

        minibatch_size = (self.batch_size / self.subdivision)
        image_pos = 0
        scale_counter = 0

        hflip_options = [False]
        hflip_options.append(True)

        # Starts with the smallest resolution
        actual_scale = self.image_scales[0]

        for i in range(self.max_iterations):
            image_idx = randint(0, self.num_samples - 1)

            batch.append((image_idx, actual_scale, choice(hflip_options), image_pos))
            image_pos += 1

            if image_pos == minibatch_size:
                yield batch
                batch = []
                image_pos = 0
                scale_counter += 1

                if len(self.image_scales) > 1 and self.scale_interval * (
                        self.batch_size / minibatch_size) == scale_counter:

                    new_scale = choice(self.image_scales)
                    while new_scale == actual_scale:
                        new_scale = choice(self.image_scales)

                    # print("Changing scale from",actual_scale,"to", new_scale)
                    actual_scale = new_scale

                    scale_counter = 0

    def __len__(self):
        return self.max_iterations


class TestSampler(torch_sampler.BatchSampler):
    def __init__(self, num_images):
        self.num_images = num_images

    def __iter__(self):
        batch = []

        hflip_options = [False]
        hflip_options.append(True)

        for scale in [480, 576, 688, 864, 1200]:
            for hflip in hflip_options:
                for image_idx in range(self.num_images):
                    batch.append((image_idx, scale, hflip, 0))
                    yield batch
                    batch = []

    def __len__(self):
        return self.num_images * len([480, 576, 688, 864, 1200]) * 2


def collate_minibatch(batch):
    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    rois = torch.cat([item[2] for item in batch])
    img_key = [item[3] for item in batch]
    gt = [item[4] for item in batch]

    return data, target, rois, img_key, gt