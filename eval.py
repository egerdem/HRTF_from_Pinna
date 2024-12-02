

from inference import BaselineHRTFPredictor
from metrics import MeanSpectralDistortion
import torch
import numpy as np

predictor = BaselineHRTFPredictor()
metric = MeanSpectralDistortion()
results = {}

for task in range(3):
    sd = SonicomDatabase(sonicom_root, training_data=False, task_id=task, folder_structure='v2')
    test_dataloader = DataLoader(sd, batch_size=1, shuffle=False)

    total_error = []
    for image_batch, hrtf_batch in tqdm.tqdm(test_dataloader):
        for images, ground_truth_hrtf in zip(image_batch, hrtf_batch):
            predicted_hrtf = torch.as_tensor(sd._compute_HRTF(predictor.predict(images).Data_IR))
            total_error.append(metric.get_spectral_distortion(ground_truth_hrtf, predicted_hrtf))
    results[task] = np.mean(total_error)

results