import argparse
import os
from os.path import abspath, dirname, join
from imageio.v3 import imread
import numpy as np
import sofar
import torch


AVG_HRTF_PATH = join(dirname(abspath(__file__)), 'data', 'Average_HRTFs.sofa')


class BaselineHRTFPredictor:
    def __init__(self, average_hrtf_path: str = AVG_HRTF_PATH):
        """Creates a predictor instance. average_HRTF_path is the path the file 'Average_HRTFs.sofa' that was delivered as part of the project."""
        self.average_hrir = sofar.read_sofa(average_hrtf_path, verbose=False)

    def predict(self, images: torch.Tensor) -> sofar.Sofa:
        """
        Predict the HRTF based on left and right images.

        Args:
            images: images for left and right pinna as 4-dimensional tensor of size (number of ears, number of images per ear, image height, image width)

        Returns:
            sofar.Sofa: Predicted HRIR in SOFA format.
        """
        return self.average_hrir


def main():
    parser = argparse.ArgumentParser(description="Baseline HRTF Inference Script")
    parser.add_argument("-l", "--left", metavar='IMAGE_PATH', type=str, nargs='+', required=True, help="List of left pinna images")
    parser.add_argument("-r", "--right", metavar='IMAGE_PATH', type=str, nargs='+', required=True, help="List of right pinna images")
    parser.add_argument("-o", "--output_path", metavar='SOFA_PATH', type=str, required=True, help="File path to save the predicted HRTF in SOFA format.")
    args = parser.parse_args()

    # load images
    left_images = torch.from_numpy(np.stack([imread(path) for path in args.left]))
    right_images = torch.from_numpy(np.stack([imread(path) for path in args.right]))
    images = torch.stack((left_images, right_images))

    # predict HRTFs
    predictor = BaselineHRTFPredictor()
    hrtf = predictor.predict(images)

    # write to output path
    os.makedirs(dirname(args.output_path), exist_ok=True)
    sofar.write_sofa(args.output_path, hrtf, compression=0)
    print(f"Saved HRTF to {args.output_path}")

if __name__ == "__main__":
    main()