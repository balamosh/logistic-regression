import os

import cv2
import numpy as np
from pyexpat import features
from tqdm import tqdm

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import ImageDataPreprocessing


class AlzheimerDataset:
    """A class for the Alzheimer dataset. This class reads the data and preprocesses it."""

    def __init__(self, config):
        """Initializes the Alzheimer dataset class instance."""
        self.config = config

        # Read an annotation file that contains the image path, set_type, and target values for the entire dataset
        self.annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annotation_filename))

        # Preprocessing class initialization
        self.preprocessing = ImageDataPreprocessing(config.preprocess_type, config.preprocess_params)

        # Read and preprocess the data
        self.data = {}
        for set_type in SetType:
            self.data[set_type.name] = self.preprocess_data(set_type)

    def preprocess_data(self, set_type: SetType) -> dict:
        """Reads and preprocesses the data.

        Args:
            set_type: Data set_type from SetType.

        Returns:
            A dict with the following data:
                {'features': images (numpy.ndarray), 'targets': targets (numpy.ndarray), 'paths': list of paths}
        """

        images = []
        paths = []

        # Get dataframe rows with corresponding set_type
        df = self.annotation[self.annotation['set'] == set_type.name]

        # Drop duplicates
        if set_type != SetType.test:
            df = df.drop_duplicates()

        tqdm_df = tqdm(df.iterrows(),
                       total=len(df),
                       desc=f'{set_type.name.title()} set images reading')

        # Read images from dataframe, create 'images' and 'paths' arrays
        for index, row in tqdm_df:
            path = os.path.normpath(row['path'])
            path = os.path.join(self.config.path_to_data ,path)
            paths.append(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img.shape != self.config.image_size:
                img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LANCZOS4)
            images.append(img)

        # Stack images to numpy array with float values
        image_stack = np.stack(images, dtype=np.float64)

        # Preprocessing aka scaling image values
        if set_type == SetType.train:
            scaled_images = self.preprocessing.train(image_stack)
        else:
            scaled_images = self.preprocessing(image_stack)

        # Create 'target' column with int values
        if set_type != SetType.test:
            targets = df['target'].to_numpy(dtype=int)
        else:
            targets = None

        return {'features': scaled_images, 'targets': targets, 'path': paths}

    def __call__(self, set_type: str) -> dict:
        """Returns preprocessed data."""
        return self.data[set_type]
