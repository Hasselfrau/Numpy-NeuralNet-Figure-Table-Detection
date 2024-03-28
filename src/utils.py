import os
import sys
import fitz
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import conf
from collections.abc import Collection
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def collect_metadata(df: pd.DataFrame, doc_id: str, page: int, img_path: str, x: np.ndarray, table: int,
                     figure: int) -> pd.DataFrame:
    """
    Collects metadata for each image and stores it in a dataframe.
    :param df: dataframe
    :param doc_id: document name
    :param page: page number
    :param img_path: path to png image of document page
    :param x: image converted to vector
    :param table: 1 if page contains table, 0 if not. None if data is missing
    :param figure: 1 if page contains figure, 0 if not. None if data is missing
    :return:
    """
    df.loc[len(df)] = [doc_id, page, img_path, x, table, figure]
    return df


def pix2np(pix):
    """
    Convert a PDFNet PixelReader object to a numpy array.
    :param pix (PixelReader): The PixelReader object containing image data.
    :return: im (ndarray): The image data as a numpy array.
    Note:
    - If the colorspace is not grayscale, the image will be converted from RGB to BGR format.
    """
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if str(pix.colorspace) != "Colorspace(CS_GRAY) - DeviceGray":
        im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im


def save_vectors(vectors: Collection[np.ndarray]) -> None:
    """
    Save a list of vectors to a dataset in an HDF5 file.
    :param vectors (List[np.ndarray]): A list of vectors to be saved.
    :return: None
    """
    # Open or create an HDF5 file
    file_path = conf.dataset_path
    file_mode = 'a'  # 'a' for append mode, 'w' for write mode (creates a new file)
    with h5py.File(file_path, file_mode) as f:
        vectors_array = np.array(vectors)
        f.create_dataset('x', data=vectors_array)
        f.close()


def convert_pdfs_to_png(input_dir: str, zoom: int = 1, filename:str = None) -> Collection:
    """
    Converts all pages of pdf files in input directory to png files into directory ./png and collects metadata to a
    pandas dataframe.
    Iterating over pdf files in directory this function:
        1. Opens pdf file
        2. Gets vectors of each page
        3. Saves as a png

    :param input_dir: directory with pdf files that we will convert
    :param zoom: page zoom (optional)
    :return: list of vectors
    """
    vectors = []
    df = pd.DataFrame(columns=["pdf_id", "page", "image_path", "x", "table", "figure"])
    # Check if the output directory exists, if not, create it
    output_dir = os.path.join(os.path.dirname(input_dir), "png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Go through each PDF file in the input directory
    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith('.pdf'):
            if filename and pdf_file != filename:
                continue
            pdf_path = os.path.join(input_dir, pdf_file)
            pdf_id = os.path.splitext(pdf_file)[0]  # Extract the ID from the PDF filename
            doc = fitz.open(pdf_path)
            mat = fitz.Matrix(zoom, zoom)

            for i, page in tqdm(enumerate(doc)):
                page_number = i + 1
                image_path = f"{output_dir}/{pdf_id}_{page_number}.png"
                # if not os.path.exists(image_path):
                pix = page.get_pixmap(matrix=mat)
                im = pix2np(pix)
                vectors.append(im)
                label = 0
                df = collect_metadata(df, pdf_id, page_number, image_path, label, table=label, figure=label)
                pix.save(image_path)

            doc.close()

            print(f"Converted {pdf_file} to {page_number} PNG images.")
    df.to_csv(f"{os.path.dirname(input_dir)}/{conf.metadata_filename}", index=False)
    return vectors


# input_directory = '/Users/michael/Projects/tables_recognition/data/pdf'
# vectors = convert_pdfs_to_png(input_directory, 1)
# save_vectors(vectors)

def load_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from an HDF5 file.
    :param dataset_path (str): Path to the HDF5 dataset file.
    :return: tuple: A tuple containing the following:
        - x (np.ndarray): Array of image vectors.
        - y_fig (np.ndarray): Array of figure labels.
        - y_table (np.ndarray): Array of table labels.
    """
    ds = h5py.File(dataset_path, "r")
    x = np.array(ds['x'][:])
    y_table = np.array(ds['y_table'][:])
    y_fig = np.array(ds['y_fig'][:])
    y_table = y_table.reshape((1, y_table.shape[0]))
    y_fig = y_fig.reshape((1, y_fig.shape[0]))
    ds.close()

    return x, y_fig, y_table


def train_test_split(x: np.ndarray,
                     y_table: np.ndarray,
                     y_fig: np.ndarray,
                     problem: str,
                     random_seed: int,
                     train_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets.

    :param x (np.ndarray): Array of image vectors.
    :param y_table (np.ndarray): Array of table labels.
    :param y_fig (np.ndarray): Array of figure labels.
    :param problem (str): Type of problem, either "table" or "figure".
    :param random_seed (int): Random seed for reproducibility.
    :param train_size (float): Proportion of the dataset to include in the training set.
    :return: tuple: A tuple containing the following:
        - train_x (np.ndarray): Training set of image vectors.
        - test_x (np.ndarray): Testing set of image vectors.
        - train_y (np.ndarray): Training set of labels (table or figure).
        - test_y (np.ndarray): Testing set of labels (table or figure).
    """
    np.random.seed(seed=random_seed)
    labels = y_fig if problem == "figure" else y_table
    unique_labels = np.unique(labels)

    # Initialize arrays to store indices for train and test sets
    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)

    # Iterate over unique labels
    for label in unique_labels:
        label_indices = np.where(labels == label)[1]
        np.random.shuffle(label_indices)

        split_point = int(train_size * len(label_indices))

        # Append the indices to train and test sets
        train_indices = np.concatenate([train_indices, label_indices[:split_point]])
        test_indices = np.concatenate([test_indices, label_indices[split_point:]])
    # Shuffle the train and test sets
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Use the indices to get our train and test sets
    train_x, train_y = x[train_indices], labels[:, train_indices]
    test_x, test_y = x[test_indices], labels[:, test_indices]

    assert labels.shape[0] == train_y.shape[0] == test_y.shape[0]
    assert x.shape[1:] == train_x.shape[1:] == test_x.shape[1:]

    return train_x, test_x, train_y, test_y


def get_train_test_set(dataset_path: str,
                       problem: str,
                       random_seed: int = 42,
                       train_size: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset, split it into training and testing sets, and return the sets.

    :param dataset_path (str): Path to the dataset.
    :param problem (str): Type of problem, either "table" or "figure".
    :param random_seed (int): Random seed for reproducibility. Default is 42.
    :param train_size (float): Proportion of the dataset to include in the training set. Default is 0.8.
    :return: tuple: A tuple containing the following:
        - train_x (np.ndarray): Training set of image vectors.
        - test_x (np.ndarray): Testing set of image vectors.
        - train_y (np.ndarray): Training set of labels (table or figure).
        - test_y (np.ndarray): Testing set of labels (table or figure).
    """
    x, y_fig, y_table = load_dataset(dataset_path=dataset_path)
    train_x, test_x, train_y, test_y = train_test_split(x=x, y_table=y_table, y_fig=y_fig, problem=problem,
                                                        random_seed=random_seed, train_size=train_size)
    return train_x, test_x, train_y, test_y


def plot_confusion_matrix(actual, predicted) -> None:
    """
    Plots confusion matrix
    :param actual: actual documents with table or figure
    :param predicted: predicted documents with table or figure
    :return: None
    """
    cf_matrix = confusion_matrix(actual, predicted)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')