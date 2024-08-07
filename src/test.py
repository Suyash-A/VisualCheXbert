import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils import test
from models.bert_labeler import bert_labeler  # Ensure this path is correct
from datasets.impressions_dataset import ImpressionsDataset  # Ensure this path is correct
from torch.utils.data import DataLoader

def collate_fn_with_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'label', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp', 'label' and 'len' where
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension, 'label' is a tensor of labels and
                                 'len' is a list of the length of each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = pad_sequence(tensor_list, batch_first=True, padding_value=0)  # Adjust padding value if necessary
    label_list = [s['label'] for s in sample_list]
    batched_labels = torch.stack(label_list, dim=0)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'label': batched_labels, 'len': len_list}
    return batch

# Define the paths to the checkpoint and test data
checkpoint_path = "/vol/bitbucket/sga23/msc_project/models/VisualCheXbert/model_path/checkpoint/visualCheXbert.pth"
csv_path = "/vol/bitbucket/sga23/msc_project/data_msc_project/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv"
list_path = "/vol/bitbucket/sga23/msc_project/data_msc_project/VisualCheXbert/jayson"  # JSON file path

# Initialize your model
model = bert_labeler()  # Ensure this matches how the model was initialized during training

# Prepare your test DataLoader
test_dataset = ImpressionsDataset(csv_path, list_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_labels)  # Use custom collate_fn_with_labels

# Call the test function
test(model, checkpoint_path, test_loader)