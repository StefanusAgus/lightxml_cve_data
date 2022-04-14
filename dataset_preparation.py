import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import sklearn
import pickle
import argparse

# this function is used to save the splitted dataset as numpy file
# use this function if the csv files are already splitted into the test and train dataset
def save_splitted_dataset_as_numpy(TRAIN_PATH, TEST_PATH):
    description_fields = ["cve_id", "merged"]
    # Initiate the dataframe containing the CVE ID and its description
    # Change the "merged" field in the description_fields variable to use other text feature such as reference

    # Process the training dataset

    df = pd.read_csv(TRAIN_PATH, usecols=description_fields)
    # Read column names from file
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE

    pd_labels = pd.read_csv(TRAIN_PATH,
                            usecols=[i for i in cols if i not in ["cve_id", "description_text", "cpe_text", "merged"]])
    # Initiate a list which contain the list of labels considered in te dataset
    list_labels = [i for i in cols if i not in ["cve_id", "description_text", "cpe_text", "merged"]]

    # Convert to numpy for splitting
    train = df.to_numpy()
    label_train = pd_labels.to_numpy()

    df_test = pd.read_csv(TEST_PATH, usecols=description_fields)
    pd_labels_test = pd.read_csv(TEST_PATH,
                            usecols=[i for i in cols if i not in ["cve_id", "description_text", "cpe_text", "merged"]])
    test = df_test.to_numpy()
    label_test = pd_labels_test.to_numpy()


    # Save the splitted data to files
    np.save("dataset/splitted/splitted_train_x.npy", train, allow_pickle=True)
    np.save("dataset/splitted/splitted_train_y.npy", label_train, allow_pickle=True)
    np.save("dataset/splitted/splitted_test_x.npy", test, allow_pickle=True)
    np.save("dataset/splitted/splitted_test_y.npy", label_test, allow_pickle=True)
    return "dataset/splitted/splitted_train_x.npy", "dataset/splitted/splitted_train_y.npy", \
           "dataset/splitted/splitted_test_x.npy", "dataset/splitted/splitted_test_y.npy"


# the dataset can be processed through sklearn dump_svmlight file and the TfIdfVectorizer
# This function assume that the splitted dataset is available in dataset/splitted folder
# We also need to create the train/test_labels.txt and train/test_texts.txt
# with each row contains the text and labels for the train/test data
def prepare_lightxml_dataset(TRAIN_CSV_PATH, TEST_CSV_PATH, CVE_LABELS_PATH):
    # Load the splitted dataset files
    train = np.load("dataset/splitted/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("dataset/splitted/splitted_train_y.npy", allow_pickle=True)
    test = np.load("dataset/splitted/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("dataset/splitted/splitted_test_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    cols = list(pd.read_csv(TRAIN_CSV_PATH, nrows=1))
    label_columns = [i for i in cols if i not in ["cve_id", "cleaned", "matchers", "merged"]]
    num_labels = len(label_columns)

    vectorizer = TfidfVectorizer().fit(train_corpus)

    idx_zero_train = np.argwhere(np.all(label_train[..., :] == 0, axis=0))
    idx_zero_test = np.argwhere(np.all(label_test[..., :] == 0, axis=0))

    train_X = vectorizer.transform(train_corpus)
    # train_Y = np.delete(label_train, idx_zero_train, axis=1)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    # test_Y = np.delete(label_test, idx_zero_test, axis=1)
    test_Y = label_test

    num_features = len(vectorizer.get_feature_names())
    num_row_train = train_X.shape[0]
    num_row_test = test_X.shape[0]

    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "dataset/cve_data/train.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "dataset/cve_data/test.txt", multilabel=True)

    train_text = []
    train_label = []
    test_text = []
    test_label = []

    cve_labels = pd.read_csv(CVE_LABELS_PATH)


    train_data = pd.read_csv(TRAIN_CSV_PATH)
    # process the label and text here
    for index, row in train_data.iterrows():
        train_text.append(row.merged.lstrip().rstrip())
        # for label below
        label = cve_labels[cve_labels["cve_id"] == row.cve_id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        # print(label_string)
        train_label.append(label_string)

    test_data = pd.read_csv(TEST_CSV_PATH)
    for index, row in test_data.iterrows():
        test_text.append(row.merged.lstrip().rstrip())
        # for label below
        label = cve_labels[cve_labels["cve_id"] == row.cve_id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        # print(label_string)
        test_label.append(label_string)


    with open("dataset/cve_data/train_texts.txt", "w", encoding="utf-8") as wr:
        for line in train_text:
            wr.write(line + "\n")

    with open("dataset/cve_data/train_labels.txt", "w", encoding="utf-8") as wr:
        for line in train_label:
            wr.write(line + "\n")

    with open("dataset/cve_data/test_texts.txt", "w", encoding="utf-8") as wr:
        for line in test_text:
            wr.write(line + "\n")

    with open("dataset/cve_data/test_labels.txt", "w", encoding="utf-8") as wr:
        for line in test_label:
            wr.write(line + "\n")


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_csv', help='Path to training csv file', default='dataset/dataset_train.csv')
    parser.add_argument('--test_csv', help='Path to test csv file', default='dataset/dataset_test.csv')
    parser.add_argument('--cve_labels_csv', help='Path to CVE-Labels csv file', default='dataset/cve_labels_merged_cleaned.csv')
    args = parser.parse_args()

    TRAIN_PATH = args.training_csv
    TEST_PATH = args.test_csv
    CVE_LABELS_PATH = args.cve_labels_csv
    print("Preparing LightXML dataset from the following:")
    print("Train data: " + TRAIN_PATH)
    print("Test data: " + TEST_PATH)
    print("CVE-Labels data: " + CVE_LABELS_PATH)
    save_splitted_dataset_as_numpy(TRAIN_PATH, TEST_PATH)
    prepare_lightxml_dataset(TRAIN_PATH, TEST_PATH, CVE_LABELS_PATH)
    print("Dataset preparation completed")



if __name__ == '__main__':
    main()