import os
from csv import reader
import threading
from typing import List
import json

from utils.utils import convert_date_to_seconds, convert_json_data_to_dataframe, evaluate, read_json


class RunWithTimeout(object):
    """
    This class is used for running a function in a separate thread with
    predefined timeout.
    """

    def __init__(self, function, args):
        self.function = function
        self.args = args
        self.answer = None

    def worker(self):
        self.answer = self.function(*self.args)

    def run(self, timeout):
        """
        Runs the function with timeout.
        :param timeout: time in seconds
        :return: test
        """
        thread = threading.Thread(target=self.worker)
        thread.start()
        thread.join(timeout)
        return self.answer


def get_from_csv(file):
    """
    Reads from the specified csv file.
    :param file: test
    :return: test
    """
    csv_reader = reader(file)
    _ = next(csv_reader)
    for feature in csv_reader:
        yield feature


def run_live_prediction(file, model):
    """
    Opens a csv file and makes predictions.
    :param file: A csv file
    :param model: Model which makes prediction
    :return: a dictionary with file name, end periods and leakages
    """
    with open(file, 'r') as f:
        print(file)
        previous_prediction = None
        amplitude = []
        end_periods = []
        for feature in get_from_csv(f):
            runner = RunWithTimeout(model.predict, (feature,))
            prediction = runner.run(10)
            if previous_prediction is None:
                previous_prediction = prediction
            elif previous_prediction != prediction:
                amplitude.append(int(previous_prediction))
                end_periods.append(convert_date_to_seconds(feature[0]))
                previous_prediction = prediction
        if len(amplitude) == 0:
            amplitude.append(int(previous_prediction))
            end_periods.append(convert_date_to_seconds('2022-03-08 00:00:00'))
    return {"file_name": os.path.basename(file), "end_periods": end_periods, "leakages": amplitude}


def evaluate_json(prediction_json, true_json):
    """
    Compares two jsons (or dictionaries) i.e. the json with ground truth and predictions
    :param prediction_json: json object where predictions are found
    :param true_json: json object where ground truth is
    :return: metric on how good you predicted
    :raises Exception: If a file is missing in either the prediction json or in ground truth json
    """
    set_one = {p['file_name'] for p in prediction_json['prediction_results']}
    set_two = {p['file_name'] for p in true_json['prediction_results']}

    if len(set_one.union(set_two)) != len(set_one):
        raise Exception("Filenames do not match or there is a missing prediction file")

    true_data = convert_json_data_to_dataframe(true_json)
    predicted_data = convert_json_data_to_dataframe(prediction_json)
    return evaluate(true_data, predicted_data)


def generate_live_predictions(files: List, model) -> dict:
    """
    Takes a list of csv files and prediction model, it generates the prediction json
    :param files: list of csv files
    :param model: prediction model
    :return: json with predictions for each csv file
    """
    return {'prediction_results': [run_live_prediction(file, model) for file in files]}


def evaluate_live(csv_files, true_results_json_file, model):
    """
    A wrapper fo evaluate_json and generate_live_predictions
    :param csv_files: list of csv files
    :param true_results_json_file: json file with ground truth
    :param model: prediction model
    :returns: metric on how good you predicted
    """
    prediction_json = generate_live_predictions(csv_files, model)
    true_json = read_json(true_results_json_file)
    return evaluate_json(prediction_json, true_json)


if __name__ == '__main__':
    from leak_detection import LeakDetection
    #print(evaluate_live(['scenario_week_example_0.csv'], 'solution0.json', LeakDetection()))
    fs = ["test_data\\" + f for f in os.listdir("test_data\\") if f.endswith("csv")]
    #fs = ["test_data\\scenario_week_example_62.csv"]
    f = generate_live_predictions(fs, LeakDetection())
    with open("results.json", "w") as outfile:
        json.dump(f, outfile)
    #print(evaluate_live([,'..\\train_data\\results.json',LeakDetection()))#['data.csv'], 'solution.json', LeakDetection()))
