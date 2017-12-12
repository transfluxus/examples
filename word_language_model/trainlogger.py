import time
import os
import re
import csv


base_folder = "logs/"

BASELINE_FILE = 'baseline.txt'

MODEL_FILE_ENDING = '.model.txt'
LOGS_FILE_ENDING = '.csv'
PLOT_FILE_ENDING = '.png'

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_logs(experiment_name, model_name, only_data=True):
    logs_file = base_folder + experiment_name + '/' + model_name + ".csv"
    # print([float(l.split(',')[2]) for l in open(logs_file).read().split('\n')[1:3]])
    if os.path.isfile(logs_file):
        with open(logs_file, newline='') as csvfile:
            log_reader = csv.reader(csvfile, delimiter=',')
            rows = list(row for row in log_reader)[1:]
            return [float(l[2]) for l in rows]
    else:
        print("You are looking for logs that don't exist: %s" % model_name)
        return []


def rename_model_files(experiment_name, model_name, new_model_name):
    old_base_name = base_folder + experiment_name + '/' + model_name
    new_base_name = base_folder + experiment_name + '/' + new_model_name

    for file_type in [MODEL_FILE_ENDING,LOGS_FILE_ENDING,PLOT_FILE_ENDING]:
        if os.path.isfile(old_base_name + file_type):
            os.rename(old_base_name + file_type, new_base_name + file_type)
        else:
            print('no existence > no renaming %s' % old_base_name + file_type)


class TrainLogger:
    def __init__(self, experiment_name, model, model_name, autostart = True, log_architecture=True):
        self.start_time = None
        self.experiment_name = experiment_name
        self.logs = []  # tuple 1. epoch, seconds, data (evtl. list)
        self.data_names = []

        self.experiment_name = experiment_name
        self.experiment_folder = base_folder + experiment_name + '/'
        make_dir(self.experiment_folder)
        self.model_name = model_name + "_" + self.get_model_name_id(model_name)
        print("logging as %s" % self.model_name)

        self.log_architecture = log_architecture
        self.model = model
        self.started = False
        if autostart:
            self.start_logging()

    def get_model_name_id(self, model_name):
        model_pattern = re.compile(model_name + '_\d+\.model\.txt')
        id = len([m for m in os.listdir(self.experiment_folder) if model_pattern.search(m)])
        while os.path.isfile(self.experiment_folder + model_name + '_' + str(id) + MODEL_FILE_ENDING):
            id += 1
        return str(id)

    def time_passed(self):
        now = time.time()
        return int(now - self.start_time)  # in seconds

    def path(self, model_name=None):
        if not model_name:
            model_name = self.model_name
        return self.experiment_folder + model_name

    def save_model_architecture(self, model, additional_parameter=[]):
        """
        additionalParams is a list of tuples: name, value

        :param additional_parameter:
        :return:
        """
        with open(self.path() + MODEL_FILE_ENDING, 'w') as fout:
            fout.write(str(model) + '\n')
            for param in additional_parameter:
                fout.write(param[0] + ": " + str(param[1]) + '\n')

    def start_logging(self):
        if not self.started:
            self.start_time = time.time()
            self.started = True

    def log(self, epoch, data):
        if self.log_architecture:
            self.save_model_architecture(self.model)
        if not self.started:
            self.start_logging()
            print("Late start. Call start_logging before training!")
        self.logs.append((epoch, self.time_passed(), data))

    def end_logging(self, plot=True):
        if not self.started:
            print("You didn't even start...")
            return
        self.started = False
        with open(self.path() + ".csv", 'w') as fout:
            csv_out = csv.writer(fout)
            csv_out.writerow(['epoch', 'seconds', 'loss'])
            for log in self.logs:
                csv_out.writerow(log)
        if plot:
            self.plot()

    def set_baseline(self):
        with open(self.experiment_folder + BASELINE_FILE, 'w') as fout:
            fout.write(self.model_name)

    def read_logs(self, model_name, only_data=True):
        return read_logs(self.experiment_folder, model_name, only_data)

    def get_baseline_model_name(self):
        if not os.path.isfile(self.experiment_folder + BASELINE_FILE):
            return ""
        else:
            return open(self.experiment_folder + BASELINE_FILE).read()

    def load(self):
        self.logs = self.read_logs(self.model_name, False)

    def plot(self, compare_with_baseline=True, compare_with=[]):
        from torchy.LogPlotter import plot_logs
        all_plot_models = [self.model_name] + compare_with
        if compare_with_baseline:
            baseline = self.get_baseline_model_name()
            if baseline:
                all_plot_models.append(baseline)
        plot_logs(self.experiment_name, all_plot_models, self.model_name)

