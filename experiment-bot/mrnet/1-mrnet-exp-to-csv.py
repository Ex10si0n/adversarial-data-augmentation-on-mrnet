import sys
import os
import re
import csv
import settings

def gen_csv(path, filename):
    folders = os.listdir(path)
    folders = [f for f in folders if re.search(r'2022|baseline', f)]

    classifier_header = ['task', 'dimension', 'eps', 'percent']
    result_header = ['LOSS', 'AUC-best', 'Accuracy-best', 'Sensitivity-best', 'Specifity-best']
    result_rows = []
    header = classifier_header + result_header

    for f in folders:
        eps, percent = 0, 0
        try:
            eps, percent = f.split('-')[-2:]
            # print('find experiment: eps = {}, percent = {}'.format(eps, percent))
            os.chdir(os.path.join(path, f, 'results'))
            files = os.listdir()
            for file in files:
                task, dimension = file.split('.')[0].split('-')[0].split('_')[-2:]
                # print('find result: task = {}, dimension = {}'.format(task, dimension))

                with open(file, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        pass
                    result = row
                    classifier = [task, dimension, eps, percent]
                    result_row = classifier + result
                    result_rows.append(result_row)

        except:
            pass

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    with open(settings.working_folder + filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in result_rows:
            writer.writerow(row)
        print('write {} results to results-temp.csv'.format(len(result_rows)))

if __name__ == '__main__':
    os.system('rm -f results-temp.csv')
    os.system('cp baseline.csv results-mr.csv')

    old_path = '/Users/ex10si0n/Projects/research/mrnet/experiments/old/'
    new_path = '/Users/ex10si0n/Projects/research/mrnet/experiments/'

    gen_csv(new_path, settings.csv_file)
    gen_csv(old_path, settings.csv_file)

