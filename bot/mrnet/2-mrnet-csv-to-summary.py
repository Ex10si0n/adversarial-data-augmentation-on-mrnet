import csv
import sys
import numpy as np
import xlsxwriter
import settings

class Mart:
    def __init__(self, task, dimension, eps, percent, facts):
        self.task = task
        self.dimension = dimension
        self.eps = eps
        self.percent = percent
        self.params = (eps, percent) 
        self.facts = facts
        self.fact_names = ["LOSS", "AUC-best", "Accuracy-best", "Sensitivity-best", "Specifity-best"]

    def __str__(self):
        return f"{self.task} {self.dimension} {self.eps} {self.percent} {self.facts}"

    def get_facts(self):
        return self.facts

    def get_facts_by_name(self, name):
        return self.facts[self.fact_names.index(name)]

class MartSummary:
    def __init__(self):
        self.marts = []

    def record(self, mart):
        self.marts.append(mart)

    def all_dimensions(self):
        dimensions = list(set([mart.dimension for mart in self.marts]))
        return sorted(dimensions)

    def fact_names(self):
        return self.marts[0].fact_names

    def all_tasks(self):
        tasks = list(set([mart.task for mart in self.marts]))
        return sorted(tasks)

    def all_params(self):
        params = list(set([mart.params for mart in self.marts]))
        return sorted(params, key=lambda x: (x[0], x[1]))

    def query(self, task='all', dimension='all', eps='all', percent='all'):
        result = []
        for mart in self.marts:
            if (task == 'all' or mart.task == task) and \
               (dimension == 'all' or mart.dimension == dimension) and \
               (eps == 'all' or mart.params[0] == eps) and \
               (percent == 'all' or mart.params[1] == percent):
                result.append(mart)
        return result

def write_excel(worksheet, x, y, table):
    for i in range(len(table)):
        for j in range(len(table[i])):
            worksheet.write(x+i, y+j, str(table[i][j]))

if __name__ == '__main__':
    result_lines = []
    header = None
    gen_summary = True

    with open(settings.working_folder + settings.csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            result_lines.append(row)

    print('Read {} lines'.format(len(result_lines)))
    marts = []
    for line in result_lines:
        print("Processing line: {}".format(line))
        try:
            marts.append(Mart(line[0], line[1], line[2], line[3], line[4:]))
        except:
            pass

    summary = MartSummary()
    for mart in marts:
        summary.record(mart)

    # print("All dimensions: ", summary.all_dimensions())
    # print("All tasks: ", summary.all_tasks())
    tables = []
    workbook = xlsxwriter.Workbook(settings.working_folder + settings.xlsx_file)
    all_params = summary.all_params()
    all_eps, all_percent = zip(*all_params)

    all_eps = sorted(list(set(all_eps)))
    all_percent = sorted(list(set(all_percent)))


    table_width, table_height = 4, 4

    for measure in summary.fact_names():
        if gen_summary:
            worksheet = workbook.add_worksheet(measure + '-summary')
            x, y = 0, 0
            for params in summary.all_params():
                y = all_eps.index(params[0]) + 1
                x = all_percent.index(params[1]) + 1
                avg = 0
                for task in summary.all_tasks():
                    for dimension in summary.all_dimensions():
                        for mart in summary.query(task, dimension, params[0], params[1]):
                            try:
                                avg += float(mart.get_facts_by_name(measure))
                            except:
                                print("Error: ", task, dimension, mart, measure)
                                continue
                avg /= 9
                worksheet.write(x, y, avg)

            for eps in all_eps:
                for percent in all_percent:
                    worksheet.write(all_percent.index(percent) + 1, 0, percent)
                    worksheet.write(0, all_eps.index(eps) + 1, eps)

        worksheet = workbook.add_worksheet(measure)
        x, y = 0, 0

        for params in summary.all_params():
            y = all_eps.index(params[0]) * (table_height + 1)
            x = all_percent.index(params[1]) * (table_width + 1)

            template = [[(params[0], params[1]), 'axial', 'coronal', 'sagittal'],\
                        ['abnormal', None, None, None],\
                        ['acl', None, None, None],\
                        ['meniscus', None, None, None]]

            table = np.ndarray(shape=(3, 3))
            for task in summary.all_tasks():
                for dimension in summary.all_dimensions():
                    for mart in summary.query(task, dimension, params[0], params[1]):
                        table[summary.all_tasks().index(task)][summary.all_dimensions().index(dimension)] = mart.get_facts_by_name(measure)

            for i in range(1, 4):
                for j in range(1, 4):
                    template[i][j] = table[i-1][j-1]

            table = np.array(template, dtype=object)
            tables.append(table)
            write_excel(worksheet, x, y, table)

    print('Write table to excel', settings.xlsx_file)

    workbook.close()
