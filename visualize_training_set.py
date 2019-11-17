'''
    read in the training data JSON files and create a single HTML file that shows the input and 
    output grid for each training and test example.

    This will give researchers a better overview of the types and complexity of the training examples

    Author(s):
        Carl Anderson (carl.anderson@weightwatchers.com)
'''

import glob
import json

filenames = [f for f in glob.glob("data/training/*.json")]

with open("training_template.html", "r+") as f:
    template = f.read()

new_data = ""

def generate_cell(val, x_idx, y_idx, sz):
    return "<div class=\"cell symbol_%s\" x=\"%s\" y=\"%s\" symbol=\"%s\" style=\"height: %spx; width: %spx;\"></div>" % (val, x_idx , y_idx , val, sz, sz)

def generate_grid(data):
    s = "<div class='input_preview'>"
    for i, row in enumerate(data):
        s += "<div class='row'>"
        for j, cell in enumerate(list(row)):
            sz = str(int(100.0 / len(row)))
            s += generate_cell(str(cell), str(i), str(j), str(sz))
        s += "</div>"
    s += "</div>"
    return s

def generate_rows(filename, data, type):
    rows = ""
    for i , example in enumerate(data):
        row = "<tr class='spaceUnder'>"
        row += "<td>" + filename + "</td>"
        row += "<td>" + type + "</td>"
        row += "<td>" + str(i) + "</td>"
        row += "<td>" + generate_grid(example['input']) + "</td>"
        row += "<td>" + generate_grid(example['output']) + "</td>"
        row += "</tr>"
        rows += row
    return rows

for filename in filenames:
    with open(filename, "r+") as jsonFile:
        data = json.load(jsonFile)
        train = data["train"]
        test = data["test"]
        new_data += generate_rows(filename, train, "train")
        new_data += generate_rows(filename, test, "test")

template = template.replace("__TRAINING_DATA__", new_data)

with open("training_data.html", "w") as f:
    f.write(template)
