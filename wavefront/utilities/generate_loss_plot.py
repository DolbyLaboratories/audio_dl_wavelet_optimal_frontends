"""
Copyright (c) 2024 Dolby Laboratories

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
import json
import matplotlib.pyplot as plt
import itertools

# Directory containing the .json files
directory = "./Logs/Accuracies"


# Function to read and parse JSON files
def read_json_files(directory):
    data = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            version = filename.split(".json")[0]
            with open(os.path.join(directory, filename), "r") as file:
                data[version] = json.load(file)
    return data


# Function to extract steps and loss values
def extract_data(data):
    epoch = {
        "1.183": 1350,
        "1.131": 2200,
        "1.132": 2200,
        "1.134": 2200,
        "1.185": 1360,
    }

    steps = {}
    for version, records in data.items():
        step_list = []
        loss_list = []
        for record in records:
            timestamp, step, loss = record
            step_list.append(step)
            loss_list.append(loss)

        step_list = [(x + 1) / epoch[version.split("-2")[0]] for x in step_list]
        steps[version] = (step_list, loss_list)
    return steps


# Function to plot the data
def plot_data(steps):
    descrip = {
        "1.183": "SincNet Ac. Scenes optim",
        "1.131": "SincNet Speak Id.",
        "1.132": "SincNet Ac. Scenes",
        "1.134": "MFCC Speak. Id.",
        "1.185": "MFCC Ac. Scenes optim",
    }
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:cyan",
        "tab:olive",
    ]
    colors_acc = ["tab:gray", "tab:orange", "tab:olive", "tab:red", "tab:brown"]

    plt.figure(figsize=(12, 8))
    i = 0
    for version, (step_list, loss_list) in steps.items():
        if "-2" in version:
            version = version.split("-2")[0]
            label = f"{descrip[version]}"
        else:
            label = f"{descrip[version]}"

        # if version != "1.134" and version != "1.131":
        step_list_cut = list(itertools.takewhile(lambda x: x < 100, step_list))
        plt.plot(
            step_list_cut,
            loss_list[: len(step_list_cut)],
            label=label,
            color=colors_acc[i],
        )
        i = i + 1

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Sentence Accuracy Over First 100 Epochs"
    )  # Training and Validation Loss Evolution Over First 100 Epochs.\nAcoustic scenes classification
    plt.legend(fontsize=11, loc="lower right")
    plt.show()


# Main function to run the process
def main(directory):
    data = read_json_files(directory)
    steps = extract_data(data)
    plot_data(steps)


# Run the main function
main(directory)
