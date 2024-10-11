# NBS Dashboard

## Name
NBS Dashboard

## Description
An application that simulates nature-based solutions.

## Installation

- Install VSCode on Windows
- Install WSL (use Ubuntu 22.04.4 LTS). You can use the Microsoft store to install it.
- Open a Linux terminal
- Install miniforge on Linux for conda (https://github.com/conda-forge/miniforge)

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

- Create and activate a conda environment called nbs-dashboard

```
conda create --name nbs-dashboard python=3.10.4
conda activate nbs-dashboard
```

- Install the Python libraries to that conda environment

```
pip install -r requirements.txt
```

- Install common data files

```
git clone --depth 1 --branch v1.0.0 git@git.wur.nl:FoodInformatics/nature-based-solutions/nbs-dashboard-common-data.git ./common_data
```

## Usage

- Change to the project folder and then open the project with vscode

```
cd nbs-dashboard
code .
```

- Select the correct interpreter in VSCode. Press Ctl-shift-P, and pick "Python: Select Interpreter". Then choose intepreter associated with the nbs-dashboard conda environment.

- To run, click the "Run and Debug" icon to the left and then click the green Play button for "Run App".

- The server will be launched. Copy the link to the launched server and open it in a web browser.

## Support
For assistance, please contact Confidence Duku (confidence.duku@wur.nl)

## Contributing
No contributions are currently accepted.

## Authors and acknowledgment
See CITATION.cff file.

## License
Unknown