# pipeline_2p
Preprocessing pipeline for 2p imaging data, VR behaviour data and synchronization between the two.

## Setup

The following instruction details how to setup the pipeline on the cn121 cluster, where data can be preprocessed directly from the network storage.

### Setting up caiman and the pipeline

Follow these steps if you want to install the pipeline in your own user folder. An installation is already available on the global scratch floder, go skip this section if you want to use that.

- Install mamba locally
- Create a virtual environment and install caiman
- Clone the repository and install the dependencies

### Accessing the server remotely with jupyter lab

- Log in into the server
- Open a new screen
- Activate the caiman virtual environment
- Open a jupyter rever with the --no-browser option, specifying the port
- Open a new terminal in your local machine and run ssh -L 8080:localhost:8080 your_sicence_loging@cn121.science.ru.nl
- you can now access the jupyter lab environment at http://localhost:8080/


## Usage 

### Session overview and parameter selection
    
Use `session_overview.ipynb` to interactively inspect the raw data (2p video and log files) and apply the full pipeline to the session in isolation. The pipeline application will save all data in a temporary folder. You can use the same notebook to visualize and inspect the result of the preprocessing and tweak the parameters of the preprocessing pipeline. Once you are satisfied with the parameter set, the notebook provides code to save it for future usage and to delete the temporary folder to free memory.
    
### Batch preprocessing 

2. Use preprocess_batch.py to run the prerocessing pipeline on a provided list of session sequentially. Only final outputs will be saved for subsequent analysis, temporary files are deleted for memory management.

## Structure

- logreader contains the module for the preprocessing of the log file from the VR system.
- pipeline contatins the module for preprocessing the 2p imagin g videos usign CaImAn
- notebooks contains a collection of notebooks for particular analysis and visualization
- scripts contains scripts used to run particular analysis
- utils contains utility functions (now mostly empty)
