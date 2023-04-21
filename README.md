# pipeline_2p
Preprocessing pipeline for 2p imaging data, VR behaviour data and synchronization between the two.

## Usage instruction

1. Use `session_overview.ipynb` to interactively inspect the raw data (2p video and log files) and apply the full pipeline to the session in isolation. The pipeline application will save all data in a temporary folder. You can use the same notebook to visualize and inspect the result of the preprocessing and tweak the parameters of the preprocessing pipeline. Once you are satisfied with the parameter set, the notebook provides code to save it for future usage and to delete the temporary folder to free memory.

2. Use preprocess_batch.py to run the prerocessing pipeline on a provided list of session sequentially. Only final outputs will be saved for subsequent analysis, temporary files are deleted for memory management.
