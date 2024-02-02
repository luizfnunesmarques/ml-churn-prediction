# Predict Customer Churn

## Project Description
Predicts customer churn based on a series of features already defined in the code.
This is a refactor from a jupyter notebook.

## Files and data description
- The library is within churn_library.py
- The tests are run at churn_script_logging_and_tests.py
- The constant file provides names for categorical columns and features used to train the model with
- A sample data file is present at /data
- A skimmed sample data for unit tests is available at /tests/data

## Running Files
#### Docker (recomended)
- A pre-built, secure image can be found at luizfnunesmarques/churn_prediction: `docker pull luizfnunesmarques/churn_predictions:1.0`.
- The tests are the entrypoint of the image i.e. `docker run churn_prediction` will run all the tests.
- The image can be also be used as a host for development by running the container mounted with a local directory: ` docker run -it --rm -v <project_path>:/app churn_predictions /bin/sh`.
- (optional for vscode users) [The dev container extension](https://code.visualstudio.com/docs/devcontainers/containers) is a superb companion when using the pre-built image

##### Local
- The target python version is 3.8.18
- Dependencies can be installed by running pip3 install -r requirements_py3.8.txt
