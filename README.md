# plasma-cell-detection

This is the code repository for the paper [*Deep learning accurately quantifies plasma cell percentages on CD138-stained bone marrow samples (2022)*](https://www.sciencedirect.com/science/article/pii/S2153353922000116). It contains annotated training data (images and labels), a trained model, and a web application for using the model to evaluate plasma cell percentages in example microscopy images. See the paper (open-access) for more details.

## Running the web application

To use the web interface, you will need to set up a Python environment with the required dependencies on a computer with sufficient disk space and CPU/GPU capabilities. The following instructions describe a method to do so by using the conda package manager on Windows. 

1. Install the latest version of Miniconda at this link (this should by default install to a folder under C:\Users\<your-user> so as to not require administrator permissions): https://docs.conda.io/en/latest/miniconda.html
2. Download the latest version of this repository in GitHub (click the green button and select "Download as ZIP") and extract its contents to a known location, e.g. C:\plasma-cell-detection for simplicity.
3. From the Start Menu, open the program Anaconda Prompt (Miniconda 3).
4. Type `cd C:\plasma-cell-detection` and press enter to navigate to that directory.
5. Create a new conda environment, run: `conda create --name dlweb --file conda-env-win-<cpu/gpu>.txt`, replacing `<cpu/gpu>` with `gpu` if you have a compatible NVIDIA GPU (CUDA >= 11.7), otherwise with `cpu`. This will install the required dependencies into a new conda environment called dlweb (if an environment with the same name already exists, it will prompt you to remove it first; you can also adjust this name as desired).
6. Type `conda activate dlweb` and press enter. The prompt should now start with (dlweb).
7. Type `flask run` and press enter to start the web app. Note on initial run, the dependent pre-trained PyTorch VGG models will be downloaded, which may take some time.
8. The web app should start in your browser, or you can access it at http://127.0.0.1:5000/ in a web browser. Follow the instructions on the app to proceed.

## Example screenshot

This screenshot shows output from uploading the image in `microscope/Image5 (138-029A).tif` to the web interface.

![](screenshots/web_app.png?raw=true)
