# plasma-cell-detection

This is the code repository for the manuscript "*Deep learning accurately quantifies plasma cell percentages on CD138-stained bone marrow samples*" (in submission). It contains annotated training data (images and labels), a trained model, and a web application for using the model to evaluate plasma cell percentages in example microscopy images. See manuscript for more details.

## Running the web application

To use the web interface, you will need to set up a Python environment with the required dependencies on a computer with sufficient disk space and CPU/GPU capabilities. The following instructions describe a method to do so by using the conda package manager on Windows. 

1. Install the latest version of Miniconda at this link (this should by default install to a folder under C:\Users\<your-user> so as to not require administrator permissions): https://docs.conda.io/en/latest/miniconda.html
2. Download the latest version of this repository in GitHub (click the green button and select "Download as ZIP") and extract its contents to a known location, e.g. C:\plasma-cell-detection for simplicity.
3. From the Start Menu, open the program Anaconda Prompt (Miniconda 3).
4. Type `cd C:\plasma-cell-detection` and press enter to navigate to that directory.
5. Create a new conda environment `conda env create -f environment.yml`. This will install the required dependencies into a new conda environment called dlweb.
6. Type `conda activate dlweb` and press enter. The prompt should indicate (dlweb).
7. Run the application using the following two commands: `set FLASK_APP=app.py`, then `flask run`. Note on initial run, the dependent pre-trained PyTorch VGG models will be downloaded, which may take some time.
8. The web application should start in your browser, or you can access it at http://127.0.0.1:5000/. In Anaconda Prompt, the log should also display `device: cuda:0` if the GPU was detected for best performance. Follow the instructions on the screen to proceed.

## Example screenshot

This screenshot shows output from uploading the image in `microscope/Image5 (138-029A).tif` to the web interface.

![](screenshots/web_app.png?raw=true)
