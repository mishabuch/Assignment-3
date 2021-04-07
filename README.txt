Unsupervised Learning - abuch6 Amisha Buch

Before running the project, you will need to install all the requirements from the requirements.txt file.
This project runs in python 3.

Folder structure
- You will need to create two folders,diabetes and phishing for the output csv files as well as the plots.
- The datasets folder contains the two datasets used for this project.

Running the algorithms
- The project has 2 main files - diabetes_experiment.py and phishing_experiment. Run each of these files independently
to run both of these experiments.
- Running these files will create output csv and phots in the diabetes/phishing folders.
- But before you run the files, make sure you give a local path in the 'dir' variable of your own machine.

Other files
- common_utils.py has all the code for plotting graphs
- dataset_loader.py loads and pre-processes datasets.
- clustering.py, pca.py, ica.py, rp.py, rfp.py and neural_network.py will be automatically run from the experiment files.

All the code for this project can be found at:
https://github.com/mishabuch/Assignment-3

REFERENCES:
The code for plotting various graphs and plots was written with the help of examples from:
https://github.com/bryanmarthin/Gatech-CS-7641-Assignment-3
https://github.com/huhu42/Unsupervised-Learning
https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1

Other references:
https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
https://machinelearningmastery.com/expectation-maximization-em-algorithm/
http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/
Wikipedia sources for various algorithms