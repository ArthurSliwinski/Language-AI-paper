# Language-AI-paper
Code for the paper assignment of the course Language &amp; AI TUe Eindhoven

Linguistic Feature Analysis and Classification

This is the code that was used for research into Author profiling of reddit users.
The goal of the research was mainly to discover indicator tokens in NLP models and whether they were "contaminating" or not. 
When implementing models for the use of author proifiling, ideally one would like to achieve reasonable accuracy,
without the actual text already giving away the to be estimated properties. 
The paper related to this project with the same title mentioned above, explains the experimental setup and discusses results and conclusions
in further detail. This README file's main purpose is to ensure reporducability of the results by going through how the code should be executed. 


The full code contains three main files of code

baseline.py ----- This file contains the implementation of the baseline models and corresponding evaluations
data_loader_2.py ----- This file loads in the data and tokenizes the words in the dataset preparing it as input of the LR model
model.py ----- This code loads in the previously manipulated data, and then trains and evaluates a LR model
visualization.py 


Data

The data comes from the SOBR corpus paper referenced at the bottom of this file. 
More specifically the data used is the posts with labels on political leaning (on the economic scale). It is saved as a csv file.
The csv contains three "columns" author_ID, post, political_leaning, representing the author name, the post content (string), and label (string).
A more detailed description can be found in the SOBR corpus, or the paper corresponding to this project. 
The data used is not disclosed in the GitHUb repository because it is used under license of the researchers that build the SOBR corpus. 


Project prerequisities:

This project was run in PyCharm an IDE by Jetbrains. Furthermore the main interperter used was Python version 3.12
it should also work in other IDE's and interpreters of python with version >= 3.9

Before running the code it is important to install the packages mentioned in requirements.txt 
One specific important note is that the version of numpy must be <2.0.0 due to dependency issues that play up otherwise

Finally the code requires your own path to the csv file containing the data, so be sure to have this at the ready, or to place it within the project folder.


File Nr 1:  baseline.py
After having installed all the requirements in the requirements file, and having dowloaded the data, the baseline file should be easy to use. 
In line 21, pd.read_csv() requires the executer of the code to fill in their own path to the csv file. The code contains comments explaining the functionailty. 
The file should run in one go and present results. 


File Nr 2: data_loader_2.py 
This code also requires the user to fill in their own path to the data, this time in line 17,
once again in the pd.read_csv()

From there on the process is pretty straightforward, but there are some steps. 

One should execute the data_loader_2 first, without changing anything apart form the path. Next the model file can be run. 
Later on, when modifying the model, in this case to remove contaminating tokens. A specific part of the preprocess_text section should be uncommented. 
The section to uncomment/modify has been indicated by comments in the code file. 

The code also includes comments that help explain the functionality of the code throughout. 


File Nr 3: model.py
In this file it is not necessary to add a path as the data has been loaded already. It is important to understand that the file will rerun the data_loader_2 file, due to imported variables. 
It is however important to run the data_loader_2.py beforehand. 

The model file can then be run, and it will return performance metrics of the model. 

If one wishes to save the top n indicator tokens in a sorted excel file, this is possible and clearly explained in comments at the end of the file. 
All it takes is some simple uncommenting, and possibly a parameter change (n). 

The code also contains comments explaining the functionality of the model and code, to once again aid reproducability.


File Nr 4: visualization.py
This file is straightforward, it can be run to produce the visualizations used in the paper.  


Final notes
It is recommended to execute this code in a well structured IDE, preferably PyCharm. 
The data source, and inspiration for the methodology can be found in the following paper. 

reference:
Chris Emmery, Marilù Miotto, Sergey Kramp, and
Bennett Kleinberg. 2024. SOBR: A corpus for sty-
lometry, obfuscation, and bias on Reddit. In Pro-
ceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources
and Evaluation (LREC-COLING 2024), pages 14967–
14983, Torino, Italia. ELRA and ICCL.
