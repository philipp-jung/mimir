# Garf-master

No experiment in the dictionary requires the presence of 4 datasets in the database, take Hosp_rules as an example (path_pos = Hosp_rules_copy in config.ini, change here if you change the dataset)
Hosp_rules is the initial clean dataset
Hosp_rules_copy is the version with the error data added, which is also our target dataset to be fixed, initially empty, copy the blueprint and add the error by insert_error.py
Hosp_rules_copy1 is the error data taken out separately when adding the error data, initially empty
Hosp_rules_copy2 is the real correct data corresponding to the added error data, initially empty

Caution:
The dataset Hosp_rules is not involved in the detection and repair process, but only serves as a data blueprint for result evaluation. The relevant code is only valid for insert_error.py, reset.py and eva.py
The datasets Hosp_rules_copy1 and Hosp_rules_copy2 are generated only when the error is generated, for the convenience of cross-checking, and are not relevant to the program. "path3" in insert_error.py
Add your own dataset need to add Label column in the last column, but all empty can be used only in eva.py for results evaluation, but because the code process contains the impact of removing the Label column, the lack of results will affect or report errors

This code has been modularized and split, default one-way training and save the model results, please run at least once in the forward direction and once in the reverse direction when you actually use it, multiple runs can improve the performance results a little.
In main.py, order = 1 means forward; order = 0 means reverse, please do not re-add error data in the second run, please comment out insert_error(path_ori, path, error_rate)
insert_error.py is used to add error, error contains 3 categories: spelling error, missing data, random replacement of other values under the same attribute column, if not needed, comment out insert_error(path_ori, path, error_rate) as well.
Increasing the values of g_pre_epochs and d_pre_epochs in config.ini (i.e., the number of iterations of the model generator and discriminator) can improve the performance a little, but at a higher time cost.

Expected results:
Test dataset with 10k data items, Hosp dataset results in accuracy of 98% ± 1% and recall of 65% ± 3%; Food dataset results in accuracy of 97% ± 2% and recall of 62% ± 5%
With the increase of data volume, the model performance is improved, and the amount of Hosp data in the paper is 100k, and the amount of Food data is 200k
For additional data, please click the following link:


Hosp：https://data.medicare.gov/data/physician-compare  
Food：https://data.cityofchicago.org  -> https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5

Flight：http://lunadong.com/fusionDataSets.htm  
UIS：https://www.cs.utexas.edu/users/ml/riddle/data.html  




