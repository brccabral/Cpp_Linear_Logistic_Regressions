# C++ Linear and Logistic Regressions

Followed videos from **AI Coding** https://www.youtube.com/watch?v=jKtbNvCT8Dc&list=PLNpKaH98va-FJ1YN8oyMQWnR1pKzPu-GI

### Eigen lib
Eigen is a library to help matrix operations.
Need to use v3.2.
Eigen library downloaded from https://eigen.tuxfamily.org/index.php?title=Main_Page  
These are just header files.

### Wine dataset
Wine dataset `wine.data` downloaded from https://archive.ics.uci.edu/ml/datasets/wine  
The larger dataset `winedata.csv` was downloaded from **AI Coding** repository https://github.com/coding-ai/machine_learning_cpp  
To use **AI Coding** Normalize() function we need to remove a few recods from the larger dataset. If you want to keep all records, need to change **AI Coding** original code to use `data.colwise().mean()` instead of variable `mean` due to floating point overflow.

### Adult dataset
The file `adult_data.csv` was downloaded from **AI Coding** repository https://github.com/coding-ai/machine_learning_cpp  