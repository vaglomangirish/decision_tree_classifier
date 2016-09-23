# Decision Tree Classifier

This is the code repository for programming assignment to program a decision tree classifier as a part of
B659 Applied Machine Learning course at Indiana University Bloomington.

## Prerequisites
- Python 2.7
- Microsoft Excel

## How to run the code?
- In the main project directory run # python main.py
- The program should display the decision tree for various datasets, Monk 1,2,3 as well as the Custom Data set, Cars.
- The program should also output the Accuracy and Confusion matrices for various runs.

## Other notes
- The program would work only for binary classification (1/0) and the dataset should have the first column as the class.
- The Accuracy graph plots for various datasets versus their depths can be found in results directory,
GraphPlot_AML_PA1.xlsx file.
- create_decision_tree(self, data_set, max_depth) function in learn_tree.py is the injection point for the algorithm
that constructs decision tree by taking in data set and max depth as inputs.
- Data sets can be found and are picked up by the program from the 'datasets' directory.

## Some comments about the results
- Acuracy plot for Monk-1 shows the expected behaviour of accuracy increasing with depth initially and then decreasing.
- The results for the own data set (cars) show a bias and hence the accuracy and confusion matrix remains constant
with depth.