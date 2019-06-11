(In progress)
# Recommendation system evaluation for collaborative filtering, content-based & hybrid approaches
Bachelor thesis about recommendation system. In this thesis a deep evaluation of a collaborative filtering method, conten-based method and hybrid approach has been carry out.

* For the **Collaborative filtering** method, a _Matrix factorization approach_ was evaluated using [_implicit.py_](implicit.readthedocs.io).
* For the **Content-based** method, a simple class for _tf-idf_ recommendations was built using [_sklearn_](scikit-learn.org).
* The **hybrid** approach just combines the results of collaborative filtering and content based methods by mixing them.





## Repository contents

* [Plots/](https://github.com/rooom13/recommendation-system-thesis/tree/master/Plots): Folder containing plots for dataset visualization.
* [collaborative_filtering/](https://github.com/rooom13/recommendation-system-thesis/tree/master/collaborative_filtering):
* content_based
* [data_visualization/](https://github.com/rooom13/recommendation-system-thesis/tree/master/data_visualization): Scripts for reading the results and obtaining metrics.
* fake_dataset
* results
* resultsBackup
* [_ReadSave.py_](https://github.com/rooom13/recommendation-system-thesis/tree/master/ReadSave.py): Simpler _.pkl_ object read/saver.
* backup.pkl
* [_evaluate.py_](https://github.com/rooom13/recommendation-system-thesis/tree/master/evaluate.py): Main loop of evaluation of the three methods.
* [_get_dataset.py_](https://github.com/rooom13/recommendation-system-thesis/tree/master/get_dataset.py): Script for download & extract the dataset.
* [_main.py_](https://github.com/rooom13/recommendation-system-thesis/tree/master/main.py): "Control panel" script for choosing options for the evaluation (which metrics, methods, randomize fold...)
* [_metrics.py_](https://github.com/rooom13/recommendation-system-thesis/tree/master/_metrics.py): Ranking metrics implementations from this [_Gist_](https://gist.github.com/bwhite/3726239).
