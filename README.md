(In progress)
# Recommendation system evaluation for collaborative filtering, content-based & hybrid approaches
Bachelor thesis about recommendation system. In this thesis a deep evaluation of a collaborative filtering method, conten-based method and hybrid approach has been carry out.

* For the **Collaborative filtering** method, a _Matrix factorization approach_ was evaluated using [_implicit.py_](implicit.readthedocs.io).
* For the **Content-based** method, a simple class for _tf-idf_ recommendations was built using _sklearn_.
* The **hybrid** approach just combines the results of collaborative filtering and content based methods by mixing them.





## Repository contents
* Plots: Folder containing plots for dataset visualization.
* collaborative_filtering
* content_based
* data_visualization: Scripts for reading the results and obtaining metrics.
* fake_dataset
* results
* resultsBackup
* _ReadSave.py_: Simpler _.pkl_ object read/saver.
* backup.pkl
* _evaluate.py_: Main loop of evaluation of the three methods.
* _get_dataset.py_: Script for download & extract the dataset.
* _main.py_: "Control panel" script for choosing options for the evaluation (which metrics, methods, randomize fold...)
* _metrics.py_: Ranking metrics implementations from this [_Gist_](https://gist.github.com/bwhite/3726239).
