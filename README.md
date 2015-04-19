# 10-605
Machine Learning with Large Datasets

Distributed SGD for Matrix Factorization on Spark

The main program is dsgd_mf.py
To run the main program, use the following command:

spark-submit dsgd_mf.py (num_factors) (num_workers) (num_iterations) \
(beta_value) (lambda_value) (inputV_filepath) (outputW_filepath) (outputH_filepath)

