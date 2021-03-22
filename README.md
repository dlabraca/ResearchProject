# ResearchProject
Implementations and demonstrations of change point detection methods for my Research Project.

BOCD.py - Bayesian Online Change Point Detection as seen in: Ryan P. Adams, David J.C. MacKay, Bayesian Online Changepoint Detection, arXiv 0710.3742 (2007)

- Code has been built upon the implementation here:https://github.com/hildensia/bayesian_changepoint_detection

- added more Gaussian cases, predictive mean and variance and inference of change point locations by maximum a posteriori estimator of posterior predictive distribution p(r_1, ..., r_{t+1} | x){1:t}).



Htests.py - Nonparametric detection for changes in location and/or scale as seen in: Ross, Gordon & Tasoulis, Dimitris & Adams, Niall. (2012). Nonparametric Monitoring of Data Streams for Changes in Location and Scale. Technometrics. 53. 379-389. 10.1198/TECH.2011.10069. 

- It is essentially a python implementation of the R package CPM: https://cran.r-project.org/web/packages/cpm/cpm.pdf

ThresholdMonte.py - Code to simulate the threshold values used in the detector. The outline of the algorithm can be seen in the paper above by  Ross, Gordon & Tasoulis, Dimitris & Adams, Niall.

Thresholds - Contains some generated thresholds of varying lengths - simulated using 10^5 streams 



Change point detection methods demo.ipynb - Demo of the detectors (also demonstation of offline detection using Ruptures https://centre-borelli.github.io/ruptures-docs/)
