# anomalyDetection

A class that removes anomalies from a set of logs, scoring registers through an Isolation Forest. Given a list of categorical features, the train method takes the top k values per feature and generate a binary variable (1 if it is present, 0 if not) for each value. The predict method allows a percentile as a parameter.
