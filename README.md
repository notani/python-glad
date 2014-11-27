# python-glad #
A Python Implementation of GLAD.

## GLAD ##
Algorithm for aggregating labels given by labeler more acculately than majority vote.

* Whitehill, Jacob, et al. "Whose vote should count more: Optimal integration of labels from labelers of unknown expertise." Advances in neural information processing systems. 2009.

## Dependency ##
* numpy
* scipy

## Usage ##
```bash
$ python glad.py data/data.txt
```

## Input Format ##

  First line:
    <numGivenLabels> <numLabelers> <numTasks> <Prior p(Z=1)>
  Following <numGivenLabels> lines:
    <taskId> <labelerId> <label:0|1>

* The task IDs must be integers and must be 0...<numImages-1>.
* The labeler IDs must be integers and must be 0...<numLabelers-1>.
