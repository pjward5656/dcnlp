# DC_nlp
Drug overdose classification from death certificates using NLP

## Getting started

This page contains a python script for analyzing free-text death certificates for drug overdose classification purposes.

### Prerequisites

This code was developed using `Anaconda`. Download and install `Anaconda` to run this code.

Running this code also requires installing the following `python` libraries:

```
pandas
spaCy
sklearn
scipy
numpy
itertools
matplotlib
pickle
```

Installing `Anaconda` should automatically pre-install these libraries. `en_web_core_sm` from `spaCy` may need to be installed separately. This code was developed using `sklearn` version 0.19.2. 

## Using the code

To run this code, create a .csv file with three fields from your death certificate data: 
  1. "DC_YEAR", the year of death
  2. "OD", a 0/1 indicator if a record is an overdose death or not
  3. "scan_field", a field consisting of the free-text fields from the death certificates 
     that you wish to use for classification, with all punctuation removed, in all caps

Local file paths in the example file will need to be changed as you use the code.
