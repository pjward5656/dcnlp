# dcnlp
Drug overdose classification from death certificates using NLP

To use this code, create a .csv file with three fields from your death certificate data: 
  1. "DC_YEAR", the year of death
  2. "OD", a 0/1 indicator if a record is an overdose death or not
  3. "scan_field", a field consisting of the free-text fields from the death certificates 
     that you wish to use for classification, with all punctuation removed, in all caps

Local file paths in the example file will need to be changed as you use the code.
