import pandas
import glob
import os

COMBINED_NAME = '\\parameters.csv'
FILE_PATH = '\\training-data\\gaussian'  # relative to this file
GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__)) + FILE_PATH

os.chdir(GLOBAL_PATH)

files = [f for f in glob.glob('*.csv')]

df = pandas.concat([pandas.read_csv(csv) for csv in files])
df.to_csv(path_or_buf=GLOBAL_PATH, index=False, encoding='utf-8')


