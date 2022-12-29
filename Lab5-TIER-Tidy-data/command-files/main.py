import pandas as pd

df = pd.read_csv('billboard.csv', encoding_errors='ignore')

if __name__ == '__main__':
    print(df)
