from functions import pltgraph
import gui
import pandas as pd

# Loading the Data Set
iris_df = pd.read_csv('./data/iris.csv')


def main():
    dataplt = pltgraph(iris_df)
    dataplt.get_linepltSLcm()

if __name__ == '__main__':
    main()