import seaborn as sns
import matplotlib.pyplot as plt

#Explore data

def explore_data(diamonds):
    """
    Explores data and its structure
    :return: summary of data
    """
    print("The first five rows of the dataset look like this:")
    print(diamonds.head())
    print("Now checking for missing values")
    print(diamonds.isnull().any())
    print("The types of the data in columns are:")
    print(diamonds.info())
    print("The shape of the dataset is:")
    print(diamonds.shape)

def barplot_against_price(var, diamonds):
    """
    plots a variable against the price in a barplot
    :param var:
    :return:
    """
    plt.figure(figsize=(6, 2))
    return sns.barplot(x=var, y="price", data=diamonds)

def histogram(var,name_for_x, name_for_y, title):
    """
    :return: histogram
    """
    plt.figure(figsize=(6, 2))
    plt.xlabel(name_for_x)
    plt.ylabel(name_for_y)
    plt.title(title)
    return plt.hist(var, bins=20, color='b')

