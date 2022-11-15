#Import useful libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read covid-19 data
data = pd.read_csv("C:/Users/ZUBBY/Desktop/python jobs/owid-covid-data.csv")

#read first five rows in the data
data.head()

#get basic statistics and information about the data
data.describe()

#select wanted columns
df1 = data[['continent','location','date', 'total_cases','total_deaths', 'human_development_index', 'population']]

df1.head()

#convert the data column to datetime datatype
df1['date']= df1['date'].astype('datetime64[ns]')


#extract year and month from the date column  
df1['year'] = df1['date'].dt.year
df1['month'] = df1['date'].dt.month

#group df1 into continents and get the average cases for each continent per month
continent_avg = df1.groupby(['continent', 'month'])['total_cases'].mean().reset_index()


#rename "total_cases" to "average_cases"
continent_avg.rename(columns ={'total_cases': 'average_cases'}, inplace = True)



def line_chart(data, x, y, legend, title):
    """The function plots line graph

    Args:
        data (dataframe): dataframe that holds the value to be plotted
        x (array): the column from data to be at the x-axis
        y (array): The data column at the y_axis
        legend (list): chart legend
        title (string): chart title
    """   
    fig, ax = plt.subplots(figsize=(12, 8),dpi =150)

    for value in data[legend].unique():
        data[data[legend] == value].plot(x=x, y=y, ax=ax, label=value)
    plt.legend(loc='upper right', fontsize = 7)
    plt.xticks(data[x].unique())
    plt.ylabel(y)
    plt.title(title)
    plt.show()

#plot line chart for continent_avg data
line_chart(continent_avg, 'month', 'average_cases' , 'continent','Average Covid-19 Cases Per Month (2020 - 2022)')


#group df1 into continents and find the total cases and total deaths per year
death_total = df1.groupby(['continent', 'year'])[['total_cases','total_deaths',]].sum().reset_index()


cases_and_deaths = death_total.groupby('continent')[['total_cases', 'total_deaths']].sum().reset_index()



def bar_plot(x, y, label, title):
    """This function plots grouped bar chart

    Args:
        x (array): first group in the grouped bar chart
        y (array): second group in the grouped bar chart
        label (list): list of strings to be used as label
        title (String): chart title
    """    
    label = label
    x_axis= x
    y_axis = y

    w = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8), dpi=90)
    rects1 = ax.bar(w - width/2, x_axis, width, label=x_axis.name)
    rects2 = ax.bar(w + width/2, y_axis, width, label=y_axis.name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.title(title, fontsize = 20)
    plt.xticks(w,labels, fontsize =14)
    plt.legend()
    ax.set_yscale('log')

    fig.tight_layout()
 
    plt.show() 

#data needed for bar graph
labels = list(cases_and_deaths['continent'].unique())
x = cases_and_deaths['total_cases']
y = cases_and_deaths['total_deaths']

#plot bar graph forcases_and_deaths data
bar_plot(x,y,labels, "Total Cases and Death Cases Per Continent")



def scatter_plot(x, y,title):
    """ The function plots a scatter plot

    Args:
        x (array): values on x-axis
        y (array): values on y-axis
        title (string): chart title
    """        
    fig, ax = plt.subplots(figsize=(16, 12), dpi=200)

    plt.title(title, fontsize = 20)
    plt.xlabel(x.name, fontsize = 16)
    plt.ylabel(y.name, fontsize = 16)
    plt.scatter( x, y,color ="blue")
    ax.set_yscale('log')
  
    plt.show()
    



death_dev = df1.groupby('location')[['total_deaths', 'human_development_index']].mean().reset_index()


#plot scatter plot for death_dev data
scatter_plot(death_dev['human_development_index'], death_dev['total_deaths'], "Relationship Between Death and Human Development Index")

