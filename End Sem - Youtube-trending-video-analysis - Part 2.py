#!/usr/bin/env python
# coding: utf-8

# # **Youtube Trending Video EDA**
# > "*Our objective in this exploratory data analysis is to gain insights into the videos that are currently popular on YouTube. We will analyze the trends and patterns of the videos that have been identified as 'trending' on the platform, examining factors such as view counts, publication dates, and channel affiliations. By exploring these data points, we hope to better understand what makes a video successful on YouTube and identify potential trends that can inform content creators and marketers."*
# 

# # **Importing the required libraries**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=False)
import seaborn as sns
import plotly.offline as pyo
from plotly.subplots import make_subplots
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')


# # **Loading the dataset into dataframe**

# In[2]:


df=pd.read_csv('CA_youtube_trending_data.csv')
df.head()


# In[3]:



#loading the category dataset into dataframe to extract the category names, as it is in json format
df1=pd.read_json('CA_category_id.json')
df1


# # **Meta Data:**
# * **Category:** the type of content that the video falls under (e.g. music, comedy, education).
# * **Title:** the title of the video.
# * **Published_at:** the date and time that the video was uploaded to YouTube.
# * **Channel_title:** the name of the channel that uploaded the video.
# * **Trending_date:** the date that the video started trending on YouTube.
# * **Tags:** any tags that were added to the video to help with discovery and search.
# * **View_count:** the number of times that the video has been viewed.
# * **Likes:** the number of likes that the video has received.
# * **Dislikes:** the number of dislikes that the video has received.
# * **Comment_count:** the number of comments that have been posted on the video.
# * **Comments_disabled:** a binary variable indicating whether comments are enabled or disabled on the video.
# * **Ratings_disabled:** a binary variable indicating whether ratings (likes and dislikes) are enabled or disabled on the video.
# * **Publish_month:** the month that the video was uploaded.
# * **Publish_day:** the day of the month that the video was uploaded.

# # **DATA PREPROCESSING**

# > # **Extracting the Category names from json file**

# In[4]:


# create an empty list to store categories
categories = []

# iterate through each item in the 'items' column of the original DataFrame,
#'enumerate' is used to keep track of the index values of each item as the code loops through the 'items' column.
for i, item in enumerate(df1['items']):           # "for i, item in df1['items'].iteritems(): can also be used instead of enumerate"
    # extract the category name
    category = item['snippet']['title']
    # append the category name and its corresponding ID (i) to the list
    categories.append({'categoryId': i, 'category_name': category})

# create a new DataFrame from the list of categories
df_categories = pd.DataFrame(categories)

# print the new DataFrame
df_categories


#  > **Merging the category into dataframe**

# In[5]:


data=df_categories.merge(df,on='categoryId')
data


# In[2]:


utube=data.copy()


# # **Data Cleaning**
# *  **Handling missing values:**  *filling in or dropping missing data points in a dataset.*
# *  **Removing duplicates:**  *removing duplicate records from a dataset to avoid double-counting.*
# *  **Correcting data format:**  *transforming data to match the expected format for analysis or to meet certain data standards.*
# *  **Handling outliers:**  *removing or correcting extreme data points that may skew analysis results.*
# *  **Normalizing data:**  *transforming data to have a common scale, enabling easier comparison between different datasets.*
# *  **Handling inconsistent data:**  *identifying and resolving discrepancies in data that arise due to inconsistencies in measurement or data entry.*
# *  **Checking for data integrity:**  *verifying that data is accurate and complete, and identifying any potential errors or anomalies.*

# > **Checking for any null values**

# In[7]:


utube.isna().sum()


# In[8]:


utube.dropna(subset=['channelTitle'],inplace=True) #drops the na in the column 


# In[9]:


utube.drop(['categoryId','video_id','channelId','thumbnail_link','description'],axis=1,inplace=True) #Drop the unwanted columns


# In[10]:


utube.rename({ 'category_name':'category',
              'publishedAt':'published_at',    #renames the column name into readable form
              'channelTitle':'channel_title'
              },axis=1,inplace=True)


# In[11]:


utube.head()


# In[12]:


utube.info()


# > **It's advisable to check the datatype of each column and convert them back to their original datatypes if necessary.**
# 
# 

# In[13]:


utube['published_at'] = pd.to_datetime(utube['published_at']).dt.strftime('%Y-%m-%d')
utube['published_at'] = pd.to_datetime(utube['published_at'])
utube['trending_date'] = pd.to_datetime(utube['trending_date']).dt.strftime('%Y-%m-%d')
utube['trending_date'] = pd.to_datetime(utube['trending_date'])


# In[14]:


utube['publish_month']=pd.to_datetime(utube['published_at']).dt.strftime('%b')
utube['publish_day']=pd.to_datetime(utube['published_at']).dt.day


# In[15]:


utube.head()


# In[16]:


utube['publish_month'].unique()


# In[17]:


utube['publish_day'].nunique()


# In[18]:


utube.describe().T


# In[19]:


def convert_scientific_to_decimal(value):
   return round(float(value), 2)


# In[20]:


utube['view_count'] = utube['view_count'].apply(convert_scientific_to_decimal)


# # **1. Which categories of videos tend to receive the most views, likes, and comments?**

# In[21]:


# group the videos by category and calculate the average metrics
category_metrics = utube.groupby('category')[['view_count' ,'likes', 'dislikes', 'comment_count']].mean()
category_metrics.sort_values(by=['view_count', 'likes', 'dislikes', 'comment_count'],ascending=[False,False,False,False],inplace=True)
category_metrics=round(category_metrics)
category_metrics


# In[22]:


# Create a list of metric names to plot
metric_names = ['view_count', 'likes', 'dislikes', 'comment_count']

# Loop through each metric and create a sorted bar chart
for metric in metric_names:
    # Sort the DataFrame by the current metric in descending order
    sorted_metrics = category_metrics.sort_values(metric, ascending=False)
    
    # Create a Bar trace object with the sorted values
    trace = go.Bar(x=sorted_metrics.index, y=sorted_metrics[metric])
    
    # Create the Figure object
    fig = go.Figure(data=[trace])
    
    # Add title and axis labels
    fig.update_layout(title='Average {} by Category'.format(metric.capitalize()),
                      xaxis_title='Category',
                      yaxis_title='Average {}'.format(metric.capitalize()))
    
    # Show the plot
    fig.show()


# # **2. Which channel has the highest total number of views, likes, dislikes, and comments in the dataset?**

# In[23]:


utube.head()


# In[24]:


grouped_data=utube.groupby('channel_title')[['view_count' ,'likes', 'dislikes', 'comment_count']].mean()
grouped_data.sort_values(by='view_count',ascending=False,inplace=True)
grouped_data.head()


# In[25]:


fig = go.Figure()

fig.add_trace(go.Bar(x=grouped_data.index[:5], y=grouped_data['view_count'], name='Views'))
fig.add_trace(go.Bar(x=grouped_data.index[:5], y=grouped_data['likes'], name='Likes'))
fig.add_trace(go.Bar(x=grouped_data.index[:5], y=grouped_data['dislikes'], name='Dislikes'))
fig.add_trace(go.Bar(x=grouped_data.index[:5], y=grouped_data['comment_count'], name='Comments'))

fig.update_layout(
    updatemenus=[
        dict(
            type='dropdown',
            buttons=[
                dict(label='Views',
                     method='update',
                     args=[{'visible': [True, False,False,False]}]),
                dict(label='Likes',
                     method='update',
                     args=[{'visible': [False, True,False,False]}]),
                dict(label='Dislikes',
                     method='update',
                     args=[{'visible': [False, False,True,False]}]),
                dict(label='Comments',
                     method='update',
                     args=[{'visible': [False, False,False,True]}])
            ],
            active=0,
            showactive=True
        )
    ]
)

fig.show()


# # **3. Do the number of views, likes, dislikes, and comments for YouTube videos in the dataset have any relationship with each other? If there is a relationship, how strong is it and in what direction does it go?**

# In[26]:


# Create a correlation matrix
corr_matrix = utube[['view_count', 'likes', 'dislikes', 'comment_count']].corr()

# Create a heatmap using plotly.graph_objs
heatmap = go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.index.values,
    y=corr_matrix.columns.values,
    colorscale="GreenS"
    
)

# Set the title of the plot
layout = go.Layout(
    title="Correlation Matrix Heatmap",
    autosize=False
    
)

# Create a figure and plot the heatmap
fig = go.Figure(data=[heatmap], layout=layout)

# Show the plot
fig.show()


# In[27]:



    # Create a scatter plot of likes vs dislikes
    plt.scatter(x='dislikes', y='likes',data=utube)
    plt.title(f'Relationship between Likes and Dislikes ')
    plt.xlabel('Likes')
    plt.ylabel('Dislikes')
    plt.show()


# # **4.What are the top 10 most commonly used tags in videos which has high views?**

# In[28]:


# Select videos with views greater than or equal to 1 million
high_views = utube[utube['view_count'] >= 1000000]

# Combine all tags from the selected videos into a single list
all_tags = high_views['tags'].str.split('|').tolist()
all_tags = [tag for tags in all_tags for tag in tags]

# Count the occurrence of each tag
tag_counts = pd.Series(all_tags).value_counts()


print(tag_counts.head(10))


# In[29]:



# Create a bar chart of the top 10 most commonly used tags
bar = go.Bar(
    x=tag_counts.head(10).index,
    y=tag_counts.head(10).values,
    marker=dict(color=tag_counts.head(10).values, colorscale='Viridis'),
)

# Set the layout of the chart
layout = go.Layout(
    title='Top 10 Most Commonly Used Tags in Videos with High Views',
    xaxis=dict(title='Tag'),
    yaxis=dict(title='Count'),
)

# Combine the chart and layout, and plot the chart
fig = go.Figure(data=[bar], layout=layout)
fig.show()


# In[30]:


month=utube.groupby('publish_month')['view_count','likes'].sum().sort_values(by=['view_count','likes'],ascending=[False,False])
month


# In[31]:


fig = go.Figure()

# change the data being plotted
fig.add_trace(go.Bar(x=month.index, y=month['view_count'], name='Total Views'))
fig.add_trace(go.Bar(x=month.index, y=month['likes'], name='Total Likes'))

fig.update_layout(
    # change the labels of the dropdown buttons
    updatemenus=[
        dict(
            type='dropdown',
            buttons=[
                dict(label='Total Views',
                     method='update',
                     args=[{'visible': [True, False]},
                           {'title': 'Total Views and Likes'}]),
                dict(label='Total Likes',
                     method='update',
                     args=[{'visible': [False, True]},
                           {'title': 'Total Views and Likes'}])
            ],
            # change the initial button that is displayed
            active=1,
            showactive=True
        )
    ]
)

fig.show()


# # **5. Which channels have the most videos?**
# 

# In[32]:




# Group the data by channel_title and count the number of occurrences
channel_counts = utube.groupby('channel_title')['title'].count()

# Sort the channels by the number of trending videos in descending order
sorted_channels = channel_counts.sort_values(ascending=False)

# Plot the result on a horizontal bar graph
plt.barh(sorted_channels.index[:10], sorted_channels.values[:10])
plt.title('Top 10 Channels with Most Trending Videos')
plt.xlabel('Number of Trending Videos')
plt.show()


# # **6. How does the day of the week of video publishing affect the number of views and comments?**

# In[33]:



# Convert the published_at column to datetime format
utube['published_at'] = pd.to_datetime(utube['published_at'])

# Extract the day of the week from the published_at column
utube['publish_day'] = utube['published_at'].dt.day_name()

# Group the data by the publish_day column and calculate the average views and comments
avg_views_comments = utube.groupby('publish_day')[['view_count']].mean()

# Plot the result on a bar graph
avg_views_comments.plot(kind='bar')
plt.title('Average Views and Comments by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()


# # **7. Which channels have published videos that received the most views within a recent period of time, and what are the top 5 among them?**

# In[34]:


# Filter the data to only include videos published within the last week
one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
recent_videos = utube[utube['published_at'] >= one_week_ago]

# Sort the recent_videos dataframe by view count in descending order
sorted_videos = recent_videos.sort_values(by='view_count', ascending=False)
sorted_videos['trend_time']=(sorted_videos['trending_date']-sorted_videos['published_at']).dt.days


max_views_per_channel = sorted_videos.groupby('channel_title').agg({'title': 'first', 'view_count': 'max','trend_time': 'first'})
max_views_per_channel.sort_values(by=['view_count','trend_time'],ascending=[False,True],inplace=True)
max_views_per_channel


# In[35]:


# Create a list of metric names to plot
graph_names = ['view_count','trend_time']

# Loop through each metric and create a sorted bar chart
for graph in graph_names:
   
    
    # Create a Bar trace object with the sorted values
    trace = go.Bar(x=max_views_per_channel.index[:5], y=max_views_per_channel[graph][:5])
    
    # Create the Figure object
    fig = go.Figure(data=[trace])
    
    # Add title and axis labels
    fig.update_layout(title='Most viewed video for a channel withn a short span',
                      xaxis_title='Channel Name',
                      yaxis_title='No. of  {}'.format(graph.capitalize()))
    
    # Show the plot
    fig.show()

