import streamlit as st
import pandas as pd
import plost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
import plotly.graph_objects as go
import statsmodels.api as sm
import cbsodata
import ipywidgets as widgets
from ipywidgets import interact
 
#inladen data
calendar = pd.read_csv("calendar.csv")
listing = pd.read_csv("listings.csv")
row_to_skip = 16341
 
# Read the CSV file skipping the specified row
listingd = pd.read_csv("listings_details.csv", skiprows=lambda x: x == row_to_skip)
neighbourhood = pd.read_csv("neighbourhoods.csv")
 
#cbs api
toc = pd.DataFrame(cbsodata.get_table_list())
 
# Downloading entire dataset (can take up to 30s)
huurverhoging = pd.DataFrame(cbsodata.get_data('83162NED'))
 
#data cleaning
df = calendar.sort_values('price', ascending = False)
df_unique = df.drop_duplicates(subset='listing_id', keep='first').copy()
df_unique_nan = df_unique.dropna()
 
#prijs een cijfer maken
df['price'] = df['price'].astype('str')
df['price'] = df['price'].apply(lambda x:x.lstrip('$'))
df['price'] = df['price'].apply(lambda x:x.replace(',',''))
df['price'] = df['price'].astype('float')
df_unique['price'] = df_unique['price'].astype('str')
df_unique['price'] = df_unique['price'].apply(lambda x:x.lstrip('$'))
df_unique['price'] = df_unique['price'].apply(lambda x:x.replace(',',''))
df_unique['price'] = df_unique['price'].astype('float')
listingd['price'] = listingd['price'].astype('str')
listingd['price'] = listingd['price'].apply(lambda x:x.lstrip('$'))
listingd['price'] = listingd['price'].apply(lambda x:x.replace(',',''))
listingd['price'] = listingd['price'].astype('float')
calendar['price'] = calendar['price'].astype('str').apply(lambda x:x.lstrip('$'))
calendar['price'] = calendar['price'].apply(lambda x:x.replace(',',''))
calendar['price'] = calendar['price'].astype('float')
df = listing
price = df.groupby('neighbourhood').price.mean()
price = price.reset_index()
number_of_offers = df.groupby('neighbourhood').count()
number_of_offers.rename(columns={'name': 'offers'}, inplace=True)
neighbourhood.set_index('neighbourhood', inplace=True)
apartments = df[['latitude','longitude','room_type']]
 
 
 
st.title('Analysis of Air bnb listings in Amsterdam')
 
#Barplot room type
# Bereken de telling van kamer types
rooms = apartments['room_type'].value_counts()
 
# Plot de staafdiagram in Streamlit
fig = plt.figure(figsize=(11, 7))
plt.bar(['Entire home/apt', 'Private room', 'Shared room'], rooms, color=['#a6611a', '#018571', 'gray'], width=0.4)
 
# Toon labels en titel
plt.xlabel('Room Type', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Room Types in Amsterdam', fontsize = 16)
 
# Toon de staafdiagram in Streamlit
st.pyplot(fig)
 
#barplot
mean_prices = listingd.groupby('neighbourhood_cleansed')['price'].mean().sort_values().index
 
# Kies een kleurenpalet voor de grafiek
colors = sns.color_palette('Paired', n_colors=len(mean_prices))
 
# Plot de staafdiagram in Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='neighbourhood', y='price', order=mean_prices, ax=ax, palette=colors)
plt.title('Mean prices per neighbourhood', fontsize=16)
 
# Voeg annotaties toe voor het gemiddelde aantal personen (accommodates)
for bar, val in zip(ax.patches, listingd.groupby('neighbourhood')['review_scores_rating'].mean()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f'{int(round(val))}', ha='center', va='bottom', fontsize=8, color='black')
 
# Stel de rotatie van de x-labels in
plt.xticks(rotation=90)
 
# Toon de staafdiagram in Streamlit
st.pyplot(fig)
 
 
#plot accomodates
listingd1= listingd.reset_index()
# Creëer een figuur met matplotlib
fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
 
# Gebruik seaborn voor de histogramplot in de gewenste stijl
sns.histplot(data=listingd1, x='accommodates', ax=ax)
 
plt.xticks(rotation=0)
plt.xlabel('Accommodates', fontsize=16)
plt.ylabel('Number of listings', fontsize=16)
plt.title('Accommodates (number of people)', fontsize=18)
 
# Toon het diagram met Streamlit
st.pyplot(fig)
 
 
#scatterplot
dflist = listingd[(listingd.price <3000)&(listingd.review_scores_rating>60)&(listingd.accommodates < 7)]
 
fig, ax = plt.subplots()
scatter = sns.scatterplot(data=dflist, y='review_scores_rating', x='price', hue='accommodates')
 
# Voeg labels toe aan de assen en titel
plt.xlabel('Price')
plt.ylabel('Review score')
plt.title('Price vs review score')
 
# Voeg legenda toe
plt.legend(title='Guest amount')
 
# Toon de scatterplot in Streamlit
st.pyplot(fig)
 
#title linear regression result
st.title('Linear regression result')
#linear
st.image("Linear_result.png", caption="Linear Regression Result", use_column_width=True)
 
 
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['year'] = calendar['date'].dt.year
merge1 = listingd[['id', 'neighbourhood_cleansed']]
merged = pd.merge(merge1, calendar, left_on='id', right_on='listing_id')
 
neighbourhoods = ['All neighbourhoods'] + merged['neighbourhood_cleansed'].unique().tolist()
 
def plot_neighbourhood(neighbourhood):
    st.subheader(f'Mean pric per year for {neighbourhood}')
 
    if neighbourhood == 'All neighbourhoods':
        mean_prices = merged.groupby(['year'])['price'].mean().reset_index()
    else:
        df_neighbourhood = merged[merged['neighbourhood_cleansed'] == neighbourhood]
        mean_prices = df_neighbourhood.groupby('year')['price'].mean().reset_index()
 
    # Maak de barplot met Seaborn
    fig, ax = plt.subplots()
    sns.barplot(x='year', y='price', data=mean_prices, ax=ax)
    ax.set(xlabel='Year', ylabel='Mean price')
    # Toon de plot in Streamlit
    st.pyplot(fig)
 
# Creëer een interactieve dropdown met buurten, inclusief de optie 'Alle buurten'
selected_neighbourhood = st.selectbox('Select neighbourhood', neighbourhoods)
plot_neighbourhood(selected_neighbourhood)
 
huurverhoging['Perioden'] = pd.to_datetime(huurverhoging.Perioden)
huurverhoging_ams = huurverhoging.loc[(huurverhoging.RegioS == 'Amsterdam')&(huurverhoging.Perioden > "2015")]
st.dataframe(huurverhoging_ams)
 
code = '''print((calendar2019.price.mean()-calendar2018.price.mean())/calendar2018.price.mean()*100)
                3.401724999038135'''
st.code(code, language='python')