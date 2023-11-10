import streamlit as st
import pandas as pd
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
import json
import geopandas as gpd
from folium.plugins import MarkerCluster
from folium.features import Choropleth
import folium
from streamlit_folium import folium_static
from folium import Choropleth
from folium.plugins import HeatMap
from branca.colormap import linear
 
#inladen data
calendar = pd.read_csv("calendar.csv")
listing = pd.read_csv("listings.csv")
#listingd = pd.read_csv("listings_details.csv")
neighbourhood = pd.read_csv("neighbourhoods.csv")
geo_ams = gpd.read_file('neighbourhoods.geojson')
 
neighbourhoods = pd.merge(geo_ams, neighbourhood, on='neighbourhood', how='left')
price = listing.groupby('neighbourhood').price.mean()
 
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
#listingd['price'] = listingd['price'].astype('str')
#listingd['price'] = listingd['price'].apply(lambda x:x.lstrip('$'))
#listingd['price'] = listingd['price'].apply(lambda x:x.replace(',',''))
#listingd['price'] = listingd['price'].astype('float')
 
df = listing
price = df.groupby('neighbourhood').price.mean()
#price = price.reset_index()
number_of_offers = df.groupby('neighbourhood').count()
number_of_offers.rename(columns={'name': 'offers'}, inplace=True)
neighbourhoods.set_index('neighbourhood', inplace=True)
apartments = df[['latitude','longitude','room_type']]
buurten = df.neighbourhood.unique()
#kaart
geo_ams["longitude"] = geo_ams.centroid.x
geo_ams["latitude"] = geo_ams.centroid.y
geo_ams.drop('neighbourhood_group', axis=1, inplace=True)
neighbourhood_price = df.groupby('neighbourhood')['price'].mean()
neighbourhood_price = pd.DataFrame({'neighbourhood':neighbourhood_price.index, 'price': neighbourhood_price.T.values})
neighbourhood_price.sort_values('price', ascending=False, inplace=True)
 
 
 
 
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
 
st.sidebar.header('Select')
 
 
st.sidebar.subheader('Choose your price')
max_prijs = st.sidebar.slider('Max price', 0,800, value=400)
plot_data = st.sidebar.multiselect('Select Neighboorhood', buurten, buurten)
 
#kaart count neighbourhood
df_sidebar = df.loc[(df.price<max_prijs)]
 
neighbourhood_count = pd.DataFrame({
    'neighbourhood': df_sidebar['neighbourhood'].value_counts().index,
    'count': df_sidebar['neighbourhood'].value_counts().values
})
 
 
 
# Row A
st.title('Dashboard')
df_gem_prijs = df_sidebar.loc[df_sidebar.neighbourhood.isin(plot_data)]
# Bereken het gemiddelde van de prijzen in de geselecteerde buurten en rond het af naar het dichtstbijzijnde hele getal
gemiddelde_prijs = round(df_gem_prijs['price'].mean())
df_total_listing = df_sidebar.loc[df_sidebar.neighbourhood.isin(plot_data)]
total_listings = df_total_listing.id.nunique()
 
# Row A
st.markdown('### Summary')
col1, col2, col3 = st.columns(3)
col1.metric("Mean price", gemiddelde_prijs)
col2.metric("Total listings", total_listings)
 
# Row B
 
Q1 = df_sidebar['price'].quantile(0.25)
Q3 = df_sidebar['price'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df_sidebar[(df_sidebar['price'] >= Q1 - 1.5 * IQR) & (df_sidebar['price'] <= Q3 + 1.5 * IQR)]
 
 
c1, c2 = st.columns((3,7))
with c1:
    st.markdown('### Heatmap')
   
 
# Plot de boxplot met Plotly Express zonder outliers
    fig = px.box(filtered_df, y='price')
 
# Update the title and x-axis label
    fig.update_layout(
        title_text='Price Boxplot',
        xaxis_title='Price',
        width=300,  # Adjust the width as needed
    )
 
# Display the boxplot in Streamlit hallo
    st.plotly_chart(fig)
with c2:
    st.markdown("### Map of the number of Airbnb's per neighbourhood ")
    # Streamlit-app lay-out en functionaliteit
    map_ams_count = folium.Map(location=[52.37, 4.89], zoom_start=11)
    choropleth = folium.Choropleth(
    geo_data=geo_ams,
    data=neighbourhood_count,
    columns=['neighbourhood', 'count'],
    key_on='feature.properties.neighbourhood',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='The number of properties',
    reset=True,
    highlight=True,  # Om highlighting toe te voegen
    line_color='black',  # Kleur van de grenzen tussen de buurten
    line_weight=1,  # Dikte van de lijnen tussen de buurten
     # Doorzichtigheid van de lijnen tussen de buurten
    name='Choropleth'
    ).add_to(map_ams_count)
 
# Voeg tooltip toe met de beschikbare velden
    folium.GeoJsonTooltip(fields=['neighbourhood'], aliases=['Neighbourhood:']).add_to(choropleth.geojson)
 
# Toon de kaart in Streamlit
    folium_static(map_ams_count)
 
     
 
#row c
c3, c4 = st.columns((5,5))
with c3:
 
    st.markdown('### Average price per neighbourhood')
 
 
# Voeg Choropleth-laag toe aan de kaart met tooltip
 
 
#barplot over average price of room
 
# Creëer een figuur met matplotlib
    fig, ax = plt.subplots(figsize=(14, 10), dpi=80)
 
# Gebruik seaborn voor de staafplot
    barplot_price_room1 = price.reset_index()
    barplot_price_room = barplot_price_room1.loc[barplot_price_room1.neighbourhood.isin(plot_data)]
    sns.barplot(data=barplot_price_room.sort_values(by='price'), x='neighbourhood', y='price', palette='BrBG', ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Neighbourhood', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.title('Average Price of Room', fontsize=18)
 
# Toon het diagram met Streamlit
    st.pyplot(fig)
 
barplot_count_room1 = number_of_offers.reset_index()
barplot_count_room = barplot_count_room1.loc[barplot_count_room1.neighbourhood.isin(plot_data)]
 
with c4:
    st.markdown('### Number of offers per neighbourhood')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
 
# Gebruik seaborn voor de staafplot
    sns.barplot(data=barplot_count_room.sort_values(by='offers'), x='neighbourhood', y='offers', palette='BrBG', ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Neighbourhood', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Number of Offers', fontsize=18)
 
# Toon het diagram met Streamlit
    st.pyplot(fig)
 
st.markdown('Source: https://www.kaggle.com/datasets/erikbruin/airbnb-amsterdam/')