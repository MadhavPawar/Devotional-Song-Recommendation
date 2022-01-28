import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



df=pd.read_csv("D:\\ExcelR_Classes_Data\\Jupyter_Notebook_Practice\\DS_Project_01_Spotify\\500_devo.csv")
df=df.rename(columns={'Release Date':'Release_Date'})
df=df.rename(columns={'Artist Name(s)':'Artist_Name'})
df['Release_Date']= pd.to_datetime(df['Release_Date'])
df['year']=df['Release_Date'].map(lambda x:x.strftime('%Y'))
df['year']=df['year'].astype(str).astype(int)
df=df.drop(['Release_Date'],axis=1)
df['duration_min'] = df['Duration (ms)']/60000
df['duration_min'] = df['duration_min'].round(2)
df=df.drop(['Spotify ID'],axis=1)
#df.duration_min.sum()/(24*365*60)
df.drop(['Duration (ms)'],inplace=True,axis=1)




#Page Layout (side bar with 3 buttons)
st.sidebar.header('Navigation')
st.sidebar.radio('Choose',['Home', 'About', 'workflow'])
###########################


    #########################

#st.image('D:ExcelR_Classes_Data/Stremalit/Spotify_Final.png')
#st.spinner('waiting...')
#def sidebar_data():


st.title('Devotional Song Recommendation Engine')
st.text('get the similar songs like your favorite ones and enjoy non-stop music.')

######################### MODEL ####################################
song_input = st.text_input("whats  Your  Favorite  Song ?")
search_button = st.button("Search")
if search_button:
    st.write('If "',song_input,'" is your favorite song then you may also like to listen:\n')


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df['Artist_Name'] = df['Artist_Name'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['Artist_Name'])
#tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Track Name']).drop_duplicates()
# Function that takes  song title as input and outputs most similar song
#################################################################

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all song with that song
    sim_scores = list(enumerate(cosine_sim[idx]))
        
    # Sort the song based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar song
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    song_indices = [i[0] for i in sim_scores]
    df['Track Name'].iloc[song_indices]
    return df['Track Name'].iloc[song_indices]

    # Return the top 10 most similar song

fn_call = get_recommendations(song_input)
st.write(fn_call)
####################################################################

#Song input from user and displaying recommendation result
##search_button = st.button("Search")
##if search_button:
    ##st.write('If "',song_input,'" is your favorite song then you may also like to listen:\n')



#dataset + train & test + model building + execution
#dataset = pd.read_csv("")


