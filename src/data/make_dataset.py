"""
Author: Ilia Altmark
"""
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import timeit
import pandas as pd
import time
from tqdm import tqdm

CID = "xxx"
SECRET = "xxx"


def main():
    client_credentials_manager = SpotifyClientCredentials(client_id=CID,
                                                          client_secret=SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # timeit library to measure the time needed to run this code
    start = timeit.default_timer()

    # create empty lists where the results are going to be stored
    artist_name = []
    track_name = []
    popularity = []
    track_id = []

    for year in range(2012, 2022):

        for i in range(0, 1000, 50):
            track_results = sp.search(q=f'year:{year} genre:techno',
                                      type='track', limit=50, offset=i)
            for i, t in enumerate(track_results['tracks']['items']):
                artist_name.append(t['artists'][0]['name'])
                track_name.append(t['name'])
                track_id.append(t['id'])
                popularity.append(t['popularity'])

    stop = timeit.default_timer()
    print('Time to run this code (in seconds):', stop - start)

    print('number of elements in the track_id list:', len(track_id))

    df_tracks = pd.DataFrame(
        {'artist_name': artist_name, 'track_name': track_name,
         'track_id': track_id, 'popularity': popularity})

    # group the entries by artist_name and track_name and check for duplicates

    grouped = df_tracks.groupby(['artist_name', 'track_name'],
                                as_index=True).size()
    print('number of duplicates:', grouped[grouped > 1].count())

    df_tracks.drop_duplicates(subset=['artist_name', 'track_name'],
                              inplace=True)

    # again measuring the time
    start = timeit.default_timer()

    # empty list, batchsize and the counter for None results
    rows = []
    batchsize = 100
    none_counter = 0

    for i in range(0, len(df_tracks['track_id']), batchsize):
        batch = df_tracks['track_id'][i:i + batchsize]
        feature_results = sp.audio_features(batch)
        for i, t in enumerate(feature_results):
            if t is None:
                none_counter = none_counter + 1
            else:
                rows.append(t)

    print('Number of tracks where no audio features were available:',
          none_counter)

    stop = timeit.default_timer()
    print('Time to run this code (in seconds):', stop - start)

    print('number of elements in the track_id list:', len(rows))

    df_audio_features = pd.DataFrame.from_dict(rows, orient='columns')
    print("Shape of the dataset:", df_audio_features.shape)

    columns_to_drop = ['analysis_url', 'track_href', 'type', 'uri']
    df_audio_features.drop(columns_to_drop, axis=1, inplace=True)

    df_audio_features.rename(columns={'id': 'track_id'}, inplace=True)

    # merge both dataframes
    # the 'inner' method will make sure that we only keep track IDs present in both datasets
    df = pd.merge(df_tracks, df_audio_features, on='track_id', how='inner')
    print("Shape of the dataset:", df.shape)

    # again measuring the time
    start = timeit.default_timer()

    # creating a dataset with additional features
    track_id = []
    key_confidence = []
    mode_confidence = []
    tempo_confidence = []
    time_signature_confidence = []
    None_counter = 0

    for i in tqdm(range(0, len(df['track_id']))):
        if i % 100 == 0:
            time.sleep(2)

        track = df['track_id'][i]

        try:
            analysis_results = sp.audio_analysis(track)['track']
        except:
            analysis_results = None

        if analysis_results == None:
            None_counter = None_counter + 1
        else:
            track_id.append(track)
            key_confidence.append(analysis_results['key_confidence'])
            mode_confidence.append(analysis_results['mode_confidence'])
            tempo_confidence.append(analysis_results['tempo_confidence'])
            time_signature_confidence.append(
                analysis_results['time_signature_confidence'])

    print('Number of tracks where no audio features were available:',
          None_counter)

    stop = timeit.default_timer()
    print('Time to run this code (in seconds):', stop - start)

    df_analysis = pd.DataFrame(
        {'track_id': track_id, 'key_confidence': key_confidence,
         'mode_confidence': mode_confidence,
         'tempo_confidence': tempo_confidence,
         'time_signature_confidence': time_signature_confidence})
    print(df_analysis.shape)

    df = pd.merge(df, df_analysis, on='track_id', how='left')
    print("Shape of the dataset:", df.shape)

    df.to_csv(
        'xxx')


if __name__ == "__main__":
    main()
