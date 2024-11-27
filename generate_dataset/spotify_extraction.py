import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from io import BytesIO
import pandas as pd
from typing import Any
import numpy as np
import os
import librosa
import requests
import numpy as np

METAL_SUBGENRES = [
    "Black",
    "Death",
    "Doom",
    "Sludge",
    "Industrial",
    "Experimental",
    "Folk",
    "Gothic",
    "Grindcore",
    "Groove",
    "Heavy",
    "Metalcore",
    "Deathcore",
    "Power",
    "Progressive",
    "Speed",
    "Symphonic",
    "Thrash",
]
FEATURE_COLUMNS = [
    "Track ID",
    "Subgenre",
    "Slice",
    "Chroma",
    "RMS",
    "Spectral Centroid",
    "Spectral Bandwidth",
    "Spectral Rolloff",
    "Zero Crossing Rate",
    "MFCC",
    "Harmony",
    "Tempo",
]
MAX_PLAYLISTS = 5
MIN_TRACKS_IN_PLAYLIST = 100
TOP_TRACKS_LIMIT = 100
SLICE_DURATION = 3


load_dotenv()
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.environ["CLIENT_ID_ZK"],
        client_secret=os.environ["CLIENT_SECRET_ZK"],
    )
)


def mean_plus_variance(feature: np.ndarray) -> float:
    """
    Calculate the sum of the mean and the variance (square of the standard deviation)
    of a numeric feature array.

    Args:
        feature np.ndarray: A numpy array object containing numeric values.

    Returns:
        float: The calculated value (mean + variance).
    """
    return np.mean(feature) + np.std(feature) ** 2


def extract_features(slice_data: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Extract audio features from a given slice of audio data.

    Args:
        slice_data (np.ndarray): Audio slice data.
        sr (int): Sampling rate.

    Returns:
        Dict[str, Any]: A dictionary of computed audio features.
    """
    return {
        "Chroma": mean_plus_variance(librosa.feature.chroma_stft(y=slice_data, sr=sr)),
        "RMS": mean_plus_variance(librosa.feature.rms(y=slice_data)),
        "Spectral Centroid": mean_plus_variance(
            librosa.feature.spectral_centroid(y=slice_data, sr=sr)
        ),
        "Spectral Bandwidth": mean_plus_variance(
            librosa.feature.spectral_bandwidth(y=slice_data, sr=sr)
        ),
        "Spectral Rolloff": mean_plus_variance(
            librosa.feature.spectral_rolloff(y=slice_data, sr=sr)
        ),
        "Zero Crossing Rate": mean_plus_variance(
            librosa.feature.zero_crossing_rate(y=slice_data)
        ),
        "MFCC": mean_plus_variance(
            librosa.feature.mfcc(y=slice_data, sr=sr, n_mfcc=20)
        ),
        "Harmony": mean_plus_variance(librosa.effects.harmonic(y=slice_data)),
        "Tempo": librosa.beat.tempo(y=slice_data, sr=sr).mean(),
    }


def get_top_100_tracks_for_all_subgenres(subgenres: list[str]) -> pd.DataFrame:
    """
    Retrieves the top 100 tracks for each given subgenre from Spotify playlists.
    Since we are processing lots of data, if an error occurs we return the DataFrame anyway.

    Args:
        subgenres (List[str]): A list of subgenres to search for.

    Returns:
        pd.DataFrame: A DataFrame containing track IDs and their associated subgenres.
    """
    labelled_tracks = []
    try:
        for subgenre in subgenres:
            playlist_results = sp.search(
                q=f"{subgenre} Metal", type="playlist", limit=MAX_PLAYLISTS
            )

            selected_playlist_id = next(
                (
                    playlist["id"]
                    for playlist in playlist_results["playlists"]["items"]
                    if playlist["tracks"]["total"] >= MIN_TRACKS_IN_PLAYLIST
                ),
                None,
            )

            playlist_tracks = sp.playlist_tracks(
                selected_playlist_id, limit=TOP_TRACKS_LIMIT
            )
            track_list = [
                track["track"]
                for track in playlist_tracks["items"]
                if track["track"] is not None
            ]

            sorted_tracks = sorted(
                track_list, key=lambda x: x["popularity"], reverse=True
            )
            labelled_tracks.extend(
                [
                    {"Track ID": track["id"], "Subgenre": subgenre}
                    for track in sorted_tracks
                ]
            )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        return pd.DataFrame(labelled_tracks, columns=["Track ID", "Subgenre"])


def extract_audio_features_from_preview(
    df: pd.DataFrame, start_index: int = 0
) -> pd.DataFrame:
    """
    Extract audio features (MEAN + SD²) for 3-second slices of preview URLs.
    Since we are processing lots of data, if an error occurs we return the DataFrame anyway.
    Args:
        df (pd.DataFrame): DataFrame with columns "Track ID" and "Subgenre".
        start_index (int): Index to start processing from (default is 0).

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    feature_results = []

    try:
        for index, row in df.iloc[start_index:].iterrows():
            print(f"Processing index: {index:04d}", end="\r")
            track_id = row["Track ID"]
            subgenre = row["Subgenre"]

            track_details = sp.track(track_id)
            preview_url = track_details.get("preview_url")

            if not preview_url:
                print(f"No preview URL for track {track_id}")
                continue

            response = requests.get(preview_url, timeout=30)
            y, sr = librosa.load(BytesIO(response.content), sr=None)

            # Always returns 29.71265306122449 hence last slice is ignored
            duration = librosa.get_duration(y=y, sr=sr)
            num_slices = int(duration // SLICE_DURATION)
            for i in range(num_slices):
                start_sample = i * SLICE_DURATION * sr
                end_sample = start_sample + SLICE_DURATION * sr
                slice_data = y[int(start_sample) : int(end_sample)]
                features = extract_features(slice_data, sr)
                features.update(
                    {"Track ID": track_id, "Subgenre": subgenre, "Slice": i + 1}
                )
                feature_results.append(features)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        return pd.DataFrame(feature_results, columns=FEATURE_COLUMNS)


def plot_feature_differences_per_label(features_df):
    """
    Plots the differences in audio features for each subgenre.

    Args:
        features_df (pd.DataFrame): DataFrame containing audio features and their associated subgenres.
    """
    # Extract numeric features and group by subgenre
    feature_columns = features_df.columns.drop("Subgenre")
    grouped = features_df.groupby("Subgenre")[feature_columns]

    # Calculate mean and standard deviation for each feature per subgenre
    mean_features = grouped.mean()
    std_features = grouped.std()

    # Create subplots for each feature
    plt.figure(figsize=(20, len(feature_columns) * 4))
    for i, feature in enumerate(feature_columns, start=1):
        plt.subplot(len(feature_columns), 1, i)
        plt.errorbar(
            mean_features.index,
            mean_features[feature],
            yerr=std_features[feature],
            fmt="o-",
            label=feature,
            capsize=5,
        )
        plt.title(f"Feature: {feature}", fontsize=14)
        plt.xlabel("Subgenre", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("feature_differences_per_label.png", dpi=300, bbox_inches="tight")
