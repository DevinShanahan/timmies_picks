import json
import os
import re
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from ftfy import fix_text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode

# --- Setup ---
NAME_MAP_PATH = Path("name_map.json")

# Load existing name map from file if available
if NAME_MAP_PATH.exists():
    with open(NAME_MAP_PATH, "r") as f:
        name_map = json.load(f)
else:
    name_map = {}

# Store in session state for persistence between reruns
if "name_map" not in st.session_state:
    st.session_state["name_map"] = name_map

if "missing" not in st.session_state:
    # Example list; replace with however you get your missing players
    st.session_state["missing"] = []

# -----------------------------
# Functions
# -----------------------------


def clean_name(name: str) -> str:
    """
    Fix mojibake + weird encodings with ftfy, then transliterate to ASCII.
    Example:
        clean_name("Martin NeÄas") -> "Martin Necas"
        clean_name("Slafkovský")      -> "Slafkovsky"
    """
    fixed = fix_text(name)         # repairs mojibake and other messy Unicode issues
    return unidecode(fixed)        # transliterate diacritics to ASCII


def fetch_player_data(season, max_season):
    """Scrape player table from Hockey Reference"""
    current_season = season == max_season
    if not current_season and os.path.exists(f'season_{season}.csv'):
        return pd.read_csv(f'season_{season}.csv')
    url = f"https://www.hockey-reference.com/leagues/NHL_{season}_skaters.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "player_stats"})
    df = pd.read_html(StringIO(str(table)), header=[0, 1])[0]  # Hockey Reference often has 2 header rows
    df.columns = [col[1] for col in df.columns.values]  # Drop first column level
    df = df[df['Rk'].ne('Rk')]  # remove repeated header rows
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    df = df.apply(pd.to_numeric, errors="ignore")
    df['Player'] = df['Player'].apply(lambda n: clean_name(n))
    df = df.drop_duplicates(subset=['Player'], keep='first')
    df['ATOI'] = df['ATOI'].str.split(':').apply(lambda x: int(x[0]) + int(x[1])/60)
    df['AVG_SOG'] = df['SOG'] / df['GP']
    df['AVG_PTS'] = df['PTS'] / df['GP']
    df['AVG_A'] = df['A'] / df['GP']
    if not current_season:
        with open(f'season_{season}.csv', 'w') as f:
            df.to_csv(f)
    return df

def prepare_training_data(df):
    """Prepare features and target for model"""
    features = ['Age','GP','AVG_SOG','SPCT','ATOI','AVG_PTS','AVG_A','+/-']
    df = df.dropna(subset=['G', 'GP'])
    X = df[features].fillna(0)
    y = (df['G'] / df['GP']).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_model(X, y):
    """Train logistic regression model"""
    model = LinearRegression()
    model.fit(X, y)
    return model

def save_model(model, scaler, filename):
    """Save model and scaler together"""
    joblib.dump({"model": model, "scaler": scaler}, filename)

def load_model(filename):
    """Load model and scaler"""
    data = joblib.load(filename)
    return data["model"], data["scaler"]

def predict_top_player(model, scaler, df_players):
    """Return the player most likely to score from a list"""
    features = ['Age','GP','AVG_SOG','SPCT','ATOI','AVG_PTS','AVG_A','+/-']
    X = scaler.transform(df_players[features].fillna(0))
    probs: LinearRegression = model.predict(X)
    df_players = df_players.assign(Score_Prob=probs)
    top = df_players.sort_values('Score_Prob', ascending=False).iloc[0]
    return f"{top['Player']} ({top['Score_Prob']:.2f} likelihood)"


def extract_player_names(text: str) -> str:
    """
    Extracts player names from text that may include team abbreviations, jersey numbers, and positions.
    Also handles cases where text is already comma-separated names.
    Returns a comma-separated string of names only.
    """
    # Split by two or more spaces or commas
    parts = re.split(r'\s{2,}|,\s*', text.strip())
    names = []

    for part in parts:
        # Match only the name part before a team abbreviation or number
        match = re.match(r"^([A-Za-zÀ-ÿ'’\.\- ]+?)(?=\s[A-Z]{2,3}\b|$)", part.strip())
        if match:
            name = match.group(1).strip()
            # Skip empty or short fragments
            if len(name.split()) >= 2:
                names.append(name)

    # If no structured matches found, assume it's already just names
    if not names:
        names = [n.strip() for n in text.split(',') if n.strip()]

    return ', '.join(names)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="NHL Goal Predictor", page_icon="🏒", layout="centered")

st.title("🏒 NHL Goal Predictor")
st.write("Predict which player from your list is most likely to score, using last season's data from Hockey-Reference.")

# season = st.number_input("Select season year", min_value=2020, max_value=2027, value=2027)
# num_seasons = st.number_input("Number of seasons to train on", min_value=1, max_value=5, value=3)
season = 2027
num_seasons = 5

# if st.button("Fetch & Train Model"):
df = None
for ssn in range(season-num_seasons, season):
    with st.spinner(f"Loading NHL {ssn} skater data..."):
        try:
            season_df = fetch_player_data(ssn, season)
            if df is None:
                df = season_df
            else:
                df = pd.concat([df, season_df], ignore_index=True)
        except Exception as e:
            st.warning(f"Error fetching data for season {ssn}: {e}")
model_filename = f"{season-num_seasons}-{season}.joblib"
try:
    model, scaler = load_model(model_filename)
    # st.success("Model loaded successfully!")
except FileNotFoundError:
    X, y, scaler = prepare_training_data(df)
    # st.warning("No saved model found. Training a new model...")
    model = train_model(X, y)
    save_model(model, scaler, model_filename)
st.session_state.model = model
st.session_state.scaler = scaler
st.session_state.df = df
# st.success(f"Model trained on {len(df)} players!")

if "model" in st.session_state:
    st.header("Enter Player Groups")
    st.write("Copy/Paste list from TH App, or enter comma-separated names for each group.")

    group1 = extract_player_names(st.text_input("Group 1 Players"))
    group2 = extract_player_names(st.text_input("Group 2 Players"))
    group3 = extract_player_names(st.text_input("Group 3 Players"))
    missing = []
    if st.button("Predict Scorers"):
        model = st.session_state.model
        scaler = st.session_state.scaler
        with st.spinner(f"Loading NHL {season} skater data..."):
            df = fetch_player_data(season, season)

        for i, g in enumerate([group1, group2, group3], start=1):
            if g.strip():
                players = [name_map.get(p.strip(), p.strip()) for p in g.split(",")]
                df_players = df[df['Player'].isin(players)]
                if df_players.empty:
                    st.error(f"**Group {i}:** No players found in dataset")
                else:
                    result = predict_top_player(model, scaler, df_players)
                    st.write(f"**Group {i}:** {result}")
                    unique_players = df_players.Player.unique()
                    if len(unique_players) < len(players):
                        missing_players = [p for p in players if p not in unique_players]
                        st.session_state.missing.extend(missing_players)
                        st.warning(f"Players not found in dataset: {missing_players}")

if st.session_state.missing:
    missing_players = list(set(st.session_state.missing))
    st.write("### Missing Players")
    for player in missing_players.copy():
        key_suffix = re.sub(r"\W+", "_", player)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(player)
        with col2:
            name_key = f"name_{key_suffix}"
            mapped_name = st.text_input(
                f"Set name for {player}",
                key=name_key,
                label_visibility="collapsed",  # hides the label
                placeholder="Enter correct name",
            )
        with col3:
            if st.button("Submit", key=f"submit_{key_suffix}"):
                mapped_name = mapped_name.strip()
                if mapped_name:
                    # Save to session + file
                    st.session_state["name_map"][player] = mapped_name
                    with open(NAME_MAP_PATH, "w") as f:
                        json.dump(st.session_state["name_map"], f, indent=2)

                    print(f"✅ Updated {player} → {mapped_name}")
                    st.session_state["missing"].remove(player)
                    st.rerun()  # instantly refreshes UI to hide this row
                else:
                    st.warning("Please enter a name before submitting.")
