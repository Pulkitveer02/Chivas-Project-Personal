import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import VerticalPitch, Pitch
import seaborn as sns
import pyarrow
import io
from matplotlib.patches import Rectangle


df = pd.read_parquet("datasets/player_stats.parquet")

df_formations = df.loc[(df['type'] == "formationPlace") & (df['value'] > 0)]
df_formations["pos"]  = [pos[0] for pos in df_formations["pos"]]

df_team_formations = pd.read_parquet("datasets/formations.parquet")

# STEP 1: Get Most Common Formation Per Team
def get_most_common_formation(df_team_formations, team_name):
    team_data = df_team_formations[df_team_formations['team'] == team_name]
    most_common = team_data['formation'].mode()
    return str(most_common.iloc[0]) if not most_common.empty else None


# STEP 2: Get Most Likely Players for That Formation
def get_likely_players(df_formations, team_name):
    # Filter for this team
    team_players = df_formations[df_formations['equipo'] == team_name]

    # Count number of appearances per player per formation place
    player_counts = (
        team_players
        .groupby(['jugador', 'value'], as_index=False)
        .size()  # returns a column named "size"
    .rename(columns={'size': 'appearance_count'})
)

    # Get the player with the most appearances for each formation place
    top_players = (
        player_counts.sort_values('appearance_count', ascending=False)
        .drop_duplicates(subset='value', keep='first')
        .sort_values('value')  # sort by formation place so you can plot easily
    )
    st.dataframe(top_players)

    return top_players

# STEP 3: Build DataFrame with Coordinates
def build_formation_dataframe(players_df, formation_code, formation_coords_dict):
    coords = formation_coords_dict.get(formation_code)
    if coords is None or len(players_df) == 0:
        return pd.DataFrame()
    
    top_players = players_df.head(len(coords)).copy()
    top_players['x'] = [pt[0] for pt in coords]
    top_players['y'] = [pt[1] for pt in coords]
    return top_players

# STEP 4: Draw Formation with mplsoccer
def plot_formation(df, team_name, formation_code):

    pitch = Pitch(pitch_type='opta')
    fig, ax = pitch.draw(figsize=(10, 7))

    
    for i, row in df.iterrows():
        pitch.scatter(row['x'], row['y'], s=2500, ax=ax, color=team_colors[team_name], edgecolors='black', zorder=3, marker = 's')
        pitch.annotate(
            row['jugador'], (row['x'], row['y']),
            va='center', ha='center', size=10, ax=ax, zorder=4, color='white'
        )
    ax.set_title(f"{team_name} — {formation_code} Formation", fontsize=14)

    return fig



def formation_usage_streamlit(df, team_name):
    """
    Streamlit version of formation usage visualization.
    - Displays value_counts
    - Plots bar chart in Streamlit
    """

    df = df[df['team'] == team_name]
    # Get value counts
    formation_counts = df["formation"].value_counts()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=formation_counts.index,
        y=formation_counts.values,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Formation Usage Frequency", fontsize=16, weight='bold')
    ax.set_xlabel("Formation")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

    sns.despine(ax=ax)
    st.pyplot(fig)  # Show the figure in the app


### FORMATION FOR THE TEAMS
formation_coordinates_opta = {
    "532": [
        (5,50),    # 1: GK
        (25,10),   # 2: RWB
        (25,90),   # 3: LWB
        (20,70),   # 6: RCB
        (20,50),   # 5: CB
        (20,30),   # 4: LCB
        (55,35),   # 7: RAM
        (40,50),   # 8: CM
        (75,65),   # 9: LST
        (75,35),   # 10: RST
        (55,65)    # 11: LAM
    ],
    "433": [
        (5,50),    # 1: GK
        (20,10),   # 2: RB
        (20,90),   # 3: LB
        (40,50),   # 4: CDM
        (20,35),   # 5: RCB
        (20,65),   # 6: LCB
        (55,35),   # 7: RAM
        (55,65),   # 8: LAM
        (80,50),   # 9: ST
        (70,10),   # 10: RW
        (70,90)    # 11: LW
    ],
    "442": [
        (5,50),    # 1: GK
        (25,10),   # 2: RB
        (25,90),   # 3: LB
        (50,40),   # 4: RCM
        (20,35),   # 5: RCB
        (20,65),   # 6: LCB
        (50,10),   # 7: RW
        (50,60),   # 8: LCM
        (70,60),   # 9: LCF
        (70,40),   # 10: RCF
        (50,90)    # 11: LW
    ],
    "4231": [
        (5,50), (25,10), (25,90), (40,70), (20,35),
        (20,65), (60,10), (40,30), (80,50), (60,50), (60,90)
    ],
    "3421": [
        (5,50), (40,10), (40,90), (25,70), (25, 50),
        (25,30), (40, 35), (40, 65), (80, 50), (65,35), (65,65)
    ],
    "3511": [
        (5,50), (40,10), (40, 90), (25,70), (25, 50),
        (25,30), (60,35), (60,65), (80,50), (60,50), (40,50)
    ],
    "343": [
        (5,50), (40,10), (40,90), (25,70), (25,50),
        (25,30), (50,35), (50,65), (75,50), (65,10), (65,90)
    ],
    "352": [
        (5,50), (40,10), (40, 90), (25,70), (25, 50),
        (25,30), (60,35), (60,65), (80,50), (60,50), (60,50)
    ],
    "3241": [
        (5,50), (40,35), (40,65), (25,70), (25, 50),
        (25,30), (65,35), (65,65), (80,50), (65,10), (65,90)
    ],
    "3412": [
        (5,50), (40,10), (40, 90), (25,70), (25, 50),
        (25,30), (55,35), (55,65), (70,50), (80,35), (80,65)
    ],
    "4312": [
        (5,50), (25,10), (25,90), (40,50), (20,35),
        (20,65), (40,30), (60,50), (80,35), (80,65), (40,70)
    ],
    "4141": [
        (5,50),    # 1: GK
        (25,10),   # 2: RB
        (25,90),   # 3: LB
        (40,50),   # 4: CDM
        (20,35),   # 5: RCB
        (20,65),   # 6: LCB
        (60,10),   # 7: RW
        (60,35),   # 8: RCM
        (80,60),   # 9: CF
        (60,65),   # 10: LCM
        (60,90)    # 11: LW
    ],
    "4132": [
        (5,50),    # 1: GK
        (25,10),   # 2: RB
        (25,90),   # 3: LB
        (40,50),   # 4: CDM
        (20,35),   # 5: RCB
        (20,65),   # 6: LCB
        (60,10),   # 7: RW
        (60,50),   # 8: CAM
        (80,35),   # 9: RCF
        (80,65),   # 10: LCF
        (60,90)    # 11: LW
    ],
    "541": [
        (5,50),    # 1: GK
        (40,10),   # 2: RWB
        (40,90),   # 3: LWB
        (25,70),   # 4: RCB
        (25,50),   # 5: CB
        (25,30),   # 6: LCB
        (55,10),   # 7: RW
        (55,35),   # 8: RCM
        (75,65),   # 9: ST
        (55,65),   # 10: LCM
        (55,90)    # 11: LW
    ],
    "451": [
        (5,50), (25,90), (25,10), (20,65), (20,35),
        (40,90), (40,65), (40,50), (40,35), (60,50), (80,50)
    ],
    "3142": [
        (5,50), (25,50), (25,35), (25,65), (40,90),
        (40,50), (40,10), (70,35), (70,65), (60,50), (80,50)
    ],
    "4411": [
        (5,50), (25,90), (25,10), (20,65), (20,35),
        (40,90), (40,65), (40,35), (70,50), (60,50), (80,50)
    ],
    "41212": [
        (5,50), (25,90), (25,10), (20,65), (20,35),
        (40,50), (60,35), (60,65), (70,50), (60,50), (80,50)
    ],
    "4222": [
        (5,50), (25,90), (25,10), (20,65), (20,35),
        (40,50), (60,90), (60,35), (70,50), (60,50), (80,50)
    ]
}

### TEAM COLOURS
team_colors = {
    "Deportivo Toluca FC": "#E00000",  # Red
    "CF América": "#FFF200",           # Yellow
    "CF Cruz Azul": "#1F60A6",         # Blue
    "Club Tigres UANL": "#FDB913",     # Gold
    "Club Necaxa": "#E31920",          # Red
    "Club León": "#007A53",            # Green
    "CF Pachuca": "#1A2E6B",           # Navy Blue
    "CF Monterrey": "#003366",         # Dark Blue
    "Club Universidad Nacional": "#003366",  # Dark Blue
    "FC Juárez": "#77B200",            # Bright Green
    "Querétaro FC": "#00529C",         # Blue
    "Atlético de San Luis": "#E31920", # Red
    "Club Santos Laguna": "#009345",   # Green
    "Club Tijuana Xoloitzcuintles de Caliente": "#D71921",  # Red
    "Atlas FC": "#E31920",             # Red
    "CD Guadalajara": "#E31920",       # Red
    "Mazatlán FC": "#5E378F",          # Purple
    "Club Puebla": "#00529C"           # Blue
}



### DEFENSE
# POSSESSIONS WON VISUALIZATION

def possession_wins(df, team):


    possession_win_metrics = ["possWonDef3rd", "possWonMid3rd", "possWonAtt3rd"]

    pos_win = df.loc[(df["type"].isin(possession_win_metrics)) & (df["equipo"] == team)]
    pos_win = pos_win.groupby(["equipo", "type"])["value"].sum().reset_index()

    att = pos_win.loc[pos_win["type"] == 'possWonAtt3rd']["value"].iloc[0]
    mid = pos_win.loc[pos_win["type"] == 'possWonMid3rd']["value"].iloc[0]
    defen = pos_win.loc[pos_win["type"] == 'possWonDef3rd']["value"].iloc[0]

    st.write(f"Possessions won by {team}\n\n Att: {att}, Mid: {mid}, Def: {defen}")
    
    values = np.array([att, mid, defen])
    norm_vals = (values - values.min()) / (values.max() - values.min())

    # Set a colormap
    cmap = plt.cm.Reds

    # Thirds rectangles
    third_heights = [33.3, 33.3, 33.3]  # 1/3rd each for half pitch
    y_starts = [0, 33.3, 66.6]


    # Visualize on a vertical pitch
    pitch = Pitch(
    pitch_type='opta',  
    pad_left=0,     # Reduce padding around the pitch
    pad_right=0,
    pad_top=0,
    pad_bottom=0
)
    fig, ax = pitch.draw(figsize=(3, 4))  # << Smaller figure size!

    # Draw each third as a colored rectangle
    for i, (y_start, val) in enumerate(zip(y_starts, norm_vals)):
        color = cmap(val)  # color intensity
        rect = Rectangle(
            (y_start, 0),
            width=33.3,
            height=100,
            color=color,
            alpha=0.7,
            zorder=0,
        )
        ax.add_patch(rect)

    # Add labels
    pitch.annotate(f"{defen}\nDef Third", xy=(25, 50), ha='center', va='center', fontsize=5, ax=ax)
    pitch.annotate(f"{mid}\nMid Third", xy=(50, 50), ha='center', va='center', fontsize=5, ax=ax)
    pitch.annotate(f"{att}\nAtt Third", xy=(75, 50), ha='center', va='center', fontsize=5, ax=ax)


    st.pyplot(fig)


# Assuming df_formations and df_team_formations are preloaded
# and formation_coordinates_opta is imported




st.title("Team Formation Visualizer")
team_selected = st.selectbox("Select a Team", sorted(df_team_formations['team'].unique()))

if team_selected:
    formation = get_most_common_formation(df_team_formations, team_selected)
    
    if formation:
        st.markdown(f"### Most Common Formation: `{formation}`")
        
        likely_players = get_likely_players(df_formations, team_selected)
        formation_df = build_formation_dataframe(likely_players, formation, formation_coordinates_opta)
        
        if not formation_df.empty:
            fig = plot_formation(formation_df, team_selected, formation)
            st.pyplot(fig)
        else:
            st.warning("Not enough player data to draw the formation.")
    else:
        st.warning("No formation data found for this team.")

formation_usage_streamlit(df_team_formations, team_selected)

st.title("Defense")

possession_wins(df, team_selected)



# def map_formation_coords(formation_type, players_df):

#     players_df = players_df.loc[(~players_df["Formation Place"].isna())]

#     plot_players = []

#     for index, row in players_df.iterrows():
#         player = {'Number' : row['Shirt Number'], 'Formation Place' : row['Formation Place']}
#         plot_players.append(player)

#     positions = formation_templates.get(formation_type)
#     if not positions:
#         raise ValueError(f"Formation {formation_type} not supported.")

#     mapped = {}
#     for player in plot_players:
#         place = int(player['Formation Place']) - 1  # 0-based index
#         if 0 <= place < len(positions):
#             mapped[player['Number']] = positions[place]
#     return mapped



# def visualize_formation(mapped_positions, title="Team Formation"):
#     pitch = Pitch(pitch_type='opta', line_color='black')
#     fig, ax = pitch.draw(figsize=(10, 7))

#     for name, (x, y) in mapped_positions.items():
#         pitch.scatter(x, y, ax=ax, s=600, color='red', edgecolors='black', zorder=2)
#         pitch.annotate(name, (x, y), ax=ax, va='center', ha='center', fontsize=9, color='white', weight='bold')

#     ax.set_title(title, fontsize=14)
    
#     st.pyplot(fig)


### MATCH MOMENTUM


# def match_momentum_df(df, match_id):
#     df = df[df["Match ID"] == match_id]

#     return df

# import matplotlib.pyplot as plt


# def plot_momentum(df):
#     # Team names and IDs
#     home_team_id = df.columns[5].split("_")[0]
#     away_team_id = df.columns[6].split("_")[0]

#     # Extract minutes and momentum values
#     minutes = df["minute"]
#     home_momentum = df[f"{home_team_id}_momentum"]
#     away_momentum = -df[f"{away_team_id}_momentum"]  # negative for 4th quadrant

#     # Plotting

#     fig, ax = plt.subplots(figsize=(14, 6))

#     # Plot the bars
#     ax.bar(minutes, home_momentum, color='red', label=home_team_id)
#     ax.bar(minutes, away_momentum, color='blue', label=away_team_id)

#     ax.axhline(0, color='black', linewidth=0.8)
#     ax.set_xlabel("Minute")
#     ax.set_ylabel("Momentum")
#     ax.set_title(f"Momentum Bar Chart: {home_team_id} vs {away_team_id}")
#     ax.legend(loc='upper left')
#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     st.pyplot(fig)








# st.title("📊 Proyecto Chivas – Analisis de Rival")

# @st.cache_data
# def load_data():
#     df_team = pd.read_csv("team_stats.csv")
#     df_player = pd.read_csv("player_stats.csv")
#     df_match_momentum = pd.read_csv("match_momentum.csv")

#     return df_team, df_player, df_match_momentum

# df_team, df_player, df_match_momentum = load_data()


# equipo = st.selectbox('Seleccione un equipo', df_team["Team ID"].sort_values().unique(), index = None)  
# jornada = st.selectbox('Seleccione la jornada', df_team["Week"].sort_values().unique(), index = None)
# filtered_df_team = filter_team_data(df_team, equipo, jornada)
# filtered_df_player = filter_player_data(df_player, equipo, jornada)


# st.subheader("Team DataFrame")
# st.dataframe(filtered_df_team)


# st.subheader("Player DataFrame")
# st.dataframe(filtered_df_player)


# ## PLOT FORMATIONS


# relevant_team = filtered_df_team.iloc[0]
# st.subheader(f"{relevant_team["Match Description"]}")
# st.write(f"**Fecha:** {relevant_team["Date"]} | **Horario:** {relevant_team["Time"]}")

# formation = str(relevant_team["Formation"])
# st.subheader(f"Formation: {"-".join(formation)}")

# mapped_positions = map_formation_coords(formation, filtered_df_player)
# visualize_formation(mapped_positions, title=f"{formation} Formation")


# ## PLOT MATCH MOMENTUM
# st.subheader("Match Momentum")

# match_id = filtered_df_team["Match ID"].iloc[0]
# filtered_df_momentum = match_momentum_df(df_match_momentum, match_id)

# # st.dataframe(filtered_df_momentum)

# plot_momentum(filtered_df_momentum)






# st.subheader("Possession con balon")

# def pass_acc_plot(df):
#     labels = df[["accurateFwdZonePass", "totalFwdZonePass", "accurateBackZonePass", "totalBackZonePass"]]

#     fig, ax = plt.subplots()
#     ax.bar()
    

    







    


# else:
#     df = df_player.copy()
#     stats = [col for col in df.columns]

#     jugadores = sorted(df['Player ID'].unique())
#     jornadas = sorted(df['Week'].unique())
    

#     col1, col2 = st.columns(2)

    
#     with col1:
#         jugador_sel = st.selectbox("Player ID", ["Todos"] + jugadores)

#     with col2:
#         jornada_sel = st.selectbox("Week", ["Todos"] + jornadas)
    
#     if jugador_sel != "Todos":
#         df = df[df['Player ID'] == jugador_sel]
#     if jornada_sel != "Todas":
#         df = df[df['Week'] == int(jornada_sel)]


    




    

