import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import VerticalPitch, Pitch
import seaborn as sns
from matplotlib.patches import Rectangle
from visualization_functions import *
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib import cm, colors as mcolors

if os.path.exists("datasets/player_stats.parquet"):
    df_player = pd.read_parquet("datasets/player_stats.parquet")
else:
    st.error("❌ player_stats.parquet not found!")


if os.path.exists("datasets/match_events.parquet"):
    df_events = pd.read_parquet("datasets/match_events.parquet")
else:
    st.error("❌ event_stats.parquet not found!")


if os.path.exists("datasets/formations.parquet"):
    df_team_formations = pd.read_parquet("datasets/formations.parquet")
else:
    st.error("❌ formations.parquet not found!")


df_formations = df_player.loc[(df_player['type'] == "formationPlace") & (df_player['value'] > 0)]
df_formations["pos"]  = [pos[0] for pos in df_formations["pos"]]


def filtered_last_matches(df_events, num_matches, fixtures_list):
    # Step 1: Filter formations for selected team
    

    # Step 3: Get last X match_ids
    recent_match_ids = list(fixtures_list.head(num_matches))

    # Step 4: Filter df_events for those match_ids and team
    filtered_events = df_events[
        (df_events['match_id'].isin(recent_match_ids)) &
        (df_events['team_name'] == team_selected)
    ].copy()

    return filtered_events




# SAQUES DE META

def saques_df(df, team_selected):

    bg_colour = "#0E1F81"
    text_colour = "#c7d5cc"


    goal_kick_columns = [
        "match_id",
        "team_id",
        "player_id",
        "player_name",
        "event_id",
        "time_min",
        "time_sec",
        "x",                 # Starting location of goal kick
        "y",
        "Pass End X",        # Where the ball was played to
        "Pass End Y",
        "outcome",
        "event_type",        # Should be "Goal Kick" or "Pass"
        "Goal Kick",         # Boolean or label marking it as a goal kick
        "Next event Goal-Kick",  # If this is the event before a goal kick
        "GK kick from hands",    # If goalkeeper kicked from hands
        "GK X Coordinate",       # Goalkeeper location
        "GK Y Coordinate",
        "GK x coordinate time of goal",  # If goal resulted from kick
        "GK y coordinate time of goal",
        "Kick Off",           # Sometimes used as context
        "Long ball",          # If it was a long delivery
        "Launch",             # If it was launched into a specific zone
        "Touch type pass",    # Eg. foot or hand
        "Touch type clearance", # Sometimes confused with kicks
        "Direction of Play",  # Helps visualize properly
        "Zone",               # Starting zone
        "Length"
    ]

    team_filter = df["team_name"] == team_selected

    gk_df = df.loc[(df['event_type'] == 'Pass') & team_filter & ((~df["Goal Kick"].isna()) | (~df["GK kick from hands"].isna()))].copy()
    

    gk_df = gk_df[goal_kick_columns]
    
    gk_df["is_long"] = gk_df["Length"] > 32

    long_gk = (gk_df['is_long'] == True).sum()
    short_gk = (gk_df['is_long'] == False).sum()
    avg_distance = round(gk_df['Length'].mean() * 1.05, 2)

    #Complete & Incomplete GKs
    gk_complete = gk_df[(gk_df["outcome"] == 1)]
    gk_incomplete = gk_df[(gk_df["outcome"] == 0)]

    
    # Create the figure
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 1, height_ratios=[1.2, 11, 1.5], figure=fig)
    fig.patch.set_facecolor(bg_colour)

    # Header (ax1)
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    ax1.set_facecolor(bg_colour)
    ax1.text(0.01, 0.5, f"Equipo: {team_selected}", fontsize=18, color=text_colour, va='center', ha='left')
    ax1.text(0.99, 0.5, "Saques de Meta", fontsize=24, color='gold', va='center', ha='right', weight='bold')

    # Pitch (ax2)
    ax2 = fig.add_subplot(gs[1])
    pitch = Pitch(pitch_type='opta', pitch_color=bg_colour, line_color=text_colour)
    pitch.draw(ax=ax2)

    pitch.arrows(gk_complete["x"], gk_complete["y"],
                 gk_complete["Pass End X"], gk_complete["Pass End Y"], width=2,
                 headwidth=10, headlength=10, color=text_colour, ax=ax2, label='Accurate GK')

    pitch.arrows(gk_incomplete["x"], gk_incomplete["y"],
                 gk_incomplete["Pass End X"], gk_incomplete["Pass End Y"], width=2,
                 headwidth=6, headlength=5, headaxislength=12,
                 color="#f33f2f", ax=ax2, label='inaccurate GK')

    ax2.legend(facecolor=bg_colour, handlelength=5, edgecolor='None', fontsize=14, loc='upper left', labelcolor=text_colour)

    # Stats (ax3)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    ax3.set_facecolor('#22312b')
    ax3.text(0.2, 0.9, f"Saques total: {gk_df.shape[0]}", fontsize=16, color=text_colour, ha='left')
    ax3.text(0.2, 0.6, f"Saques largos: {long_gk}", fontsize=16, color=text_colour, ha='left')
    ax3.text(0.2, 0.3, f"Saques cortos: {short_gk}", fontsize=16, color=text_colour, ha='left')

    ax3.text(0.7, 0.8, f"Distancia promedia: {avg_distance} m", fontsize=16, color=text_colour, ha='center')
    # ax3.text(0.7, 0.4, f"Long GK accuracy: ", fontsize=16, color=text_colour, ha='center')

    st.pyplot(fig)









# Assuming df_formations and df_team_formations are preloaded
# and formation_coordinates_opta is imported




st.title("Chivas - Analisis de Rival")
team_selected = st.selectbox("Selcciona Equipo a Analizar", sorted(df_team_formations['team'].unique()))

#Filtros
#Rival
# rival_selected = st.selectbox("Selcciona Rival", sorted(df_team_formations['team'].unique()))
#Condicion
# condicion_selected = st.selectbox("Condición del equipo", ["Home", "Away"])
# Visual Cantidad de partidos
# Obtener fixtures únicos de ese equipo
unique_fixtures = df_team_formations[df_team_formations['team'] == team_selected]['fixture_id'].drop_duplicates()


if team_selected:
    formation = get_most_common_formation(df_team_formations, team_selected)
    
    if formation:
        st.markdown(f"### Most Common Formation: `{formation}`")
        
        likely_players = get_likely_players(df_formations, df_player, team_selected)
        formation_df = build_formation_dataframe(likely_players, formation, formation_coordinates_opta)
        
        if not formation_df.empty:
            fig = plot_formation(formation_df, team_selected, formation)
            st.pyplot(fig)
        else:
            st.warning("Not enough player data to draw the formation.")
    else:
        st.warning("No formation data found for this team.")

formation_usage_streamlit(df_team_formations, team_selected)


# Slider para seleccionar número de últimos partidos (limitado al número de fixtures únicos disponibles)
num_matches = st.slider("Seleccione cantidad de partidos anteriores", min_value=1, max_value=len(unique_fixtures), value=5)

df_events_filtered = filtered_last_matches(df_events, num_matches, unique_fixtures)

st.subheader(f"Ultimos {num_matches} Partidos seleccionados")

st.title("Ofensivo")
st.subheader("Saques de Meta")

saques_df(df_events_filtered, team_selected)




st.subheader("Zonas de Ataque")

def plot_attack_zones(df, team_name):
    # Step 1: Filter for selected team
    team_df = df[df['team_name'] == team_name].copy()

    # Step 2: Define attacking events
    attacking_events = ['Pass', 'Take On', 'Goal']

    # Step 3: Filter for successful attacking actions in the attacking half
    attack_df = team_df[
        (team_df['event_type'].isin(attacking_events)) &
        (team_df['outcome'] == 1) &
        (team_df['x'] > 50)
    ].copy()

    # Step 4: Drop rows with missing y
    attack_df = attack_df.dropna(subset=['y'])

    # Step 5: Zone classification based on y
    def classify_zone(y):
        if y < 21:
            return 'IZ'   # Inner Left Wing
        elif y < 37:
            return 'CI'   # Centre-Left
        elif y < 63:
            return 'C'    # Central
        elif y < 79:
            return 'CD'   # Centre-Right
        else:
            return 'D'    # Right Wing

    attack_df['attack_zone'] = attack_df['y'].apply(classify_zone)

    # Step 6: Calculate zone percentages
    zone_counts = (
        attack_df['attack_zone'].value_counts(normalize=True) * 100
    ).round(1).reindex(['IZ', 'CI', 'C', 'CD', 'D'], fill_value=0.0)

    # Step 7: Set up vertical pitch and color mapping
    norm = mcolors.Normalize(vmin=0, vmax=40)
    cmap = cm.RdYlGn
    pitch = VerticalPitch(pitch_type='opta', half=True, pad_left=0, pad_right=0)
    fig, ax = pitch.draw(figsize=(4, 5))

    # Step 8: Define zones (x, y, width, height)
    zones = {
        'IZ': (79, 50, 21, 50),
        'CI': (63, 50, 16, 50),
        'C' : (37, 50, 26, 50),
        'CD': (21, 50, 16, 50),
        'D' : (0,  50, 21, 50)
    }

    for zone, (x, y, w, h) in zones.items():
        pct = zone_counts.get(zone, 0)
        color = cmap(norm(pct))
        rect = Rectangle((x, y), w, h, color=color, alpha=0.4, zorder=1)
        ax.add_patch(rect)

    # Step 9: Add annotations
    annotations = {
        'IZ': (75, 85),
        'CI': (75, 70),
        'C' : (75, 50),
        'CD': (75, 30),
        'D' : (75, 15)
    }

    for zone, (x_anno, y_anno) in annotations.items():
        pitch.annotate(f"{zone_counts[zone]}%\n{zone}", xy=(x_anno, y_anno),
                       ha='center', va='center', fontsize=6, ax=ax)

    # Step 10: Prepare summary table
    zone_table = pd.DataFrame({
        'Zone': zone_counts.index,
        'Action %': zone_counts.values
    })

    return fig, zone_table


fig, zone_table = plot_attack_zones(df_events_filtered, team_selected)

st.pyplot(fig)





st.subheader("Passes")

def passes_filter_df(df, team_selected, zone_selected, player_selected):
    # 1. Require team selection
    if not team_selected:
        st.warning("Please select a team.")
        st.stop()

    # 2. Filter only pass events
    pass_df = df[df["event_type"] == "Pass"]

    # 3. Column selection
    columns_to_keep = [
        'match_id', 'team_id', 'player_id', 'player_name',
        'x', 'y', 'Pass End X', 'Pass End Y', 'outcome',
        'time_min', 'time_sec', 'event_id', "Zone", "Length", "team_name"
    ]
    pass_df = pass_df[columns_to_keep]

    # 4. Apply filters
    team_filter = pass_df['team_name'] == team_selected

    # Show all zones if none selected
    if not zone_selected:
        zone_filter = pass_df['Zone'].notna()
    else:
        zone_filter = pass_df['Zone'].isin(zone_selected)

    # Show all players if none selected
    if not player_selected:
        player_filter = pass_df['player_name'].notna()
    else:
        player_filter = pass_df['player_name'].isin(player_selected)

    # 5. Final filtered DF
    pass_df = pass_df.loc[team_filter & zone_filter & player_filter].reset_index(drop=True)

    return pass_df


def pass_df_treatment(passes_df):

    bg_colour = "#0E1F81"
    text_colour = "#c7d5cc"



    # Filter passes
    pass_df_complete = passes_df[passes_df["outcome"] == 1]
    pass_df_incomplete = passes_df[passes_df["outcome"] == 0]

    fwd_pass_df = passes_df[passes_df["Pass End X"] > passes_df["x"]]
    bck_pass_df = passes_df[passes_df["x"] >= passes_df["Pass End X"]]

    passes_attempted = passes_df.shape[0]
    passes_completed = pass_df_complete.shape[0]
    # passing_accuracy = round(passes_completed / passes_attempted * 100, 2)
    fwd_passes = fwd_pass_df.shape[0]
    back_passes = bck_pass_df.shape[0]

    # Create the figure
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 1, height_ratios=[1.2, 11, 1.5], figure=fig)
    fig.patch.set_facecolor(bg_colour)

    # Header (ax1)
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    ax1.set_facecolor(bg_colour)
    ax1.text(0.01, 0.5, f"Equipo: {team_selected}", fontsize=18, color=text_colour, va='center', ha='left')
    ax1.text(0.5, 0.5, f"Zona(s): {zone_selected}", fontsize=18, color=text_colour, va='center', ha='center')
    ax1.text(0.99, 0.5, "Mapa de Pases", fontsize=24, color='gold', va='center', ha='right', weight='bold')

    # Pitch (ax2)
    ax2 = fig.add_subplot(gs[1])
    pitch = Pitch(pitch_type='opta', pitch_color=bg_colour, line_color=text_colour)
    pitch.draw(ax=ax2)

    pitch.arrows(pass_df_complete["x"], pass_df_complete["y"],
                 pass_df_complete["Pass End X"], pass_df_complete["Pass End Y"], width=2,
                 headwidth=10, headlength=10, color=text_colour, ax=ax2, label='Pases completados')

    pitch.arrows(pass_df_incomplete["x"], pass_df_incomplete["y"],
                 pass_df_incomplete["Pass End X"], pass_df_incomplete["Pass End Y"], width=2,
                 headwidth=6, headlength=5, headaxislength=12,
                 color="#f33f2f", ax=ax2, label='Pases incompletos')

    ax2.legend(facecolor=bg_colour, handlelength=5, edgecolor='None', fontsize=14, loc='upper left', labelcolor=text_colour)

    # Stats (ax3)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    ax3.set_facecolor('#22312b')
    ax3.text(0.2, 0.9, f"Passes Attempted: {passes_attempted}", fontsize=16, color=text_colour, ha='left')
    ax3.text(0.2, 0.6, f"Passes Completed: {passes_completed}", fontsize=16, color=text_colour, ha='left')
    # ax3.text(0.2, 0.3, f"Accuracy: {passing_accuracy}%", fontsize=16, color=text_colour, ha='left')

    ax3.text(0.7, 0.8, f"Forward Passes: {fwd_passes}", fontsize=16, color=text_colour, ha='center')
    ax3.text(0.7, 0.4, f"Backward Passes: {back_passes}", fontsize=16, color=text_colour, ha='center')

    st.pyplot(fig)

zone_selected = st.multiselect("Selcciona la zona", options=(df_events["Zone"].unique()), default=None)
player_selected = st.multiselect("Selcciona los jugadores", options=(df_events[df_events["team_name"] == team_selected]["player_name"].unique()), default=None)


pass_df = passes_filter_df(df_events_filtered, team_selected, zone_selected, player_selected)
st.dataframe(pass_df)
pass_df_treatment(pass_df)




def plot_defensive_stats(df, team_name, selected_events=None):
    df.columns = df.columns.str.strip()
    df_team = df[df["team_name"] == team_name]
    def_df = df_team[df_team["Defensive"] == 1].copy()

    all_event_types = list(def_df["event_type"].unique())
    if selected_events:
        def_df = def_df[def_df["event_type"].isin(selected_events)]

    duel_df = def_df.copy().dropna(subset=["x", "y", "outcome", "event_type"])

    pitch = Pitch(pitch_type='opta', line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 6))

    event_markers = {
        'Aerial': 'o',
        'Foul': 'x',
        'Tackle': '^',
        'Blocked Pass': 's',
        'Dispossessed': 'D',
        'Challenge': 'P',
        'Take On': '*',
        'Interception': 'H'
    }
    colors = {1: 'green', 0: 'red'}

    for event_type in duel_df["event_type"].unique():
        for outcome in [0, 1]:
            subset = duel_df[(duel_df['event_type'] == event_type) & (duel_df['outcome'] == outcome)]
            pitch.scatter(subset['x'], subset['y'], ax=ax,
                          marker=event_markers.get(event_type, 'o'),
                          color=colors[outcome],
                          label=f'{event_type} - {"Success" if outcome == 1 else "Fail"}',
                          alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.set_title("Defensive Events and Outcomes", fontsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    attempts = def_df.groupby("event_type").size().rename("Attempted")
    successes = def_df[def_df["outcome"] == 1].groupby("event_type").size().rename("Successful")
    def_stats = pd.concat([attempts, successes], axis=1).fillna(0)
    def_stats["Success %"] = round((def_stats["Successful"] / def_stats["Attempted"]) * 100, 2)
    def_stats = def_stats.reset_index().rename(columns={"event_type": "Event Type"})

    return fig, def_stats, all_event_types


def plot_attacking_stats(df, team_name, selected_events=None):
    df.columns = df.columns.str.strip()
    df_team = df[df["team_name"] == team_name]

    att_list = list(df_team.loc[(~df_team["Offensive"].isna()), "event_type"].unique())
    att_list = [event for event in att_list if event not in ["Pass", "Deleted event"]]

    if selected_events:
        att_list = selected_events

    att_df = df_team[df_team["event_type"].isin(att_list)].copy()
    att_df = att_df.dropna(subset=["x", "y", "outcome", "event_type"])

    pitch = Pitch(pitch_type='opta', line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 6))

    event_markers = {
        'Aerial': 'o',
        'Foul': '^',
        'Take On': '*',
        'Tackle': 'D',
        'Challenge': 'P',
        'Dispossessed': 'v'
    }
    colors = {1: 'green', 0: 'red'}

    for event_type in att_df["event_type"].unique():
        for outcome in [0, 1]:
            subset = att_df[(att_df['event_type'] == event_type) & (att_df['outcome'] == outcome)]
            pitch.scatter(
                subset["x"], subset["y"], ax=ax,
                color=colors[outcome],
                marker=event_markers.get(event_type, 'o'),
                alpha=0.6, label=f"{event_type} - {'Success' if outcome == 1 else 'Fail'}",
                edgecolors='black', linewidth=0.5
            )

    ax.set_title("Attacking Events and Outcomes", fontsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    attempts = att_df.groupby("event_type").size().rename("Attempted")
    successes = att_df[att_df["outcome"] == 1].groupby("event_type").size().rename("Successful")
    att_stats = pd.concat([attempts, successes], axis=1).fillna(0)
    att_stats["Success %"] = round((att_stats["Successful"] / att_stats["Attempted"]) * 100, 2)
    att_stats = att_stats.reset_index().rename(columns={"event_type": "Event Type"})

    return fig, att_stats, list(set(att_list))



# ---- Streamlit UI ----
st.title("Team Event Stats Visualizer")


# Choose visualisation type
vis_type = st.radio("Choose visualisation type:", ["Defensive", "Attacking"])



if vis_type == "Defensive":
    _, full_stats_df, all_def_events = plot_defensive_stats(df_events, team_selected)
    selected_def_events = st.multiselect("Select defensive event types to show on pitch:", all_def_events, default=all_def_events)
    fig, filtered_stats_df, _ = plot_defensive_stats(df_events, team_selected, selected_def_events)

else:
    _, full_stats_df, all_att_events = plot_attacking_stats(df_events, team_selected)
    selected_att_events = st.multiselect("Select attacking event types to show on pitch:", all_att_events, default=all_att_events)
    fig, filtered_stats_df, _ = plot_attacking_stats(df_events, team_selected, selected_att_events)

# Add stats table to the visualization itself
# Select top 5-8 rows to fit cleanly
table_df = filtered_stats_df.head(8)  # adjust as needed for space

ax = fig.axes[0]  # get the main axis from the pitch

# Display plot with table
st.pyplot(fig)

# Show complete table separately (not filtered)
st.subheader("Full Summary Table")
st.dataframe(full_stats_df)

















# def big_chance(df, team):
#     big_chances = ["bigChanceCreated", "bigChanceScored", "bigChanceMissed"]

#     big_chances = df.loc[(df["type"].isin(big_chances)) & (df["equipo"] == team)]
#     big_chances = big_chances.groupby(["equipo", "type"])["value"].sum().reset_index()

#     st.dataframe(big_chances)


# def set_pieces(df, team):
#     corners = ["totalCornersIntobox", "accurateCornersIntobox"]

#     fk_indirecto = ["freekickCross", "accurateFreekickCross"]

#     fk_directo = ["attFreekickTotal", "attFreekickTarget", "attFreekickPost", "attFreekickGoal", "attFreekickMiss"]

#     saque_banda = ["totalThrows", "accurateThrows"]


#     corners_df = df.loc[(df["type"].isin(corners)) & (df["equipo"] == team)]
#     corner_takers = corners_df.groupby(["jugador", "type"])["value"].sum().reset_index().sort_values(by="value", ascending=False)
#     corners_df = corners_df.groupby(["equipo", "type"])["value"].sum().reset_index()

#     indirecto_df = df.loc[(df["type"].isin(fk_indirecto)) & (df["equipo"] == team)]
#     fk_takers = indirecto_df.groupby(["jugador", "type"])["value"].sum().reset_index().sort_values(by="value", ascending=False)
#     indirecto_df = indirecto_df.groupby(["equipo", "type"])["value"].sum().reset_index()

#     st.dataframe(corner_takers)
#     st.dataframe(corners_df)

#     st.dataframe(fk_takers)
#     st.dataframe(indirecto_df)

