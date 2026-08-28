"""Declarative feature spec.

Counting stats are summed across squads within a player-season (so a January
transfer keeps his whole season), then converted to per-90 rates. Rate-like
columns that cannot be summed are combined as a minutes-weighted mean.
"""
from __future__ import annotations

# Counting stats per source table -> summed across squads, then rated per 90.
COUNTS: dict[str, list[str]] = {
    "standard": ["Gls", "Ast", "G_minus_PK", "PK", "PKatt",
                 "xG_Expected", "npxG_Expected", "xAG_Expected"],
    "shooting": ["Sh_Standard", "SoT_Standard", "FK_Standard"],
    "passing": ["Cmp_Total", "Att_Total", "PrgDist_Total", "Cmp_Long", "Att_Long",
                "KP", "Final_Third", "PPA", "CrsPA", "Prog"],
    "passing_types": ["Crs_Pass", "TB_Pass", "Sw_Pass", "CK_Pass"],
    "defense": ["Tkl_Tackles", "TklW_Tackles", "Att 3rd_Tackles", "Blocks_Blocks",
                "Int", "Clr", "Err", "Press_Pressures", "Succ_Pressures"],
    "possession": ["Touches_Touches", "Att Pen_Touches", "Att 3rd_Touches",
                   "Succ_Dribbles", "Att_Dribbles", "Carries_Carries",
                   "PrgDist_Carries", "Prog_Carries", "CPA_Carries",
                   "Mis_Carries", "Dis_Carries", "Prog_Receiving", "Rec_Receiving"],
    "gca": ["SCA_SCA", "GCA_GCA"],
    "misc": ["Fls", "Fld", "Off", "Recov", "Won_Aerial", "Lost_Aerial",
             "PKwon", "PKcon"],
    "playing_time": ["Subs_Subs", "Compl_Starts",
                     "onG_Team.Success", "onGA_Team.Success"],
}

# Volume columns: summed, but kept as levels rather than turned into rates.
VOLUME = {"standard": ["Min_Playing", "MP_Playing", "Starts_Playing"]}

# Rate-like columns: minutes-weighted mean across squads, kept as levels.
WEIGHTED = {
    "shooting": ["Dist_Standard"],
    "playing_time": ["PPM_Team.Success", "Min_percent_Playing.Time"],
}

# Short, readable names for the columns that survive into the model.
RENAME = {
    "Gls": "goals", "Ast": "assists", "G_minus_PK": "np_goals",
    "PK": "pens_made", "PKatt": "pens_att",
    "xG_Expected": "xg", "npxG_Expected": "npxg", "xAG_Expected": "xag",
    "Sh_Standard": "shots", "SoT_Standard": "shots_on_target",
    "FK_Standard": "fk_shots", "Dist_Standard": "avg_shot_dist",
    "Cmp_Total": "passes_cmp", "Att_Total": "passes_att",
    "PrgDist_Total": "prog_pass_dist", "Cmp_Long": "long_cmp", "Att_Long": "long_att",
    "KP": "key_passes", "Final_Third": "passes_final_third",
    "PPA": "passes_pen_area", "CrsPA": "crosses_pen_area", "Prog": "prog_passes",
    "Crs_Pass": "crosses", "TB_Pass": "through_balls", "Sw_Pass": "switches",
    "CK_Pass": "corners",
    "Tkl_Tackles": "tackles", "TklW_Tackles": "tackles_won",
    "Att 3rd_Tackles": "tackles_att_3rd", "Blocks_Blocks": "blocks",
    "Int": "interceptions", "Clr": "clearances", "Err": "errors",
    "Press_Pressures": "pressures", "Succ_Pressures": "pressures_succ",
    "Touches_Touches": "touches", "Att Pen_Touches": "touches_att_pen",
    "Att 3rd_Touches": "touches_att_3rd", "Succ_Dribbles": "dribbles_succ",
    "Att_Dribbles": "dribbles_att", "Carries_Carries": "carries",
    "PrgDist_Carries": "prog_carry_dist", "Prog_Carries": "prog_carries",
    "CPA_Carries": "carries_pen_area", "Mis_Carries": "miscontrols",
    "Dis_Carries": "dispossessed", "Prog_Receiving": "prog_passes_rec",
    "Rec_Receiving": "passes_received",
    "SCA_SCA": "sca", "GCA_GCA": "gca",
    "Fls": "fouls", "Fld": "fouled", "Off": "offsides", "Recov": "recoveries",
    "Won_Aerial": "aerials_won", "Lost_Aerial": "aerials_lost",
    "PKwon": "pk_won", "PKcon": "pk_conceded",
    "Subs_Subs": "sub_apps", "Compl_Starts": "complete_matches",
    "onG_Team.Success": "team_goals_on", "onGA_Team.Success": "team_ga_on",
    "PPM_Team.Success": "points_per_match", "Min_percent_Playing.Time": "minutes_pct",
    "Min_Playing": "minutes", "MP_Playing": "matches", "Starts_Playing": "starts",
}

# Stats FBref stopped publishing in 2022-23. Present and complete for every
# labelled season, but a 2022-23 player cannot be scored on them without imputation.
DISCONTINUED_2023 = ["pressures", "pressures_succ", "prog_carries"]

# Transfermarkt's detailed position collapsed to the seven outfield groups.
POSITION_MAP = {
    "Centre-Back": "CB",
    "Right-Back": "FB", "Left-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM", "midfield": "CM",
    "Attacking Midfield": "AM", "Second Striker": "AM",
    "Right Winger": "W", "Left Winger": "W",
    "Right Midfield": "W", "Left Midfield": "W",
    "Centre-Forward": "ST",
}
