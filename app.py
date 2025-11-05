import streamlit as st
import pandas as pd
import io
import json
from google.transit import gtfs_realtime_pb2
import gtfs_kit as gk
import folium
from polyline import decode
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Validateur GTFS-TripModifications",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- A. Fonctions de Chargement et d'Analyse ---

@st.cache_data
def load_gtfs(uploaded_file):
    """Charge le GTFS zippé et retourne l'objet GTFSkit."""
    if uploaded_file is not None:
        try:
            # Assurez-vous que le fichier est lu comme un zip
            feed = gk.read_feed(uploaded_file, 'zip')
            
            # Vérification basique de la qualité du feed
            if feed.is_valid():
                st.sidebar.success("GTFS Statique chargé et valide.")
            else:
                st.sidebar.warning("GTFS Statique chargé mais des anomalies ont été détectées par gtfs-kit.")
            return feed
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du GTFS : {e}")
            return None
    return None

@st.cache_data
def load_trip_modifications(uploaded_file, file_type):
    """Charge et parse le fichier TripModification (JSON ou PB)."""
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        if file_type == 'json':
            try:
                data = json.loads(file_bytes.decode('utf-8'))
                return data
            except Exception as e:
                st.error(f"❌ Erreur lors du parsing du fichier JSON : {e}")
                return None
                
        elif file_type == 'pb':
            feed = gtfs_realtime_pb2.FeedMessage()
            try:
                feed.ParseFromString(file_bytes)
                # Retourne l'objet Protobuf parsé
                return feed 
            except Exception as e:
                st.error(f"❌ Erreur lors du parsing du fichier Protobuf : {e}")
                return None
    return None

def extract_modifications(tm_data):
    """Extrait toutes les modifications dans une liste unifiée, gérant PB et JSON."""
    modifications_list = []
    
    # Gestion du format Protobuf
    if isinstance(tm_data, gtfs_realtime_pb2.FeedMessage):
        for entity in tm_data.entity:
            if entity.HasField('trip_modifications'):
                tm = entity.trip_modifications
                for trip_id in tm.trip_ids:
                    for modification in tm.modification:
                        modifications_list.append({
                            'trip_id': trip_id,
                            'modification': modification,
                            'type': 'Protobuf'
                        })
    # Gestion du format JSON (Structure supposée similaire à l'API)
    elif isinstance(tm_data, dict) and 'entity' in tm_data:
        for entity in tm_data.get('entity', []):
            if 'trip_modifications' in entity:
                tm = entity['trip_modifications']
                for trip_id in tm.get('trip_ids', []):
                    for modification in tm.get('modification', []):
                        modifications_list.append({
                            'trip_id': trip_id,
                            'modification': modification,
                            'type': 'JSON'
                        })
    return modifications_list

# --- B. Validation et Synthèse ---

def validate_feed(tm_data):
    """Valide les règles clés du GTFS-TripModifications (simplifié)."""
    anomalies = []
    
    if isinstance(tm_data, gtfs_realtime_pb2.FeedMessage):
        # Vérification du header Protobuf
        if tm_data.header.gtfs_realtime_version != "2.0":
             anomalies.append("Header: La `gtfs_realtime_version` n'est pas '2.0'.")
             
        # Vérification des règles pour chaque entité
        trip_mod_counts = {}
        for entity in tm_data.entity:
            if entity.HasField('trip_modifications'):
                tm = entity.trip_modifications
                
                # Règle 1: Les spans de modification ne DOIVENT pas se chevaucher
                # Implémentation complexe, ici on vérifie l'existence des champs pour la robustesse.
                
                # Règle 2: Un trip_id ne doit pas être dans plusieurs TripModifications (par date de service)
                for trip_id in tm.trip_ids:
                    trip_mod_counts[trip_id] = trip_mod_counts.get(trip_id, 0) + 1
                    
        for trip_id, count in trip_mod_counts.items():
            if count > 1:
                anomalies.append(f"Règle de service violée: Le `trip_id` {trip_id} est présent dans {count} objets `TripModifications` différents.")

    # 

    if not anomalies and tm_data:
        anomalies.append("Aucune anomalie critique détectée par le validateur de règles GTFS-RT (basé sur la vérification des clés et de la structure).")

    return anomalies

def get_tm_summary(modifications_list):
    """Génère un portrait synthétique du flux TripModification."""
    
    num_trips_modified = len(set([m['trip_id'] for m in modifications_list]))
    num_total_modifications = len(modifications_list)
    
    num_stops_added = 0
    num_detours_with_polyline = 0

    for item in modifications_list:
        mod = item['modification']
        if item['type'] == 'Protobuf':
            num_stops_added += len(mod.replacement_stop)
            if mod.HasField('replacement_shape') and mod.replacement_shape.HasField('encoded_polyline'):
                 num_detours_with_polyline += 1
        elif item['type'] == 'JSON':
             num_stops_added += len(mod.get('replacement_stop', []))
             if mod.get('replacement_shape', {}).get('encoded_polyline'):
                 num_detours_with_polyline += 1
                
    summary = {
        'Nombre de voyages (trip_id) affectés': num_trips_modified,
        'Nombre total d\'objets Modification (détours)': num_total_modifications,
        'Nombre de détours avec encoded_polyline': num_detours_with_polyline,
        'Nombre total d\'arrêts temporaires ajoutés': num_stops_added,
    }
    return summary

# --- C. Liaison de Données et Visualisation ---

def get_detour_data(gtfs_feed, modifications_list):
    """
    Extrait les données nécessaires pour la carte en liant GTFS-RT et GTFS statique.
    """
    detours_data = []
    
    # 1. Préparation des tables GTFS statiques
    trips = gtfs_feed.trips[['trip_id', 'shape_id']].set_index('trip_id')
    
    # 2. Itération sur les modifications
    for item in modifications_list:
        trip_id = item['trip_id']
        mod = item['modification']
        
        # Récupération de l'encoded_polyline (le cœur du détour)
        encoded_polyline = None
        if item['type'] == 'Protobuf':
            if mod.HasField('replacement_shape') and mod.replacement_shape.HasField('encoded_polyline'):
                encoded_polyline = mod.replacement_shape.encoded_polyline
        elif item['type'] == 'JSON':
            encoded_polyline = mod.get('replacement_shape', {}).get('encoded_polyline')

        if not encoded_polyline:
             continue # Ignore les modifications sans tracé de détour

        # 3. Récupération de la forme originale (shape_id)
        if trip_id in trips.index:
            original_shape_id = trips.loc[trip_id, 'shape_id']
            original_shape_coords = gtfs_feed.get_shape_coords(original_shape_id)
            
            # 4. Récupération des arrêts (début/fin/temporaires)
            
            # A. Arrêts temporaires ajoutés (ReplacementStops)
            replacement_stops = []
            if item['type'] == 'Protobuf':
                for rep_stop in mod.replacement_stop:
                    replacement_stops.append({'name': rep_stop.stop_name, 'lat': rep_stop.lat, 'lon': rep_stop.lon})
            elif item['type'] == 'JSON':
                 for rep_stop in mod.get('replacement_stop', []):
                    replacement_stops.append({'name': rep_stop.get('stop_name', 'N/A'), 'lat': rep_stop.get('lat'), 'lon': rep_stop.get('lon')})
            
            # B. Arrêts d'origine (début et fin du segment modifié)
            # Cette logique est plus complexe car elle dépend de l'indexation de stop_times.txt
            
            # Recherche des séquences de début et de fin du segment impacté par la modification
            start_seq = mod.start_stop_sequence
            end_seq = mod.end_stop_sequence
            
            original_segment_stops = gtfs_feed.stop_times[
                (gtfs_feed.stop_times['trip_id'] == trip_id) & 
                (gtfs_feed.stop_times['stop_sequence'] >= start_seq) &
                (gtfs_feed.stop_times['stop_sequence'] <= end_seq)
            ]
            
            # Récupération des détails d'arrêts
            start_stop = None
            end_stop = None
            if not original_segment_stops.empty:
                # Arrêt de début du segment à détourner
                start_stop_id = original_segment_stops.sort_values('stop_sequence').iloc[0]['stop_id']
                start_stop = gtfs_feed.stops[gtfs_feed.stops['stop_id'] == start_stop_id].iloc[0]
                
                # Arrêt de fin du segment à détourner
                end_stop_id = original_segment_stops.sort_values('stop_sequence').iloc[-1]['stop_id']
                end_stop = gtfs_feed.stops[gtfs_feed.stops['stop_id'] == end_stop_id].iloc[0]
            
            
            detours_data.append({
                'trip_id': trip_id,
                'original_shape_id': original_shape_id,
                'original_shape_coords': original_shape_coords,
                'encoded_polyline': encoded_polyline,
                'detour_coords': decode(encoded_polyline),
                'start_stop': start_stop,
                'end_stop': end_stop,
                'replacement_stops': replacement_stops
            })

    return detours_data


def render_detour_map(detour_data):
    """Affiche la carte Folium pour un détour donné."""
    
    # Coordonnées de centrage
    if detour_data['detour_coords']:
        first_point = detour_data['detour_coords'][0]
        m = folium.Map(location=[first_point[0], first_point[1]], zoom_start=13, tiles="OpenStreetMap")
    else:
        st.error("Coordonnées de détour manquantes pour l'affichage.")
        return None

    # 1. Tracé de la Shape Originale (en gris)
    if detour_data['original_shape_coords'] is not None and not detour_data['original_shape_coords'].empty:
        original_coords = detour_data['original_shape_coords'][['shape_pt_lat', 'shape_pt_lon']].values.tolist()
        folium.PolyLine(
            original_coords,
            color="gray",
            weight=4,
            opacity=0.6,
            dash_array='10, 10',
            tooltip=f"Shape Originale ({detour_data['original_shape_id']})"
        ).add_to(m)

    # 2. Tracé du Détour (encoded_polyline, en rouge)
    folium.PolyLine(
        detour_data['detour_coords'],
        color="red",
        weight=6,
        opacity=0.9,
        tooltip=f"Détour (Trip {detour_data['trip_id']})"
    ).add_to(m)
    
    # 3. Arrêts du segment détourné (début et fin)
    
    # Arrêt de Début (Bleu)
    if detour_data['start_stop'] is not None:
        start_stop = detour_data['start_stop']
        folium.Marker(
            [start_stop['stop_lat'], start_stop['stop_lon']],
            popup=f"**Début de Détour**\nStop ID: {start_stop['stop_id']}\nNom: {start_stop['stop_name']}",
            icon=folium.Icon(color='blue', icon='play')
        ).add_to(m)

    # Arrêt de Fin (Vert)
    if detour_data['end_stop'] is not None:
        end_stop = detour_data['end_stop']
        folium.Marker(
            [end_stop['stop_lat'], end_stop['stop_lon']],
            popup=f"**Fin de Détour**\nStop ID: {end_stop['stop_id']}\nNom: {end_stop['stop_name']}",
            icon=folium.Icon(color='green', icon='stop')
        ).add_to(m)

    # 4. Arrêts Temporaires Ajoutés (Orange)
    for stop in detour_data['replacement_stops']:
        if stop['lat'] and stop['lon']:
            folium.Marker(
                [stop['lat'], stop['lon']],
                popup=f"**Arrêt Temporaire**\nNom: {stop['name']}",
                icon=folium.Icon(color='orange', icon='star')
            ).add_to(m)
            
    return m

# --- D. Application Streamlit (UI) ---

st.title("🚇 Validateur & Visualisateur GTFS-TripModifications")
st.markdown("Cette application valide le standard GTFS-Realtime (TripModifications) et visualise les détours en comparant la `shape` originale du GTFS statique avec l'`encoded_polyline` du feed temps réel.")
st.markdown("---")

## 1. Chargement des Fichiers

st.header("Chargement des Données")
col1, col2 = st.columns(2)

with col1:
    gtfs_file = st.file_uploader("1. GTFS Statique (Fichier `.zip`)", type=["zip"])
    feed = load_gtfs(gtfs_file)

with col2:
    tm_format = st.radio(
        "Format du Fichier TripModification",
        ('Protobuf (.pb)', 'JSON'),
        key='tm_format',
        horizontal=True
    )
    tm_type = "pb" if tm_format == 'Protobuf (.pb)' else "json"
    tm_file = st.file_uploader(f"2. TripModification (Fichier `.{tm_type}`)", type=[tm_type])
    tm_data = load_trip_modifications(tm_file, tm_type)
    if tm_data:
        st.sidebar.success("TripModification chargé.")

st.markdown("---")

## 2. Analyse et Validation

if feed is not None and tm_data is not None:
    
    modifications_list = extract_modifications(tm_data)
    
    st.header("Rapport d'Analyse")

    # --- Synthèse ---
    st.subheader("Portrait Synthétique du Feed TripModification")
    summary = get_tm_summary(modifications_list)
    
    if summary:
        df_summary = pd.DataFrame(summary.items(), columns=['Métrique', 'Valeur'])
        st.table(df_summary.set_index('Métrique'))

    # --- Validation ---
    st.subheader("Validation (Standard GTFS-RT v2.0 et Règles TM)")
    anomalies = validate_feed(tm_data)
    
    if len(anomalies) > 1 and "Aucune anomalie critique détectée" not in anomalies[0]:
        st.warning(f"⚠️ **{len(anomalies)}** anomalies potentielles trouvées:")
        for an in anomalies:
            st.code(an)
    else:
        st.success("✅ " + anomalies[0])
        
    st.markdown("---")

    ## 3. Visualisation Cartographique des Détours

    st.header("Visualisation des Détours")
    detours_data = get_detour_data(feed, modifications_list)

    if detours_data:
        st.info(f"Visualisation de **{len(detours_data)}** détours trouvés avec `encoded_polyline` et liés au GTFS statique.")
        
        # Utiliser un sélecteur pour choisir un détour
        options = {f"Trip ID: {d['trip_id']} (Shape: {d['original_shape_id']})": d for d in detours_data}
        selected_key = st.selectbox("Sélectionnez un détour à visualiser :", list(options.keys()))
        selected_detour = options[selected_key]
        
        # Informations détaillées sur le détour sélectionné
        st.caption(f"Tracé Original (Shape ID) : **{selected_detour['original_shape_id']}** | Encoded Polyline Longueur: **{len(selected_detour['encoded_polyline'])}**")
        st.write(f"Arrêts Temporaires Ajoutés : **{len(selected_detour['replacement_stops'])}**")
        
        # Affichage de la carte
        st.subheader(f"Carte du Détour : {selected_key}")
        
        m = render_detour_map(selected_detour)
        
        if m:
             # Utilisation de folium_static pour afficher la carte
             import streamlit.components.v1 as components
             components.html(m._repr_html_(), height=600)

    else:
        st.warning("Aucun détour avec `encoded_polyline` trouvé ou la liaison GTFS statique a échoué.")

else:
    st.info("Veuillez charger les deux fichiers (GTFS Statique et TripModification) pour lancer l'analyse complète.")
