from __future__ import annotations
import streamlit as st
st.set_page_config(
    page_title="Analyse TripModifications + GTFS — JSON/PB/Textproto + carte",
    layout="wide"
)

import json, csv, io, zipfile, sys, re, hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Tuple, Set
from pathlib import Path
import pandas as pd  # facultatif (diagnostics/exports)
import folium
import streamlit.components.v1 as components

# --- Version de schéma (invalide caches et cartes quand la structure/version change) ---
SCHEMA_VERSION = "2025-10-29-segments-added-canceled-v1"

# 0) Import protobuf local si dispo (gtfs_realtime_pb2.py)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import gtfs_realtime_pb2 as gtfs_local
except Exception:
    gtfs_local = None


# 1) camelCase → snake_case
_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')
def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub('_', name).lower()

def _normalize_json_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _camel_to_snake(k) if isinstance(k, str) else k
            out[nk] = _normalize_json_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_json_keys(x) for x in obj]
    return obj


# 2) Modèles
@dataclass
class StopSelector:
    stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None

@dataclass
class ReplacementStop:
    stop_id: Optional[str] = None
    id: Optional[str] = None
    stop_lat: Optional[float] = None
    stop_lon: Optional[float] = None
    travel_time_to_stop: int = 0

@dataclass
class Modification:
    start_stop_selector: Optional[StopSelector] = None
    end_stop_selector: Optional[StopSelector] = None
    replacement_stops: List[ReplacementStop] = field(default_factory=list)
    propagated_modification_delay: Optional[int] = None

@dataclass
class SelectedTrips:
    trip_ids: List[str] = field(default_factory=list)
    shape_id: Optional[str] = None

@dataclass
class TripModEntity:
    entity_id: str
    selected_trips: List[SelectedTrips]
    service_dates: List[str] = field(default_factory=list)
    start_times: List[str] = field(default_factory=list)
    modifications: List[Modification] = field(default_factory=list)

@dataclass
class GtfsStatic:
    trips: Dict[str, Dict[str, str]] = field(default_factory=dict)
    stop_times: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    stops_present: Set[str] = field(default_factory=set)
    # stop_id -> {"lat": float, "lon": float, "name": str}
    stops_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # shape_id -> [(lat, lon), ...] (shapes.txt)
    shapes_points: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

@dataclass
class TripCheck:
    trip_id: str
    exists_in_gtfs: bool
    start_seq_valid: bool
    end_seq_valid: bool
    start_seq: Optional[int] = None
    end_seq: Optional[int] = None
    notes: List[str] = field(default_factory=list)

@dataclass
class EntityReport:
    entity_id: str
    total_selected_trip_ids: int
    service_dates: List[str]
    modification_count: int
    trips: List[TripCheck]
    replacement_stops_unknown_in_gtfs: List[str] = field(default_factory=list)

@dataclass
class RtShapes:
    # shape_id -> [(lat, lon), ...] (polyline principale RT)
    shapes: Dict[str, List[Tuple[float, float]]]
    # shape_id -> [ [(lat,lon),...], ... ] segments ajoutés (turquoise)
    added_segments: Dict[str, List[List[Tuple[float, float]]]] = field(default_factory=dict)
    # shape_id -> [ [(lat,lon),...], ... ] segments annulés (violet)
    canceled_segments: Dict[str, List[List[Tuple[float, float]]]] = field(default_factory=dict)


# 3) Décodage polyline
def _sanitize_polyline(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # enlever \n, \r, \t et doubles échappements
    s = s.replace("\\n", "").replace("\\r", "").replace("\\t", "")
    s = s.replace("\\\\", "\\")
    s = re.sub(r"\s+", "", s)
    return s

def _legacy_decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    coords = []
    index, lat, lon = 0, 0, 0
    encoded = (encoded or "").strip()
    L = len(encoded)
    while index < L:
        result = 0; shift = 0
        while True:
            b = ord(encoded[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result = 0; shift = 0
        while True:
            b = ord(encoded[index]) - 63; index += 1
            result |= (b & 0x1f) << shift; shift += 5
            if b < 0x20: break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon
        coords.append((lat / 1e5, lon / 1e5))
    return coords

def _valid_coords(cs: List[Tuple[float,float]]) -> bool:
    return bool(cs) and all(-90 <= la <= 90 and -180 <= lo <= 180 for la, lo in cs)

def decode_polyline(encoded: str, mode: str = "auto") -> List[Tuple[float, float]]:
    enc = _sanitize_polyline(encoded)
    if not enc:
        return []
    try:
        import polyline as pl  # type: ignore
    except Exception:
        return _legacy_decode_polyline(enc)

    if mode == "p5":
        try:
            c5 = pl.decode(enc, precision=5)
            return c5 if _valid_coords(c5) else _legacy_decode_polyline(enc)
        except Exception:
            return _legacy_decode_polyline(enc)
    if mode == "p6":
        try:
            c6 = pl.decode(enc, precision=6)
            return c6 if _valid_coords(c6) else _legacy_decode_polyline(enc)
        except Exception:
            return _legacy_decode_polyline(enc)

    # auto
    try: c5 = pl.decode(enc, precision=5)
    except Exception: c5 = []
    try: c6 = pl.decode(enc, precision=6)
    except Exception: c6 = []
    if _valid_coords(c5) and not _valid_coords(c6): return c5
    if _valid_coords(c6) and not _valid_coords(c5): return c6
    if _valid_coords(c5) and _valid_coords(c6):
        def span(cs):
            lats = [la for la,_ in cs]; lons = [lo for _,lo in cs]
            return (max(lats)-min(lats)) + (max(lons)-min(lons))
        return c6 if span(c6) > span(c5)*1.3 else c5
    return _legacy_decode_polyline(enc)


# 4) Parsing TripMods & Shapes — JSON / PB / TEXTPROTO
def _detect_tripmods_format_bytes(b: bytes) -> str:
    head = (b[:4096] or b'')
    hs = head.lstrip()
    if hs.startswith(b'{') or hs.startswith(b'['):
        return 'json'
    try:
        txt = head.decode('utf-8', 'ignore')
    except Exception:
        return 'pb'
    if any(s in txt for s in ('entity', 'trip_modifications', 'shape', 'encoded_polyline', 'shape_id')):
        return 'textproto'
    return 'pb'

def _coerce_selector(obj: Dict[str, Any]) -> StopSelector:
    if not isinstance(obj, dict):
        return StopSelector()
    seq = obj.get('stop_sequence')
    sid = obj.get('stop_id') or None
    try:
        seq = int(seq) if seq is not None and f"{seq}".strip() != '' else None
    except Exception:
        seq = None
    return StopSelector(stop_sequence=seq, stop_id=sid if (sid or None) else None)

def _to_float(v) -> Optional[float]:
    try:
        if v is None: return None
        s = str(v).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

def _coerce_repl_stop(obj: Dict[str, Any]) -> Optional[ReplacementStop]:
    if not isinstance(obj, dict): return None
    sid = obj.get('stop_id')
    rid = obj.get('id')
    la = _to_float(obj.get('stop_lat'))
    lo = _to_float(obj.get('stop_lon'))
    t = obj.get('travel_time_to_stop', 0)
    try: t = int(t)
    except Exception: t = 0
    if (sid is None) and (la is None or lo is None):
        return None
    return ReplacementStop(stop_id=str(sid) if sid is not None else None,
                           id=str(rid) if rid not in (None, "") else None,
                           stop_lat=la, stop_lon=lo,
                           travel_time_to_stop=t)

def _coerce_selected_trips(obj: Dict[str, Any]) -> SelectedTrips:
    trips = obj.get('trip_ids') or []
    if isinstance(trips, str): trips = [trips]
    trips = [str(t).strip() for t in trips if str(t).strip()]
    shape_id = obj.get('shape_id')
    shape_id = str(shape_id) if shape_id not in (None, '') else None
    return SelectedTrips(trip_ids=trips, shape_id=shape_id)

# -- JSON --
def parse_tripmods_json(feed: Dict[str, Any]) -> List[TripModEntity]:
    entities = feed.get('entity') or []
    out: List[TripModEntity] = []
    for e in entities:
        tm = e.get('trip_modifications')
        if not tm:
            continue
        sel_raw = tm.get('selected_trips') or []
        selected = [_coerce_selected_trips(s) for s in sel_raw]
        service_dates = tm.get('service_dates') or []
        if not service_dates:
            for s in sel_raw:
                dates = s.get('service_dates')
                if dates: service_dates.extend(dates)
        service_dates = [str(d) for d in service_dates]
        start_times = [str(t) for t in tm.get('start_times') or []]
        mods: List[Modification] = []
        for m in tm.get('modifications') or []:
            repl_list = []
            for rs in m.get('replacement_stops') or []:
                r = _coerce_repl_stop(rs)
                if r: repl_list.append(r)
            start_sel = _coerce_selector(m.get('start_stop_selector') or {})
            end_sel = _coerce_selector(m.get('end_stop_selector') or {})
            mods.append(Modification(
                start_stop_selector=start_sel,
                end_stop_selector=end_sel,
                replacement_stops=repl_list,
                propagated_modification_delay=m.get('propagated_modification_delay')
            ))
        out.append(TripModEntity(
            entity_id=str(e.get('id')),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

def _collect_shapes_json(feed: Dict[str, Any], decode_mode: str) -> RtShapes:
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    added_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}
    canceled_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}
    for e in feed.get('entity', []):
        sh = e.get('shape')
        if sh and isinstance(sh, dict):
            sid = str(sh.get('shape_id') or '')
            enc = sh.get('encoded_polyline')
            if sid and enc:
                try:
                    shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception:
                    pass
            # added/canceled : string ou liste
            def _ensure_list(v):
                if v is None:
                    return []
                return v if isinstance(v, list) else [v]
            for enc2 in _ensure_list(sh.get('added_encoded_polylines')):
                try:
                    coords = decode_polyline(enc2, mode=decode_mode)
                    if len(coords) >= 2:
                        added_by_sid.setdefault(sid, []).append(coords)
                except Exception:
                    pass
            for enc2 in _ensure_list(sh.get('canceled_encoded_polylines')):
                try:
                    coords = decode_polyline(enc2, mode=decode_mode)
                    if len(coords) >= 2:
                        canceled_by_sid.setdefault(sid, []).append(coords)
                except Exception:
                    pass
    return RtShapes(shapes=shapes, added_segments=added_by_sid, canceled_segments=canceled_by_sid)

# -- Protobuf binaire --
def parse_tripmods_protobuf(data: bytes) -> List[TripModEntity]:
    proto = gtfs_local
    if proto is None:
        from google.transit import gtfs_realtime_pb2 as proto
    feed = proto.FeedMessage(); feed.ParseFromString(data)
    out: List[TripModEntity] = []
    for ent in feed.entity:
        if not hasattr(ent, 'trip_modifications'):
            continue
        tm = ent.trip_modifications
        selected: List[SelectedTrips] = []
        for s in getattr(tm, 'selected_trips', []):
            trips = list(getattr(s, 'trip_ids', []))
            shape_id = getattr(s, 'shape_id', None) or None
            selected.append(SelectedTrips(trip_ids=trips, shape_id=shape_id))
        service_dates = list(getattr(tm, 'service_dates', []))
        start_times = list(getattr(tm, 'start_times', []))
        mods: List[Modification] = []
        for m in getattr(tm, 'modifications', []):
            repl: List[ReplacementStop] = []
            for rs in getattr(m, 'replacement_stops', []):
                sid = getattr(rs, 'stop_id', None)
                rid = getattr(rs, 'id', None) if hasattr(rs, 'id') else None
                la = getattr(rs, 'stop_lat', None) if hasattr(rs, 'stop_lat') else None
                lo = getattr(rs, 'stop_lon', None) if hasattr(rs, 'stop_lon') else None
                try: la = float(la) if la is not None else None
                except Exception: la = None
                try: lo = float(lo) if lo is not None else None
                except Exception: lo = None
                tt = int(getattr(rs, 'travel_time_to_stop', 0)) if hasattr(rs, 'travel_time_to_stop') else 0
                if (sid is None) and (la is None or lo is None):
                    continue
                repl.append(ReplacementStop(
                    stop_id=sid if sid is not None else None,
                    id=rid if rid not in (None, "") else None,
                    stop_lat=la, stop_lon=lo,
                    travel_time_to_stop=tt
                ))
            start_sel = StopSelector(
                stop_sequence=getattr(m.start_stop_selector, 'stop_sequence', None) if hasattr(m, 'start_stop_selector') else None,
                stop_id=getattr(m.start_stop_selector, 'stop_id', None) if hasattr(m, 'start_stop_selector') else None,
            )
            end_sel = StopSelector(
                stop_sequence=getattr(m.end_stop_selector, 'stop_sequence', None) if hasattr(m, 'end_stop_selector') else None,
                stop_id=getattr(m.end_stop_selector, 'stop_id', None) if hasattr(m, 'end_stop_selector') else None,
            )
            mods.append(Modification(
                start_stop_selector=start_sel,
                end_stop_selector=end_sel,
                replacement_stops=repl,
                propagated_modification_delay=getattr(m, 'propagated_modification_delay', None)
            ))
        out.append(TripModEntity(
            entity_id=str(getattr(ent, 'id', '')),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=mods
        ))
    return out

def _collect_shapes_pb(data: bytes, decode_mode: str) -> RtShapes:
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    added_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}
    canceled_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}
    proto = gtfs_local
    if proto is None:
        try:
            from google.transit import gtfs_realtime_pb2 as proto
        except Exception:
            return RtShapes(shapes)
    feed = proto.FeedMessage(); feed.ParseFromString(data)
    for ent in feed.entity:
        if hasattr(ent, 'shape') and ent.shape:
            sid = str(getattr(ent.shape, 'shape_id', '') or '')
            enc = getattr(ent.shape, 'encoded_polyline', None)
            if sid and enc:
                try:
                    shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception:
                    pass
            # NEW: added/canceled (repeated string)
            if sid:
                if hasattr(ent.shape, 'added_encoded_polylines'):
                    for enc2 in getattr(ent.shape, 'added_encoded_polylines'):
                        try:
                            coords = decode_polyline(enc2, mode=decode_mode)
                            if len(coords) >= 2:
                                added_by_sid.setdefault(sid, []).append(coords)
                        except Exception:
                            pass
                if hasattr(ent.shape, 'canceled_encoded_polylines'):
                    for enc2 in getattr(ent.shape, 'canceled_encoded_polylines'):
                        try:
                            coords = decode_polyline(enc2, mode=decode_mode)
                            if len(coords) >= 2:
                                canceled_by_sid.setdefault(sid, []).append(coords)
                        except Exception:
                            pass
    return RtShapes(shapes=shapes, added_segments=added_by_sid, canceled_segments=canceled_by_sid)

# -- TEXTPROTO (dump ASCII) --
def _lines(b: bytes):
    for raw in b.decode('utf-8', 'ignore').splitlines():
        yield raw.strip()

def parse_textproto_feed(b: bytes, decode_mode: str) -> Tuple[List[TripModEntity], RtShapes]:
    ents: List[TripModEntity] = []
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    added_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}
    canceled_by_sid: Dict[str, List[List[Tuple[float, float]]]] = {}

    cur_id: Optional[str] = None
    in_tm = False
    in_shape = False
    capturing_poly = False
    capturing_added = False
    capturing_canceled = False

    tm_buf: Dict[str, Any] = {}
    shape_buf: Dict[str, Any] = {}

    def _flush_tm():
        nonlocal tm_buf, cur_id
        if not tm_buf:
            return
        selected: List[SelectedTrips] = tm_buf.get('selected_trips', [])
        service_dates = tm_buf.get('service_dates', [])
        start_times = tm_buf.get('start_times', [])
        modifications = tm_buf.get('modifications', [])
        ents.append(TripModEntity(
            entity_id=str(cur_id or ""),
            selected_trips=selected,
            service_dates=service_dates,
            start_times=start_times,
            modifications=modifications
        ))
        tm_buf = {}

    def _flush_shape():
        nonlocal shape_buf
        sid = shape_buf.get('shape_id')
        if sid:
            if shape_buf.get('encoded_polyline'):
                enc = str(shape_buf['encoded_polyline'])
                try:
                    shapes[sid] = decode_polyline(enc, mode=decode_mode)
                except Exception:
                    pass
            added_list = shape_buf.get('added_encoded_polylines') or []
            canceled_list = shape_buf.get('canceled_encoded_polylines') or []
            if added_list:
                for enc2 in added_list:
                    try:
                        coords = decode_polyline(str(enc2), mode=decode_mode)
                        if len(coords) >= 2:
                            added_by_sid.setdefault(sid, []).append(coords)
                    except Exception:
                        pass
            if canceled_list:
                for enc2 in canceled_list:
                    try:
                        coords = decode_polyline(str(enc2), mode=decode_mode)
                        if len(coords) >= 2:
                            canceled_by_sid.setdefault(sid, []).append(coords)
                    except Exception:
                        pass
        shape_buf = {}

    def _start_new_entity(new_id: Optional[str]):
        nonlocal in_tm, in_shape, capturing_poly, capturing_added, capturing_canceled, cur_id
        if in_tm: _flush_tm()
        if in_shape: _flush_shape()
        in_tm = False
        in_shape = False
        capturing_poly = capturing_added = capturing_canceled = False
        cur_id = new_id

    for line in _lines(b):
        if not line:
            continue

        # Gestion des captures multilignes
        if in_shape and (capturing_poly or capturing_added or capturing_canceled):
            if (line.startswith('id ') or line.startswith('entity')
                or line.startswith('shape_id ') or line.startswith('trip_modifications')
                or line.startswith('shape ') or line.startswith('selected_trips')
                or line.startswith('service_dates ') or line.startswith('start_times ')
                or line.startswith('modifications ') or line.startswith('start_stop_selector')
                or line.startswith('end_stop_selector') or line.startswith('replacement_stops')
                or line.startswith('encoded_polyline ')
                or line.startswith('added_encoded_polylines ')
                or line.startswith('canceled_encoded_polylines ')):
                capturing_poly = capturing_added = capturing_canceled = False
            else:
                if capturing_poly:
                    shape_buf['encoded_polyline'] = shape_buf.get('encoded_polyline', '') + line
                elif capturing_added:
                    lst = shape_buf.setdefault('added_encoded_polylines', [])
                    if lst: lst[-1] = lst[-1] + line
                elif capturing_canceled:
                    lst = shape_buf.setdefault('canceled_encoded_polylines', [])
                    if lst: lst[-1] = lst[-1] + line
                continue

        if line.startswith('entity') or line.startswith('id '):
            new_id = None
            if line.startswith('id '):
                new_id = line[3:].strip()
            else:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == 'id':
                    new_id = parts[2]
            _start_new_entity(new_id)
            continue

        if line.startswith('trip_modifications'):
            in_tm, in_shape = True, False
            capturing_poly = capturing_added = capturing_canceled = False
            tm_buf = dict(selected_trips=[], service_dates=[], start_times=[], modifications=[])
            continue

        if line.startswith('shape'):
            in_tm, in_shape = False, True
            capturing_poly = capturing_added = capturing_canceled = False
            shape_buf = {}
            continue

        if in_tm:
            if line.startswith('selected_trips'):
                continue
            if line.startswith('trip_ids '):
                ids = line[len('trip_ids '):].strip().split()
                if not tm_buf['selected_trips'] or tm_buf['selected_trips'][-1].trip_ids:
                    tm_buf['selected_trips'].append(SelectedTrips())
                tm_buf['selected_trips'][-1].trip_ids = ids
                continue
            if line.startswith('shape_id '):
                sid = line[len('shape_id '):].strip()
                if not tm_buf['selected_trips']:
                    tm_buf['selected_trips'].append(SelectedTrips())
                tm_buf['selected_trips'][-1].shape_id = sid
                continue
            if line.startswith('service_dates '):
                tm_buf['service_dates'] += line[len('service_dates '):].strip().split()
                continue
            if line.startswith('start_times '):
                tm_buf['start_times'] += line[len('start_times '):].strip().split()
                continue
            if line.startswith('modifications'):
                continue
            if line.startswith('start_stop_selector'):
                tm_buf['modifications'].append(Modification(
                    start_stop_selector=StopSelector(), end_stop_selector=StopSelector(),
                    replacement_stops=[]
                ))
                continue
            if line.startswith('end_stop_selector'):
                if not tm_buf['modifications']:
                    tm_buf['modifications'].append(Modification(
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector(),
                        replacement_stops=[]
                    ))
                continue
            if line.startswith('stop_sequence '):
                val = line[len('stop_sequence '):].strip()
                try: seq = int(val)
                except: seq = None
                if tm_buf['modifications']:
                    m = tm_buf['modifications'][-1]
                    if m.start_stop_selector and m.start_stop_selector.stop_sequence is None:
                        m.start_stop_selector.stop_sequence = seq
                    else:
                        if m.end_stop_selector is None: m.end_stop_selector = StopSelector()
                        m.end_stop_selector.stop_sequence = seq
                continue
            if line.startswith('replacement_stops'):
                if not tm_buf['modifications']:
                    tm_buf['modifications'].append(Modification(
                        start_stop_selector=StopSelector(), end_stop_selector=StopSelector(),
                        replacement_stops=[]
                    ))
                continue
            # Champs replacement_stop
            if line.startswith('stop_id '):
                sid = line[len('stop_id '):].strip()
                if tm_buf['modifications']:
                    m = tm_buf['modifications'][-1]
                    if not m.replacement_stops or (m.replacement_stops and m.replacement_stops[-1].stop_id is not None):
                        m.replacement_stops.append(ReplacementStop())
                    m.replacement_stops[-1].stop_id = sid
                continue
            if line.startswith('id '):  # id du replacement stop
                rid = line[len('id '):].strip()
                if tm_buf['modifications'] and tm_buf['modifications'][-1].replacement_stops:
                    tm_buf['modifications'][-1].replacement_stops[-1].id = rid
                continue
            if line.startswith('stop_lat '):
                sla = line[len('stop_lat '):].strip()
                la = None
                try: la = float(sla) if sla != "" else None
                except: la = None
                if tm_buf['modifications'] and tm_buf['modifications'][-1].replacement_stops:
                    tm_buf['modifications'][-1].replacement_stops[-1].stop_lat = la
                continue
            if line.startswith('stop_lon '):
                slo = line[len('stop_lon '):].strip()
                lo = None
                try: lo = float(slo) if slo != "" else None
                except: lo = None
                if tm_buf['modifications'] and tm_buf['modifications'][-1].replacement_stops:
                    tm_buf['modifications'][-1].replacement_stops[-1].stop_lon = lo
                continue
            continue

        if in_shape:
            if line.startswith('shape_id '):
                shape_buf['shape_id'] = line[len('shape_id '):].strip()
                continue
            if line.startswith('encoded_polyline '):
                enc = line[len('encoded_polyline '):]
                shape_buf['encoded_polyline'] = enc
                capturing_poly = True
                continue
            if line.startswith('added_encoded_polylines '):
                enc2 = line[len('added_encoded_polylines '):]
                shape_buf.setdefault('added_encoded_polylines', []).append(enc2)
                capturing_added = True
                continue
            if line.startswith('canceled_encoded_polylines '):
                enc2 = line[len('canceled_encoded_polylines '):]
                shape_buf.setdefault('canceled_encoded_polylines', []).append(enc2)
                capturing_canceled = True
                continue
            continue

    if in_tm: _flush_tm()
    if in_shape: _flush_shape()
    return ents, RtShapes(shapes=shapes, added_segments=added_by_sid, canceled_segments=canceled_by_sid)

def load_tripmods_bytes(file_bytes: bytes, decode_mode: str) -> Tuple[List[TripModEntity], Optional[Dict[str, Any]], RtShapes]:
    fmt = _detect_tripmods_format_bytes(file_bytes)
    if fmt == 'json':
        raw = json.loads(file_bytes.decode('utf-8'))
        feed = _normalize_json_keys(raw)
        ents = parse_tripmods_json(feed)
        shapes = _collect_shapes_json(feed, decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
            shapes.added_segments = {sid: segs for sid, segs in shapes.added_segments.items() if sid in needed_shape_ids}
            shapes.canceled_segments = {sid: segs for sid, segs in shapes.canceled_segments.items() if sid in needed_shape_ids}
        return ents, feed, shapes
    if fmt == 'textproto':
        ents, shapes = parse_textproto_feed(file_bytes, decode_mode=decode_mode)
        needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
        if needed_shape_ids:
            shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
            shapes.added_segments = {sid: segs for sid, segs in shapes.added_segments.items() if sid in needed_shape_ids}
            shapes.canceled_segments = {sid: segs for sid, segs in shapes.canceled_segments.items() if sid in needed_shape_ids}
        return ents, None, shapes
    # PB binaire
    ents = parse_tripmods_protobuf(file_bytes)
    shapes = _collect_shapes_pb(file_bytes, decode_mode)
    needed_shape_ids = {s.shape_id for e in ents for s in e.selected_trips if s.shape_id}
    if needed_shape_ids:
        shapes.shapes = {sid: poly for sid, poly in shapes.shapes.items() if sid in needed_shape_ids}
        shapes.added_segments = {sid: segs for sid, segs in shapes.added_segments.items() if sid in needed_shape_ids}
        shapes.canceled_segments = {sid: segs for sid, segs in shapes.canceled_segments.items() if sid in needed_shape_ids}
    return ents, None, shapes


# 5) Ensembles cibles (GTFS filtré)
def compute_needed_sets(ents: List[TripModEntity]) -> Tuple[Set[str], Set[str]]:
    needed_trip_ids: Set[str] = set()
    needed_stop_ids: Set[str] = set()
    for e in ents:
        for s in e.selected_trips:
            needed_trip_ids.update([str(tid) for tid in s.trip_ids if tid])
        for m in e.modifications:
            for rs in m.replacement_stops:
                if rs.stop_id:
                    needed_stop_ids.add(str(rs.stop_id))
            for sel in (m.start_stop_selector, m.end_stop_selector):
                if sel and sel.stop_id:
                    needed_stop_ids.add(str(sel.stop_id))
    return needed_trip_ids, needed_stop_ids


# 6) Chargement GTFS FILTRÉ (+ coordonnées des stops nécessaires + shapes.txt)
def load_gtfs_zip_filtered_bytes(zip_bytes: bytes, needed_trip_ids: Set[str], needed_stop_ids: Set[str]) -> GtfsStatic:
    trips: Dict[str, Dict[str, str]] = {}
    stop_times: Dict[str, List[Dict[str, str]]] = {}
    stops_present: Set[str] = set()
    stops_info: Dict[str, Dict[str, Any]] = {}
    shapes_points: Dict[str, List[Tuple[float, float]]] = {}

    if not needed_trip_ids and not needed_stop_ids:
        return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present,
                          stops_info=stops_info, shapes_points=shapes_points)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        # trips.txt
        if 'trips.txt' in zf.namelist() and needed_trip_ids:
            with zf.open('trips.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    tid = (row.get('trip_id') or '').strip()
                    if tid in needed_trip_ids:
                        trips[tid] = {k: (v or "").strip() for k, v in row.items()}
        # stop_times.txt
        stops_from_trips: Set[str] = set()
        if 'stop_times.txt' in zf.namelist() and needed_trip_ids:
            with zf.open('stop_times.txt') as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))
                for r in reader:
                    tid = (r.get('trip_id') or '').strip()
                    if tid in needed_trip_ids:
                        rec = {k: (v or "").strip() for k, v in r.items()}
                        try:
                            rec['stop_sequence'] = str(int(rec.get('stop_sequence', '').strip()))
                        except Exception:
                            rec['stop_sequence'] = ''
                        stop_times.setdefault(tid, []).append(rec)
                        sid = (rec.get('stop_id') or '').strip()
                        if sid:
                            stops_from_trips.add(sid)
            for tid, lst in stop_times.items():
                lst.sort(key=lambda x: int(x['stop_sequence'] or 0))

        # Union des stops requis
        all_needed_stop_ids: Set[str] = set(needed_stop_ids) | stops_from_trips

        # stops.txt — coordonnées
        if 'stops.txt' in zf.namelist() and all_needed_stop_ids:
            with zf.open('stops.txt') as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')):
                    sid = (row.get('stop_id') or '').strip()
                    if sid in all_needed_stop_ids:
                        try:
                            lat = float((row.get('stop_lat') or '').strip())
                            lon = float((row.get('stop_lon') or '').strip())
                        except Exception:
                            lat = lon = None
                        name = (row.get('stop_name') or '').strip()
                        if lat is not None and lon is not None:
                            stops_info[sid] = {"lat": lat, "lon": lon, "name": name}
                            stops_present.add(sid)

        # shapes.txt — tracé originel (pour les trips connus)
        needed_shape_ids: Set[str] = set()
        for tid, rec in trips.items():
            sid = rec.get('shape_id') or ''
            if sid:
                needed_shape_ids.add(sid)
        if 'shapes.txt' in zf.namelist() and needed_shape_ids:
            temp: Dict[str, List[Tuple[int, float, float]]] = {}
            with zf.open('shapes.txt') as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))
                for r in reader:
                    sid = (r.get('shape_id') or '').strip()
                    if sid in needed_shape_ids:
                        try:
                            la = float((r.get('shape_pt_lat') or '').strip())
                            lo = float((r.get('shape_pt_lon') or '').strip())
                            seq = int((r.get('shape_pt_sequence') or '').strip())
                        except Exception:
                            continue
                        temp.setdefault(sid, []).append((seq, la, lo))
            for sid, rows in temp.items():
                rows.sort(key=lambda x: x[0])
                shapes_points[sid] = [(la, lo) for _, la, lo in rows]

    return GtfsStatic(trips=trips, stop_times=stop_times, stops_present=stops_present,
                      stops_info=stops_info, shapes_points=shapes_points)


# 7) Analyse
def _seq_from_selector(sel: StopSelector, stop_times_list: List[Dict[str, str]]) -> Optional[int]:
    if sel is None:
        return None
    if sel.stop_sequence is not None:
        return sel.stop_sequence
    if sel.stop_id:
        for r in stop_times_list:
            if r.get('stop_id') == sel.stop_id:
                try: return int(r.get('stop_sequence') or 0)
                except Exception: return None
    return None

def analyze_tripmods_with_gtfs(gtfs: GtfsStatic, ents: List[TripModEntity]) -> Tuple[List[EntityReport], Dict[str, int]]:
    reports: List[EntityReport] = []
    totals = dict(total_entities=len(ents), total_trip_ids=0, total_modifications=0,
                  missing_trip_ids=0, invalid_selectors=0, unknown_replacement_stops=0)
    for e in ents:
        trip_checks: List[TripCheck] = []
        repl_unknown: List[str] = []
        tot_trip_ids = sum(len(sel.trip_ids) for sel in e.selected_trips)
        for sel in e.selected_trips:
            for trip_id in sel.trip_ids:
                exists = trip_id in gtfs.trips
                st_list = gtfs.stop_times.get(trip_id, [])
                start_ok = end_ok = False
                start_seq = end_seq = None
                notes: List[str] = []
                if not exists:
                    notes.append("trip_id absent du GTFS (filtré ou inexistant)")
                    totals["missing_trip_ids"] += 1
                else:
                    for m in e.modifications:
                        sseq = _seq_from_selector(m.start_stop_selector, st_list) if st_list else None
                        eseq = _seq_from_selector(m.end_stop_selector, st_list) if st_list else None
                        if start_seq is None and sseq is not None: start_seq = sseq
                        if end_seq is None and eseq is not None: end_seq = eseq
                        if sseq is None or eseq is None:
                            notes.append("start/end selector non résolu sur ce trip")
                            totals["invalid_selectors"] += 1
                        else:
                            start_ok = True; end_ok = True
                trip_checks.append(TripCheck(
                    trip_id=trip_id, exists_in_gtfs=exists,
                    start_seq_valid=start_ok, end_seq_valid=end_ok,
                    start_seq=start_seq, end_seq=end_seq, notes=notes
                ))
        for m in e.modifications:
            for rs in m.replacement_stops:
                sid = rs.stop_id
                if sid and sid not in gtfs.stops_present:
                    repl_unknown.append(sid)
                    totals["unknown_replacement_stops"] += 1
        totals["total_trip_ids"] += tot_trip_ids
        totals["total_modifications"] += len(e.modifications)
        reports.append(EntityReport(
            entity_id=e.entity_id,
            total_selected_trip_ids=tot_trip_ids,
            service_dates=e.service_dates,
            modification_count=len(e.modifications),
            trips=trip_checks,
            replacement_stops_unknown_in_gtfs=sorted(set(repl_unknown))
        ))
    return reports, totals


# 8) Folium — carte (détour ROUGE) + originel (VERT) + arrêts originels (BLANC/vert/rouge) + replacements (ROSE) + segments ajoutés/annulés
# 8) Folium — carte (détour ROUGE) + originel (VERT) + arrêts originels (BLANC/vert/rouge) + replacements (ROSE) + segments ajoutés/annulés
def build_folium_map_for_polyline(
    poly: List[Tuple[float, float]],
    shape_id: Optional[str] = None,
    replacement_stop_points: Optional[List[Tuple[float, float, str]]] = None,
    original_poly: Optional[List[Tuple[float, float]]] = None,
    original_stop_points: Optional[List[Tuple[float, float, str]]] = None,
    original_shape_id: Optional[str] = None,
    added_segments: Optional[List[List[Tuple[float, float]]]] = None,      # turquoise
    canceled_segments: Optional[List[List[Tuple[float, float]]]] = None,   # violet
    montreal_center: Tuple[float, float] = (45.5017, -73.5673),
    zoom_start: int = 12
):
    def _valid_ll(la, lo) -> bool:
        return (la is not None and lo is not None
                and -90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0
                and not (abs(la) < 1e-8 and abs(lo) < 1e-8))  # évite (0,0)

    if not poly or len(poly) < 2:
        return None
    latlons_poly = [(la, lo) for la, lo in poly if _valid_ll(la, lo)]
    if len(latlons_poly) < 2:
        return None

    m = folium.Map(
        location=montreal_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        min_zoom=8
    )

    # --- NOUVEAU : Création des groupes de couches (FeatureGroups) ---
    fg_detour = folium.FeatureGroup(name="Détour RT (Rouge)", show=True)
    fg_original = folium.FeatureGroup(name="Tracé Originel GTFS (Vert)", show=True)
    fg_replacements = folium.FeatureGroup(name="Arrêts Remplacement (Rose)", show=True)
    fg_added = folium.FeatureGroup(name="Segments Ajoutés (Turquoise)", show=True)
    fg_canceled = folium.FeatureGroup(name="Segments Annulés (Violet)", show=True)

    # --- MODIFIÉ : Ajout des éléments aux groupes respectifs ---

    # Détour (rouge vif) -> fg_detour
    folium.PolyLine(
        locations=latlons_poly,
        color="#f70707",
        weight=5,
        opacity=0.9,
        tooltip=f"shape_id (détour): {shape_id or 'n/a'}",
    ).add_to(fg_detour)
    folium.CircleMarker(latlons_poly[0], radius=6, color="green",
                        fill=True, fill_opacity=0.9, tooltip="Départ du détour").add_to(fg_detour)
    folium.CircleMarker(latlons_poly[-1], radius=6, color="red",
                        fill=True, fill_opacity=0.9, tooltip="Arrivée du détour").add_to(fg_detour)

    # Segments ajoutés/annulés
    all_ext_points: List[Tuple[float, float]] = []

    # MODIFIÉ : _draw_segments prend maintenant un FeatureGroup en argument
    def _draw_segments(segments, color, label, feature_group):
        nonlocal all_ext_points
        if not segments:
            return
        for seg in segments:
            seg_ll = [(la, lo) for la, lo in seg if _valid_ll(la, lo)]
            if len(seg_ll) >= 2:
                folium.PolyLine(
                    locations=seg_ll,
                    color=color,
                    weight=6,
                    opacity=0.95,
                    tooltip=f"{label} (shape_id: {shape_id or 'n/a'})",
                ).add_to(feature_group) # Ajout au groupe
                all_ext_points.extend(seg_ll)

    _draw_segments(added_segments, "#40E0D0", "Ajouté", fg_added)      # -> fg_added
    _draw_segments(canceled_segments, "#8A2BE2", "Annulé", fg_canceled)   # -> fg_canceled

    # Tracé originel (VERT) -> fg_original
    if original_poly and len(original_poly) >= 2:
        latlons_orig = [(la, lo) for la, lo in original_poly if _valid_ll(la, lo)]
        if len(latlons_orig) >= 2:
            folium.PolyLine(
                locations=latlons_orig,
                color="#2ca02c",
                weight=4,
                opacity=0.85,
                tooltip=f"Tracé originel (shapes.txt): {original_shape_id or 'n/a'}",
            ).add_to(fg_original)

    # Arrêts du tracé originel -> fg_original
    if original_stop_points:
        n = len(original_stop_points)
        for idx, (la, lo, lab) in enumerate(original_stop_points):
            if not _valid_ll(la, lo):
                continue
            if idx == 0:
                color = "green"; fill = "green"; radius = 7
            elif idx == n - 1:
                color = "red"; fill = "red"; radius = 7
            else:
                color = "#666666"; fill = "#ffffff"; radius = 5
            folium.CircleMarker(
                location=(la, lo),
                radius=radius,
                color=color,
                fill=True,
                fill_color=fill,
                fill_opacity=0.95,
                weight=2,
                tooltip=str(lab or "stop_id")
            ).add_to(fg_original)

    # Replacement stops (ROSE) -> fg_replacements
    if replacement_stop_points:
        for la, lo, lab in replacement_stop_points:
            if _valid_ll(la, lo):
                folium.CircleMarker(
                    location=(la, lo),
                    radius=7,
                    color="#ff69b4",
                    fill=True,
                    fill_color="#ff69b4",
                    fill_opacity=0.95,
                    weight=2,
                    tooltip=lab or "Arrêt de remplacement"
                ).add_to(fg_replacements)

    # --- NOUVEAU : Ajout des groupes à la carte ---
    fg_detour.add_to(m)
    fg_original.add_to(m)
    fg_replacements.add_to(m)
    fg_added.add_to(m)
    fg_canceled.add_to(m)
    
    # Emprise STRICTE sur le détour + segments ajoutés/annulés
    # (Cette logique reste inchangée)
    bbox_pts = latlons_poly + all_ext_points if all_ext_points else latlons_poly
    lats = [la for la, _ in bbox_pts]
    lons = [lo for _, lo in bbox_pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    span_lat = max(max_lat - min_lat, 1e-5)
    span_lon = max(max_lon - min_lon, 1e-5)
    pad_lat = max(span_lat * 0.15, 0.002)
    pad_lon = max(span_lon * 0.15, 0.002)
    sw = (min_lat - pad_lat, min_lon - pad_lon)
    ne = (max_lat + pad_lat, max_lon + pad_lon)
    m.fit_bounds([sw, ne], padding=(30, 30))

    # --- NOUVEAU : Ajout du sélecteur de couches ---
    folium.LayerControl(collapsed=False).add_to(m)

    return m

# 9) CACHE — split resource/data + carte HTML en cache resource
def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

@st.cache_resource(show_spinner=True)
def resource_parse_tripmods(tripmods_bytes: bytes, decode_flag: str, schema: str):
    ents, feed_json, rt_shapes = load_tripmods_bytes(tripmods_bytes, decode_mode=decode_flag)
    return ents, feed_json, rt_shapes

@st.cache_resource(show_spinner=True)
def resource_load_gtfs(gtfs_bytes: bytes, needed_trip_ids_sorted: Tuple[str, ...], needed_stop_ids_sorted: Tuple[str, ...], schema: str):
    gtfs = load_gtfs_zip_filtered_bytes(gtfs_bytes, set(needed_trip_ids_sorted), set(needed_stop_ids_sorted))
    if not hasattr(gtfs, "shapes_points") or gtfs.shapes_points is None:
        gtfs.shapes_points = {}
    return gtfs

@st.cache_resource(show_spinner=False)
def resource_build_map_html(
    shape_id_key: str,
    poly_key: Tuple[Tuple[float, float], ...],
    stops_key: Tuple[Tuple[float, float, str], ...],
    original_poly_key: Tuple[Tuple[float, float], ...],
    original_stop_points_key: Tuple[Tuple[float, float, str], ...],
    original_shape_id_key: str,
    added_segments_key: Tuple[Tuple[Tuple[float, float], ...], ...],
    canceled_segments_key: Tuple[Tuple[Tuple[float, float], ...], ...],
    schema: str
) -> str:
    """
    Construit la carte Folium et renvoie **l'HTML** (string) mis en cache.
    Les clés (...) sont des tuples hashables.
    """
    poly = [(la, lo) for (la, lo) in poly_key]
    stops = [(la, lo, lab) for (la, lo, lab) in stops_key]
    orig = [(la, lo) for (la, lo) in original_poly_key] if original_poly_key else None
    orig_stops = [(la, lo, lab) for (la, lo, lab) in original_stop_points_key] if original_stop_points_key else None
    added_segments = [[(la, lo) for (la, lo) in seg] for seg in added_segments_key] if added_segments_key else None
    canceled_segments = [[(la, lo) for (la, lo) in seg] for seg in canceled_segments_key] if canceled_segments_key else None

    fmap = build_folium_map_for_polyline(
        poly, shape_id=shape_id_key,
        replacement_stop_points=stops,
        original_poly=orig,
        original_stop_points=orig_stops,
        original_shape_id=original_shape_id_key or None,
        added_segments=added_segments,
        canceled_segments=canceled_segments
    )
    if fmap is None:
        return "<div>Carte indisponible</div>"
    return fmap.get_root().render()

@st.cache_data(show_spinner=True, ttl=86400, max_entries=32, hash_funcs={bytes: _hash_bytes})
def cache_views(tripmods_bytes: bytes, gtfs_bytes: bytes, decode_flag: str, schema: str) -> Dict[str, Any]:
    # 1) Parse TripMods
    ents, feed_json, rt_shapes = resource_parse_tripmods(tripmods_bytes, decode_flag, schema)
    needed_trip_ids, needed_stop_ids = compute_needed_sets(ents)
    tid_sorted = tuple(sorted(needed_trip_ids))
    sid_sorted = tuple(sorted(needed_stop_ids))

    # 2) GTFS filtré
    gtfs = resource_load_gtfs(gtfs_bytes, tid_sorted, sid_sorted, schema)
    present_trip_ids = sorted(gtfs.trips.keys())
    missing_trip_ids = sorted(set(tid_sorted) - set(present_trip_ids))

    # 3) Analyse
    reports, totals = analyze_tripmods_with_gtfs(gtfs, ents)
    reports_plain = [asdict(r) for r in reports]
    total_shapes = len(rt_shapes.shapes)

    # 4) Prépare les vues “Détails”
    details_tables_by_entity: Dict[str, List[Dict[str, Any]]] = {}
    temp_stops_points_by_entity: Dict[str, List[List[Any]]] = {}
    shape_for_plot_by_entity: Dict[str, Optional[str]] = {}
    original_poly_by_entity: Dict[str, List[List[float]]] = {}
    original_shape_id_by_entity: Dict[str, Optional[str]] = {}
    original_stop_points_by_entity: Dict[str, List[List[Any]]] = {}
    original_stop_ids_by_entity: Dict[str, List[str]] = {}
    added_segments_by_entity: Dict[str, List[List[List[float]]]] = {}
    canceled_segments_by_entity: Dict[str, List[List[List[float]]]] = {}

    def _is_poly_anormal(coords: List[Tuple[float,float]]) -> bool:
        if not coords or len(coords) < 2: return True
        lats = [la for la,_ in coords]; lons = [lo for _,lo in coords]
        return (max(lats)-min(lats) + max(lons)-min(lons)) < 1e-4

    stops_info = getattr(gtfs, "stops_info", {}) or {}
    shapes_pts = getattr(gtfs, "shapes_points", {}) or {}

    for r in reports_plain:  # <-- Utiliser reports_plain (la liste de dicts)
        ent_id = r['entity_id']  # <-- Accès par clé
        ent_obj = next((e for e in ents if e.entity_id == ent_id), None)
        if not ent_obj or not ent_obj.modifications:
            continue

        shape_ids_in_entity = [s.shape_id for s in ent_obj.selected_trips if s.shape_id]
        shape_id_for_plot = next((sid for sid in shape_ids_in_entity if sid in rt_shapes.shapes), None)
        shape_for_plot_by_entity[ent_id] = shape_id_for_plot

        # Segments ajoutés/annulés pour la shape RT retenue
        if shape_id_for_plot:
            add_segs = rt_shapes.added_segments.get(shape_id_for_plot, [])
            can_segs = rt_shapes.canceled_segments.get(shape_id_for_plot, [])
            added_segments_by_entity[ent_id] = [[[la, lo] for (la, lo) in seg] for seg in add_segs]
            canceled_segments_by_entity[ent_id] = [[[la, lo] for (la, lo) in seg] for seg in can_segs]
        else:
            added_segments_by_entity[ent_id] = []
            canceled_segments_by_entity[ent_id] = []

        # Tableau diagnostics
        trip_counts: Dict[str, int] = {}
        for s in ent_obj.selected_trips:
            for tid in s.trip_ids:
                trip_counts[tid] = trip_counts.get(tid, 0) + 1
        mixed_shapes = len({sid for sid in shape_ids_in_entity if sid}) > 1
        detail_rows: List[Dict[str, Any]] = []

        chosen_original_shape_id: Optional[str] = None
        chosen_original_trip_id: Optional[str] = None

        # r.trips est une liste de dict (asdict sur dataclasses)
        for t in r['trips']:  # <-- Accès par clé
            st_list = gtfs.stop_times.get(t['trip_id'], [])
            stop_times_count = len(st_list)

            trip_shape_id = None
            for s in ent_obj.selected_trips:
                if t['trip_id'] in s.trip_ids and s.shape_id:
                    trip_shape_id = s.shape_id
                    break
            if not trip_shape_id and shape_ids_in_entity:
                trip_shape_id = shape_ids_in_entity[0]

            poly = rt_shapes.shapes.get(trip_shape_id, []) if trip_shape_id else []
            poly_points = len(poly)
            poly_anormal = _is_poly_anormal(poly)

            selectors_incomplets = not (t.get('start_seq_valid') and t.get('end_seq_valid'))
            ordre_ok = ""
            ecart_seq = ""
            if (t.get('start_seq') is not None) and (t.get('end_seq') is not None):
                ordre_ok = "oui" if t.get('start_seq') <= t.get('end_seq') else "non"
                try:
                    ecart_seq = int(t.get('end_seq')) - int(t.get('start_seq'))
                except Exception:
                    ecart_seq = ""

            duplicate_trip = trip_counts.get(t['trip_id'], 0) > 1
            detail_rows.append({
                "trip_id": t['trip_id'],
                "existe dans GTFS": "oui" if t['exists_in_gtfs'] else "non",
                "start_seq": t.get('start_seq') if t.get('start_seq') is not None else "",
                "end_seq": t.get('end_seq') if t.get('end_seq') is not None else "",
                "selectors OK": "oui" if (t.get('start_seq_valid') and t.get('end_seq_valid')) else "non",
                "notes": "; ".join(t.get('notes', [])) if t.get('notes') else "",
                "shape_id (trip)": trip_shape_id or "",
                "shape dispo": "oui" if (trip_shape_id and poly_points >= 2) else "non",
                "pts polyline": poly_points,
                "polyline anormale": "oui" if poly_anormal else "non",
                "stop_times (nb)": stop_times_count,
                "ordre start<=end": ordre_ok,
                "écart seq": ecart_seq,
                "selectors incomplets": "oui" if selectors_incomplets else "non",
                "trip en double (entité)": "oui" if duplicate_trip else "non",
                "mixed shapes (entité)": "oui" if mixed_shapes else "non",
            })

            if chosen_original_shape_id is None and t['exists_in_gtfs']:
                trip_row = gtfs.trips.get(t['trip_id'], {})
                static_sid = (trip_row.get('shape_id') or '').strip()
                if static_sid and static_sid in shapes_pts and len(shapes_pts.get(static_sid, [])) >= 2:
                    chosen_original_shape_id = static_sid
                    chosen_original_trip_id = t['trip_id']

        details_tables_by_entity[ent_id] = detail_rows

        # Replacement stops (ROSE)
        tmp_points: List[List[Any]] = []
        seen_keys: Set[Tuple[float, float, str]] = set()
        for m in ent_obj.modifications:
            for rs in m.replacement_stops:
                label = (rs.id or rs.stop_id or "").strip()
                la = rs.stop_lat
                lo = rs.stop_lon
                if (la is None or lo is None) and rs.stop_id:
                    info = stops_info.get(rs.stop_id)
                    if info:
                        la = info.get("lat"); lo = info.get("lon")
                if la is None or lo is None:
                    continue
                key = (round(la, 7), round(lo, 7), label)
                if key in seen_keys:
                    continue
                tmp_points.append([la, lo, label if label else "replacement_stop"])
                seen_keys.add(key)
        temp_stops_points_by_entity[ent_id] = tmp_points

        # Tracé originel / shape_id
        if chosen_original_shape_id:
            orig = shapes_pts.get(chosen_original_shape_id, [])
            original_poly_by_entity[ent_id] = [[la, lo] for (la, lo) in orig]
            original_shape_id_by_entity[ent_id] = chosen_original_shape_id
        else:
            original_poly_by_entity[ent_id] = []
            original_shape_id_by_entity[ent_id] = None

        # Arrêts du tracé originel
        orig_pts: List[List[Any]] = []
        orig_ids: List[str] = []
        if chosen_original_trip_id:
            st_list_for_orig = gtfs.stop_times.get(chosen_original_trip_id, [])
            for rec in st_list_for_orig:
                sid = (rec.get('stop_id') or '').strip()
                if not sid:
                    continue
                info = stops_info.get(sid)
                if not info:
                    continue
                la, lo = info.get("lat"), info.get("lon")
                if la is None or lo is None:
                    continue
                orig_pts.append([la, lo, sid])
                orig_ids.append(sid)
        original_stop_points_by_entity[ent_id] = orig_pts
        original_stop_ids_by_entity[ent_id] = orig_ids

    # shapes (détour RT) → JSON‑compatibles
    shapes_plain: Dict[str, List[List[float]]] = {
        sid: [[la, lo] for (la, lo) in coords] for sid, coords in rt_shapes.shapes.items()
    }

    # KPI GTFS
    gtfs_kpi = dict(
        trips=len(gtfs.trips),
        stop_times=sum(len(v) for v in gtfs.stop_times.values()),
        stops_present=len(gtfs.stops_present),
    )

    return {
        "schema_version": schema,
        "feed_json": feed_json,
        "reports": reports_plain,
        "totals": totals,
        "total_shapes": total_shapes,
        "needed_trip_ids": list(tid_sorted),
        "needed_stop_ids": list(sid_sorted),
        "details_tables_by_entity": details_tables_by_entity,
        "temp_stops_points_by_entity": temp_stops_points_by_entity,
        "shape_for_plot_by_entity": shape_for_plot_by_entity,
        "shapes_plain": shapes_plain,
        "original_poly_by_entity": original_poly_by_entity,
        "original_shape_id_by_entity": original_shape_id_by_entity,
        "original_stop_points_by_entity": original_stop_points_by_entity,
        "original_stop_ids_by_entity": original_stop_ids_by_entity,
        "added_segments_by_entity": added_segments_by_entity,
        "canceled_segments_by_entity": canceled_segments_by_entity,
        "gtfs_kpi": gtfs_kpi,
        "present_trip_ids": present_trip_ids,
        "missing_trip_ids": missing_trip_ids,
    }


# 10) UI — formulaire + session_state
st.title("Analyse TripModifications (JSON/PB/Textproto) vs GTFS — Carte Folium")
st.caption(
    "Détour (rouge), tracé originel (vert, issu de shapes.txt), "
    "arrêts originels (blancs, avec départ en vert et terminus en rouge), "
    "arrêts de remplacement (rose, positionnés avec stop_lat/stop_lon si fournis). "
    "Segments **ajoutés (turquoise)** et **annulés (violet)** s'affichent lorsque fournis. "
    "Polylines nettoyées, analyse et diagnostics pré‑calculés et mis en cache. "
    "Carte HTML pré‑rendue et centrée strictement sur le détour."
)

with st.sidebar:
    st.header("Données d’entrée")
    with st.form("inputs_form"):
        tripmods_file = st.file_uploader("TripModifications (JSON / PB / textproto)", type=["json", "pb", "pbf", "bin", "txt"], key="tripmods_up")
        gtfs_file = st.file_uploader("GTFS (.zip)", type=["zip"], key="gtfs_up")
        decode_mode = st.selectbox("Décodage polylines", ["Auto (recommandé)", "Précision 1e-5", "Précision 1e-6"], index=0, key="decode_sel")
        dump_first = st.checkbox("Afficher le 1er trip_mod normalisé", value=False, key="dump_first_cb")
        submitted = st.form_submit_button("Analyser", type="primary")

decode_flag = {"Auto (recommandé)": "auto", "Précision 1e-5": "p5", "Précision 1e-6": "p6"}[st.session_state.get("decode_sel", "Auto (recommandé)")]
if submitted:
    if not tripmods_file or not gtfs_file:
        st.error("Merci de sélectionner **TripModifications** (JSON/PB/textproto) **et** **GTFS** (.zip).")
    else:
        tripmods_bytes = tripmods_file.getvalue()
        gtfs_bytes = gtfs_file.getvalue()
        res = cache_views(tripmods_bytes, gtfs_bytes, decode_flag, SCHEMA_VERSION)
        st.session_state["last_results"] = res
        st.session_state["last_params"] = dict(decode_flag=decode_flag, schema_version=SCHEMA_VERSION)
        st.success("Analyse terminée ✅")

# Récupération des résultats (et vérif de version de schéma)
res = st.session_state.get("last_results")
if res and res.get("schema_version") != SCHEMA_VERSION:
    res = None
    st.session_state.pop("last_results", None)

if not res:
    st.info("Charge un TripModifications (JSON / PB / textproto) puis un GTFS (.zip), choisis la précision de décodage, et clique **Analyser**.")
    st.caption("Astuce : si ta polyligne vient d’un JSON, elle doit être « déséchappée » (retrait des \\\\ et \\n).")
    st.stop()

feed_json = res["feed_json"]
reports_plain = res["reports"]; totals = res["totals"]; total_shapes = res["total_shapes"]
needed_trip_ids = res["needed_trip_ids"]; needed_stop_ids = res["needed_stop_ids"]
details_tables_by_entity = res["details_tables_by_entity"]
temp_stops_points_by_entity = res["temp_stops_points_by_entity"]
shape_for_plot_by_entity = res["shape_for_plot_by_entity"]
shapes_plain = res["shapes_plain"]
original_poly_by_entity = res["original_poly_by_entity"]
original_shape_id_by_entity = res["original_shape_id_by_entity"]
original_stop_points_by_entity = res["original_stop_points_by_entity"]
original_stop_ids_by_entity = res["original_stop_ids_by_entity"]
added_segments_by_entity = res["added_segments_by_entity"]
canceled_segments_by_entity = res["canceled_segments_by_entity"]
gtfs_kpi = res["gtfs_kpi"]

# KPIs
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Entités", totals["total_entities"])
c2.metric("trip_ids sélectionnés", totals["total_trip_ids"])
c3.metric("modifications", totals["total_modifications"])
c4.metric("trip_ids manquants", totals["missing_trip_ids"])
c5.metric("selectors non résolus", totals["invalid_selectors"])
c6.metric("repl. stops inconnus GTFS", totals["unknown_replacement_stops"])
c7.metric("Shapes dans le feed (RT)", total_shapes)

# Liste des trip_id manquants (si dispo)
if totals.get("missing_trip_ids", 0) > 0:
    missing = res.get("missing_trip_ids", [])
    if missing:
        with st.expander("🚫 Voir la liste des trip_id manquants (uniques)"):
            st.write(f"**{len(missing)}** trip_id absents du GTFS (uniques).")
            st.code("\n".join(missing), language="text")
            st.download_button(
                "⬇️ Télécharger la liste (TXT)",
                data="\n".join(missing),
                file_name="trip_ids_manquants.txt",
                mime="text/plain"
            )

st.info(f"Filtrage GTFS → trips requis: {len(needed_trip_ids):,} · stops requis: {len(needed_stop_ids):,} · shapes RT disponibles: {total_shapes:,}")
st.success(f"GTFS filtré : **{gtfs_kpi['trips']:,} trips**, **{gtfs_kpi['stop_times']:,} stop_times**, **{gtfs_kpi['stops_present']:,} stops**")
st.success(f"TripModifications : **{len(reports_plain)} entités**")

# Aperçus optionnels
if st.session_state.get("dump_first_cb") and reports_plain:
    with st.expander("Aperçu du 1er trip_mod (normalisé)"):
        st.json(reports_plain[0])

if feed_json is not None:
    with st.expander("Aperçu brut du feed JSON (après normalisation camel→snake)"):
        st.json(feed_json)

# Synthèse par entité
table = [{
    "entity_id": r["entity_id"],
    "trip_ids (sélectionnés)": r["total_selected_trip_ids"],
    "modifications": r["modification_count"],
    "service_dates": ", ".join(r["service_dates"]),
    "repl_stops inconnus": ", ".join(r["replacement_stops_unknown_in_gtfs"]) if r["replacement_stops_unknown_in_gtfs"] else ""
} for r in reports_plain if r.get("modification_count", 0) > 0]
st.subheader("Synthèse par entité")
st.dataframe(table, width="stretch", height=360)

# Détails + carte par entité
st.subheader("Détails")
for r in reports_plain[:200]:
    ent_id = r["entity_id"]
    if r["modification_count"] <= 0:
        continue

    with st.expander(f"Entité {ent_id} — {r['total_selected_trip_ids']} trips — {r['modification_count']} modifications"):
        st.write("**Dates** :", ", ".join(r["service_dates"]) if r["service_dates"] else "—")
        st.write("**Replacement stops inconnus dans GTFS (peuvent être temporaires)** :", ", ".join(r["replacement_stops_unknown_in_gtfs"]) if r["replacement_stops_unknown_in_gtfs"] else "—")

        rows = details_tables_by_entity.get(ent_id, [])
        if rows:
            st.dataframe(rows, width="stretch", height=260)
        else:
            st.info("Aucune ligne de diagnostic pour cette entité.")

        rt_shape_id_for_plot = shape_for_plot_by_entity.get(ent_id)
        orig_shape_id = original_shape_id_by_entity.get(ent_id)
        st.write(f"**Shape détour (RT)** : {rt_shape_id_for_plot or '—'}")
        st.write(f"**Shape originel (GTFS shapes.txt)** : {orig_shape_id or '—'}")

        if rt_shape_id_for_plot:
            coords_list = shapes_plain.get(rt_shape_id_for_plot, [])
            poly = [(float(la), float(lo)) for la, lo in coords_list]
            if poly and len(poly) >= 2:
                poly_key = tuple((round(la, 6), round(lo, 6)) for la, lo in poly)

                tmp_pts = temp_stops_points_by_entity.get(ent_id, [])
                stops_key = tuple((round(p[0], 6), round(p[1], 6), str(p[2])) for p in tmp_pts)

                orig_list = original_poly_by_entity.get(ent_id, [])
                orig_poly = [(float(la), float(lo)) for la, lo in orig_list]
                original_poly_key = tuple((round(la, 6), round(lo, 6)) for la, lo in orig_poly) if orig_poly else tuple()

                orig_stop_pts = original_stop_points_by_entity.get(ent_id, [])
                orig_stops_key = tuple((round(p[0], 6), round(p[1], 6), str(p[2])) for p in orig_stop_pts)

                add_list = added_segments_by_entity.get(ent_id, [])
                can_list = canceled_segments_by_entity.get(ent_id, [])
                def _segkey(lst):
                    return tuple(
                        tuple((round(p[0], 6), round(p[1], 6)) for p in seg if isinstance(p, list) and len(p) == 2)
                        for seg in lst
                    )
                added_key = _segkey(add_list)
                canceled_key = _segkey(can_list)

                map_html = resource_build_map_html(
                    rt_shape_id_for_plot or "",
                    poly_key,
                    stops_key,
                    original_poly_key,
                    orig_stops_key,
                    orig_shape_id or "",
                    added_key,
                    canceled_key,
                    SCHEMA_VERSION
                )
                components.html(map_html, height=460, scrolling=False)

                if orig_stop_pts:
                    st.markdown("**Arrêts du tracé originel (ordre `stop_times`) :**")
                    ids_list = original_stop_ids_by_entity.get(ent_id, [])
                    st.markdown("\n".join(f"{i+1}. `{sid}`" for i, sid in enumerate(ids_list)))
            else:
                st.info("Polyline (détour) vide ou invalide pour cette entité.")
        else:
            st.info("Aucune polyline 'encoded_polyline' (détour RT) disponible pour cette entité.")

# Export JSON
export_json = {
    "schema_version": SCHEMA_VERSION,
    "totals": totals,
    "total_shapes": total_shapes,
    "entities": reports_plain,
}
st.download_button("📥 Télécharger le rapport JSON",
    data=json.dumps(export_json, ensure_ascii=False, indent=2),
    file_name="rapport_tripmods.json", mime="application/json"
)
