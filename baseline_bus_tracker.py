# -*- coding: utf-8 -*-
"""
baseline_bus_tracker.py
LightGBM(구간 Δt 예측) + 1D Kalman Filter(실시간 진행률) 베이스라인

동작:
1) 실시간 XML API에서 특정 ROUTEID의 스냅샷(itemList...)을 POLL_SEC 주기로 수집
2) 연속 PATHSEQ 변화로 구간 Δt 라벨 구축
3) LightGBM으로 Δt 예측모델 학습(피처: segment_len_m, hour, dow, DIRCD, CONGESTION, pathseq_mean_dt)
4) 추론 + 1D 칼만 필터로 진행률/좌표 추정 → live_positions.json 갱신

필요 파일:
- path_nodes.csv : ROUTEID,DIRCD,PATHSEQ,lat,lon  (구간 좌표 보간용)
"""

import os, time, json, math
from dataclasses import dataclass
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import requests
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from lightgbm import LGBMRegressor

# -------------------- 사용자 설정 --------------------
ROUTE_ID = os.getenv("BUS_ROUTE_ID", "165000110")  # 실험할 노선
POLL_SEC = int(os.getenv("POLL_SEC", "5"))  # API 폴링 주기
COLLECT_MIN = int(os.getenv("COLLECT_MIN", "3"))  # 초기 수집(학습용) 분
TRACK_MIN = int(os.getenv("TRACK_MIN", "2"))  # 추적(서비스) 분
PATH_NODES_CSV = os.getenv("PATH_NODES_CSV", "path_nodes.csv")
OUT_EVENTS_CSV = "events_log.csv"
OUT_TRAIN_CSV = "train_segments.csv"
OUT_LIVE_JSON = "live_positions.json"

# 실제 엔드포인트로 교체하세요 (serviceKey, routeId, pageNo/numOfRows 포함)
API_ROUTE_BUSES = (
    "http://apis.data.go.kr/xxxx/BusLcinfoInqireService/getRouteAcctoBusLcList"
)

SERVICE_KEY = os.getenv("BUS_API_SERVICE_KEY")  # 필수

# ----------------------------------------------------


# ====== 유틸 ======
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """두 좌표간 거리(m)"""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def lerp(a, b, t):  # 0~1
    return a + (b - a) * t


# ====== 정적 경로 노드 테이블 ======
class SegmentTable:
    """
    path_nodes.csv(ROUTEID, DIRCD, PATHSEQ, lat, lon)를 읽어
    - 각 구간(PATHSEQ -> PATHSEQ+1)의 길이(m)와 양 끝 위경도 제공
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} 가 없습니다. (필수)")
        df = pd.read_csv(csv_path)
        req_cols = {"ROUTEID", "DIRCD", "PATHSEQ", "lat", "lon"}
        if not req_cols.issubset(df.columns):
            raise ValueError(f"{csv_path} 컬럼 필요: {req_cols}")

        self.df = df
        # 인덱스: (ROUTEID, DIRCD, PATHSEQ)
        self.df.set_index(["ROUTEID", "DIRCD", "PATHSEQ"], inplace=True)

    def next_key(
        self, route_id: str, dircd: int, pathseq: int
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """
        현재 PATHSEQ의 start(lat,lon), next(lat,lon), segment_len_m 반환.
        없으면 None
        """
        key = (route_id, dircd, pathseq)
        key_next = (route_id, dircd, pathseq + 1)
        if key not in self.df.index or key_next not in self.df.index:
            return None
        lat1 = float(self.df.loc[key, "lat"])
        lon1 = float(self.df.loc[key, "lon"])
        lat2 = float(self.df.loc[key_next, "lat"])
        lon2 = float(self.df.loc[key_next, "lon"])
        seglen = haversine_m(lat1, lon1, lat2, lon2)
        return lat1, lon1, lat2, lon2, seglen


# ====== API 클라이언트 ======
def fetch_route_buses(route_id: str) -> pd.DataFrame:
    """
    itemList 스냅샷 → DataFrame
    """
    if not SERVICE_KEY:
        raise RuntimeError("환경변수 BUS_API_SERVICE_KEY 가 필요합니다.")
    url = f"{API_ROUTE_BUSES}?serviceKey={SERVICE_KEY}&routeId={route_id}&pageNo=1&numOfRows=999"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    now = datetime.now()
    rows = []
    for item in root.iter("itemList"):
        rows.append(
            {
                "ts": now,
                "ROUTEID": item.findtext("ROUTEID"),
                "BUSID": item.findtext("BUSID"),
                "BUS_NUM_PLATE": item.findtext("BUS_NUM_PLATE"),
                "DIRCD": int(item.findtext("DIRCD")),
                "PATHSEQ": int(item.findtext("PATHSEQ")),
                "LATEST_STOPSEQ": int(item.findtext("LATEST_STOPSEQ")),
                "LATEST_STOP_ID": item.findtext("LATEST_STOP_ID"),
                "LATEST_STOP_NAME": item.findtext("LATEST_STOP_NAME"),
                "CONGESTION": int(item.findtext("CONGESTION")),
                "REMAIND_SEAT": int(item.findtext("REMAIND_SEAT")),
                "LASTBUSYN": int(item.findtext("LASTBUSYN")),
            }
        )
    return pd.DataFrame(rows)


# ====== 1D 칼만 필터 ======
@dataclass
class KF1D:
    seglen: float
    Q: float
    R: float
    x: float = 0.0  # 누적 거리(m)
    P: float = 1.0  # 분산

    def predict(self, v_mps: float, dt: float):
        self.x += v_mps * dt
        self.P += self.Q

    def update(self):
        z = self.seglen  # 관측: 구간 끝
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        self.x = 0.0  # 다음 구간로 리셋


# ====== ETA 모델 (LightGBM) ======
class ETAModel:
    def __init__(self):
        # P50(중앙값) & P90(상위) 두 개 퀀타일 모델
        self.q50 = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=600,
        )
        self.q90 = LGBMRegressor(
            objective="quantile",
            alpha=0.9,
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=600,
        )
        self.feats = [
            "segment_len_m",
            "hour",
            "dow",
            "DIRCD",
            "CONGESTION",
            "pathseq_mean_dt",
        ]
        self.ready = False
        self.pathseq_mean: Dict[Tuple[str, int, int], float] = {}

    def fit(self, df_train: pd.DataFrame):
        # pathseq별 평균 Δt 타깃 인코딩
        grp = df_train.groupby(["ROUTEID", "DIRCD", "PATHSEQ"])["dt_sec"].mean()
        self.pathseq_mean = grp.to_dict()
        df_train = df_train.copy()
        df_train["pathseq_mean_dt"] = df_train.apply(
            lambda r: self.pathseq_mean.get(
                (r.ROUTEID, r.DIRCD, r.PATHSEQ), grp.mean()
            ),
            axis=1,
        )
        self.q50.fit(df_train[self.feats], df_train["dt_sec"])
        self.q90.fit(df_train[self.feats], df_train["dt_sec"])
        self.ready = True

    def _featurize(self, row: pd.Series) -> np.ndarray:
        key = (row.ROUTEID, row.DIRCD, row.PATHSEQ)
        pm = self.pathseq_mean.get(
            key, np.mean(list(self.pathseq_mean.values())) if self.pathseq_mean else 5.0
        )
        f = np.array(
            [row.segment_len_m, row.hour, row.dow, row.DIRCD, row.CONGESTION, pm]
        ).reshape(1, -1)
        return f

    def predict_p50_p90(self, row: pd.Series) -> Tuple[float, float]:
        f = self._featurize(row)
        p50 = float(self.q50.predict(f)[0])
        p90 = float(self.q90.predict(f)[0])
        return max(1.0, p50), max(p50 + 1.0, p90)


# ====== 수집 → 라벨 빌드 ======
class Collector:
    """
    스냅샷을 누적해 BUSID별 PATHSEQ 변화를 감지.
    (+1 증가인 경우만 라벨로 사용)
    """

    def __init__(self):
        self.last_by_bus: Dict[str, pd.Series] = {}
        self.events = []  # 원본 이벤트 로그(스냅샷)

    def push_snapshot(self, df_snap: pd.DataFrame):
        for _, r in df_snap.iterrows():
            self.events.append(r)
        # nothing else; 라벨은 build_labels()에서

    def build_labels(self, segtbl: SegmentTable) -> pd.DataFrame:
        df = pd.DataFrame(self.events).sort_values(["BUSID", "ts"])
        rows = []
        prev: Dict[str, pd.Series] = {}
        for _, r in df.iterrows():
            bid = r.BUSID
            if bid in prev:
                a = prev[bid]
                # 같은 방향/노선만 인정
                if (r.ROUTEID == a.ROUTEID) and (r.DIRCD == a.DIRCD):
                    step = r.PATHSEQ - a.PATHSEQ
                    if step == 1:
                        # 구간 길이
                        seg = segtbl.next_key(r.ROUTEID, r.DIRCD, a.PATHSEQ)
                        if seg is None:  # 좌표가 없으면 패스
                            prev[bid] = r
                            continue
                        _, _, _, _, seglen = seg
                        dt = (r.ts - a.ts).total_seconds()
                        if 1 <= dt <= 600:  # 10분 초과/이상치 제외
                            rows.append(
                                {
                                    "ROUTEID": r.ROUTEID,
                                    "BUSID": bid,
                                    "DIRCD": r.DIRCD,
                                    "PATHSEQ": a.PATHSEQ,
                                    "segment_len_m": seglen,
                                    "dt_sec": dt,
                                    "hour": a.ts.hour,
                                    "dow": a.ts.weekday(),
                                    "CONGESTION": a.CONGESTION,
                                }
                            )
            prev[bid] = r
        out = pd.DataFrame(rows)
        return out


# ====== 좌표 보간 ======
def interpolate_latlon(lat1, lon1, lat2, lon2, progress: float) -> Tuple[float, float]:
    return (lerp(lat1, lat2, progress), lerp(lon1, lon2, progress))


# ====== 라이브 트래커 ======
@dataclass
class BusState:
    pathseq: int
    dircd: int
    last_ts: datetime
    seglen: float
    v_pred: float  # m/s
    kf: KF1D
    lat1: float
    lon1: float
    lat2: float
    lon2: float


class LiveTracker:
    def __init__(self, segtbl: SegmentTable, model: ETAModel):
        self.segtbl = segtbl
        self.model = model
        self.state: Dict[str, BusState] = {}  # BUSID -> state
        self.Q = 1.0  # 나중에 모델오차 기반 튜닝 권장
        self.R = 2.0**2

    def _start_segment(self, r: pd.Series) -> Optional[BusState]:
        seg = self.segtbl.next_key(r.ROUTEID, r.DIRCD, r.PATHSEQ)
        if seg is None:
            return None
        lat1, lon1, lat2, lon2, seglen = seg
        # 피처 생성
        row = pd.Series(
            {
                "ROUTEID": r.ROUTEID,
                "DIRCD": r.DIRCD,
                "PATHSEQ": r.PATHSEQ,
                "segment_len_m": seglen,
                "hour": r.ts.hour,
                "dow": r.ts.weekday(),
                "CONGESTION": r.CONGESTION,
            }
        )
        p50, p90 = self.model.predict_p50_p90(row)
        v_pred = seglen / max(p50, 1.0)
        kf = KF1D(seglen, Q=self.Q, R=self.R, x=0.0, P=1.0)
        return BusState(
            r.PATHSEQ, r.DIRCD, r.ts, seglen, v_pred, kf, lat1, lon1, lat2, lon2
        )

    def update_with_snapshot(self, df_snap: pd.DataFrame):
        now = datetime.now()
        # 1) 기존 상태를 예측 한 번(폴링 주기만큼)
        for bid, st in self.state.items():
            dt = POLL_SEC
            st.kf.predict(st.v_pred, dt)
            st.last_ts = now

        # 2) 새 스냅샷 반영
        for _, r in df_snap.iterrows():
            bid = r.BUSID
            if bid not in self.state:
                st = self._start_segment(r)
                if st:
                    self.state[bid] = st
                continue

            st = self.state[bid]
            # 방향이 바뀌거나 PATHSEQ가 증가했는지 확인
            if (r.DIRCD != st.dircd) or (r.PATHSEQ > st.pathseq):
                # 도착으로 간주 → KF update + 다음 세그먼트 시작
                st.kf.update()
                # 다음 세그 시작 상태
                st2 = self._start_segment(r)
                if st2:
                    self.state[bid] = st2
            else:
                # 같은 세그면 그냥 최신 ts만 갱신
                st.last_ts = r.ts

    def export_positions(self, route_id: str) -> Dict:
        """
        현재 상태를 좌표/진행률로 반환
        """
        out = {
            "routeId": route_id,
            "updatedAt": datetime.now().isoformat(),
            "buses": [],
        }
        for bid, st in self.state.items():
            progress = max(0.0, min(st.kf.x / max(st.seglen, 1.0), 1.0))
            lat, lon = interpolate_latlon(st.lat1, st.lon1, st.lat2, st.lon2, progress)
            out["buses"].append(
                {
                    "BUSID": bid,
                    "dir": st.dircd,
                    "pathseq": st.pathseq,
                    "progress": round(progress, 3),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "sigma_m": round(math.sqrt(st.kf.P), 1),
                }
            )
        return out


# ====== 메인 플로우 ======
def main():
    print(
        f"[CONFIG] ROUTE={ROUTE_ID}, POLL={POLL_SEC}s, COLLECT={COLLECT_MIN}m, TRACK={TRACK_MIN}m"
    )
    segtbl = SegmentTable(PATH_NODES_CSV)

    # ---- A. 초기 수집 (학습 데이터)
    print("[A] collecting snapshots for training...")
    coll = Collector()
    end_collect = datetime.now() + timedelta(minutes=COLLECT_MIN)
    while datetime.now() < end_collect:
        snap = fetch_route_buses(ROUTE_ID)
        if not snap.empty:
            coll.push_snapshot(snap)
        time.sleep(POLL_SEC)

    events_df = pd.DataFrame(coll.events)
    if not events_df.empty:
        events_df.to_csv(OUT_EVENTS_CSV, index=False, encoding="utf-8-sig")
        print(f"  saved snapshots -> {OUT_EVENTS_CSV} ({len(events_df)})")

    train_df = coll.build_labels(segtbl)
    if train_df.empty:
        raise RuntimeError(
            "학습 라벨이 비었습니다. POLL_SEC를 줄이거나 수집 시간을 늘려보세요."
        )
    train_df.to_csv(OUT_TRAIN_CSV, index=False, encoding="utf-8-sig")
    print(f"  built labels -> {OUT_TRAIN_CSV} ({len(train_df)})")

    # ---- B. 모델 학습
    print("[B] training LightGBM...")
    eta_model = ETAModel()
    eta_model.fit(train_df)
    print("  model ready.")

    # ---- C. 라이브 추적
    print("[C] live tracking & exporting positions...")
    tracker = LiveTracker(segtbl, eta_model)
    end_track = datetime.now() + timedelta(minutes=TRACK_MIN)
    while datetime.now() < end_track:
        snap = fetch_route_buses(ROUTE_ID)
        if not snap.empty:
            tracker.update_with_snapshot(snap)
            live = tracker.export_positions(ROUTE_ID)
            with open(OUT_LIVE_JSON, "w", encoding="utf-8") as f:
                json.dump(live, f, ensure_ascii=False, indent=2)
            print(f"  updated {OUT_LIVE_JSON} (buses={len(live['buses'])})")
        time.sleep(POLL_SEC)

    print("done.")


if __name__ == "__main__":
    main()
