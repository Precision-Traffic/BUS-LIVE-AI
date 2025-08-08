"""
Incheon Bus API (XML) → LightGBM + 1D Kalman Filter
"""

import requests, xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
import time, os, sys

# -----------------------------------------------------------------------------
# 0. 환경변수 / 상수
SERVICE_KEY = os.getenv("BUS_API_SERVICE_KEY")  # .env 또는 시스템 변수 권장
ROUTE_ID    = "165000111"                       # 실험용 노선
POLL_SEC    = 5                                 # API 폴링 주기(초)

# ----------------------------------------------------------------------------- 
# 1. XML → DataFrame 수집 함수
def fetch_route_events(route_id: str) -> pd.DataFrame:
    """API를 호출해 <itemList> 모든 버스 이벤트를 DataFrame으로 반환"""
    base = (
        "http://apis.data.go.kr/xxxx/BusLcinfoInqireService/"
        "getRouteAcctoBusLcList"
        f"?serviceKey={SERVICE_KEY}&routeId={route_id}&pageNo=1&numOfRows=999"
    )
    rsp = requests.get(base, timeout=10)
    rsp.raise_for_status()
    root = ET.fromstring(rsp.text)

    rows = []
    for item in root.iter("itemList"):
        rows.append(
            dict(
                ts=datetime.now(),                         # 수신시각
                ROUTEID=item.findtext("ROUTEID"),
                BUSID=item.findtext("BUSID"),
                PATHSEQ=int(item.findtext("PATHSEQ")),
                LATEST_STOP_ID=item.findtext("LATEST_STOP_ID"),
                STOP_NAME=item.findtext("LATEST_STOP_NAME"),
                DIRCD=int(item.findtext("DIRCD")),
                CONGESTION=item.findtext("CONGESTION"),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 
# 2. 이벤트 누적 & Δt 라벨링
def build_event_log(route_id: str, run_minutes=2) -> pd.DataFrame:
    """
    • POLL_SEC 간격으로 API를 호출하여 run_minutes 동안 이벤트를 수집
    • BUSID+PATHSEQ 기준으로 Δt_sec, segment_len_m 라벨 부여
    """
    raw_events = []

    end_time = datetime.now() + timedelta(minutes=run_minutes)
    while datetime.now() < end_time:
        df_now = fetch_route_events(route_id)
        if not df_now.empty:
            raw_events.append(df_now)
        time.sleep(POLL_SEC)

    if not raw_events:
        print("수집된 데이터가 없습니다.")
        sys.exit(1)

    df = pd.concat(raw_events).sort_values(["BUSID", "ts"])
    # Δt 계산
    df["Δt_sec"] = (
        df.groupby("BUSID")["ts"].diff().dt.total_seconds()
    )  # 첫 이벤트 NaN
    # segment_len_m JOIN
    seg_tbl = pd.read_csv("segment_lengths.csv")  # ROUTEID, PATHSEQ, segment_len_m
    df = df.merge(seg_tbl, on=["ROUTEID", "PATHSEQ"], how="left")
    df = df.dropna(subset=["Δt_sec", "segment_len_m"])  # 학습용 레코드만
    return df


# ----------------------------------------------------------------------------- 
# 3. LightGBM 학습 함수
def train_lgbm(train_df: pd.DataFrame) -> lgb.Booster:
    feats = ["segment_len_m", "hour", "dow"]
    train_df["hour"] = train_df["ts"].dt.hour
    train_df["dow"] = train_df["ts"].dt.dayofweek

    cutoff = train_df["ts"].quantile(0.7)
    train, valid = train_df[train_df["ts"] < cutoff], train_df[train_df["ts"] >= cutoff]

    dtrain = lgb.Dataset(train[feats], label=train["Δt_sec"])
    dvalid = lgb.Dataset(valid[feats], label=valid["Δt_sec"])

    params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=31,
        metric="l1",
        verbose=-1,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dvalid],
        early_stopping_rounds=50,
    )
    print(f"LightGBM valid MAE: {model.best_score['valid_0']['l1']:.1f}s")
    return model


# ----------------------------------------------------------------------------- 
# 4. 1D 칼만 필터 클래스
class KF1D:
    def __init__(self, seg_len, q_var, r_var):
        self.seg_len = seg_len
        self.x, self.P = 0.0, 1.0
        self.Q, self.R = q_var, r_var

    def predict(self, v, dt=1.0):
        self.x += v * dt
        self.P += self.Q

    def update(self):
        z = self.seg_len                      # 관측 = 구간길이
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        self.x = 0.0                          # 다음 구간으로 초기화


# ----------------------------------------------------------------------------- 
# 5. 실시간 추정 데모 (Rx 스트림은 생략, 단발 DataFrame으로 시뮬)
def simulate(df: pd.DataFrame, model: lgb.Booster):
    feats = ["segment_len_m", "hour", "dow"]
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek

    q_var = (df["Δt_sec"].std()) ** 2
    r_var = 2.0**2

    demo_bus = df[df["BUSID"] == df["BUSID"].iat[0]].sort_values("ts")
    kf = None

    for i, row in demo_bus.iterrows():
        # 구간 시작 → 칼만 필터 새로 준비
        if kf is None or row["PATHSEQ"] == demo_bus["PATHSEQ"].min():
            kf = KF1D(row["segment_len_m"], q_var, r_var)

        eta_pred = model.predict(row[feats].values.reshape(1, -1))[0]  # sec
        v_pred = row["segment_len_m"] / eta_pred                       # m/s

        # 1초 단위 진행률 출력
        elapsed = 0
        while elapsed < row["Δt_sec"]:
            kf.predict(v_pred, dt=1)
            progress = min(kf.x / row["segment_len_m"], 1.0)
            print(
                f"{row['ts'] + timedelta(seconds=elapsed):%H:%M:%S} "
                f"BUS {row['BUSID']} seg#{row['PATHSEQ']:02d} "
                f"progress={progress:.2f}  σ={np.sqrt(kf.P):.1f}m"
            )
            elapsed += 1

        # 정류소 도달 업데이트
        kf.update()


# ----------------------------------------------------------------------------- 
if __name__ == "__main__":
    print("▶ 1단계: XML 이벤트 수집 중…")
    events = build_event_log(ROUTE_ID, run_minutes=2)   # 2분간 샘플 수집
    print(f"수집 레코드: {len(events)}")

    print("▶ 2단계: LightGBM 학습")
    lgbm = train_lgbm(events)

    print("▶ 3단계: 실시간 추정 시뮬레이션")
    simulate(events, lgbm)
