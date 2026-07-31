"""
Live dashboard for the Multi-Lane Vehicle Speed Estimation system.

Reads the CSV log + violation snapshots produced by speed_tracker.py and
displays them as a monitoring dashboard: KPIs, per-lane stats, speed
distribution, violation timeline, and a gallery of recent violations.

Usage:
    streamlit run dashboard.py -- --log violations.csv --snapshots violation_snapshots --refresh 5

(Streamlit args go after the video path as usual; the double-dash before
your own args is required so Streamlit doesn't try to parse them itself.)
"""

import argparse
import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Speed Violation Dashboard", layout="wide", page_icon="🚦")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="violations.csv")
    parser.add_argument("--snapshots", default="violation_snapshots")
    parser.add_argument("--refresh", type=int, default=5, help="Auto-refresh interval in seconds")
    # Streamlit passes its own args too; parse_known_args avoids crashing on those
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(ttl=3)
def load_data(log_path):
    if not os.path.isfile(log_path):
        return pd.DataFrame(columns=["track_id", "lane", "speed_kmh", "frame_idx", "timestamp"])
    df = pd.read_csv(log_path)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def render_kpis(df, speed_limit_hint=None):
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    avg_speed = df["speed_kmh"].mean() if not df.empty else 0
    max_speed = df["speed_kmh"].max() if not df.empty else 0
    busiest_lane = df["lane"].mode()[0] if not df.empty else "—"

    col1.metric("Total Violations", f"{total}")
    col2.metric("Avg Violation Speed", f"{avg_speed:.1f} km/h")
    col3.metric("Highest Speed Recorded", f"{max_speed:.1f} km/h")
    col4.metric("Busiest Violation Lane", busiest_lane)


def render_charts(df):
    left, right = st.columns(2)

    with left:
        st.subheader("Violations per Lane")
        if not df.empty:
            lane_counts = df["lane"].value_counts().reset_index()
            lane_counts.columns = ["lane", "count"]
            fig = px.bar(lane_counts, x="lane", y="count", color="lane",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No violations logged yet.")

    with right:
        st.subheader("Speed Distribution")
        if not df.empty:
            fig = px.histogram(df, x="speed_kmh", nbins=20,
                                color_discrete_sequence=["#E74C3C"])
            fig.update_layout(height=350, xaxis_title="Speed (km/h)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to plot yet.")

    st.subheader("Violations Over Time")
    if not df.empty:
        timeline = df.set_index("timestamp").resample("1min").size().reset_index(name="violations")
        fig = px.line(timeline, x="timestamp", y="violations", markers=True,
                      color_discrete_sequence=["#2980B9"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timeline will populate as violations are logged.")


def render_gallery(df, snapshots_dir, max_items=8):
    st.subheader("Recent Violations")
    if df.empty:
        st.info("No violations to show yet.")
        return

    recent = df.sort_values("timestamp", ascending=False).head(max_items)
    cols = st.columns(4)

    for i, (_, row) in enumerate(recent.iterrows()):
        col = cols[i % 4]
        # snapshot filenames follow track_{id}_frame_{idx}.jpg from speed_tracker.py
        candidates = [
            f for f in os.listdir(snapshots_dir)
            if f.startswith(f"track_{row['track_id']}_frame_")
        ] if os.path.isdir(snapshots_dir) else []

        with col:
            if candidates:
                img_path = os.path.join(snapshots_dir, candidates[0])
                st.image(Image.open(img_path), use_container_width=True)
            else:
                st.markdown("*(snapshot not found)*")
            st.caption(
                f"**{row['speed_kmh']:.0f} km/h** — {row['lane']} — "
                f"ID {row['track_id']} — {row['timestamp'].strftime('%H:%M:%S')}"
            )


def render_table(df):
    with st.expander("Full violation log"):
        st.dataframe(
            df.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def main():
    args = parse_args()

    st.title("🚦 Multi-Lane Vehicle Speed Violation Dashboard")
    st.caption(f"Reading `{args.log}` — auto-refreshing every {args.refresh}s")

    df = load_data(args.log)

    render_kpis(df)
    st.divider()
    render_charts(df)
    st.divider()
    render_gallery(df, args.snapshots)
    st.divider()
    render_table(df)

    time.sleep(args.refresh)
    st.rerun()


if __name__ == "__main__":
    main()