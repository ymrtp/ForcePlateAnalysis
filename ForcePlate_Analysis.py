import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import time
import io
import imageio.v2 as imageio  # 警告回避のため v2 としてインポート

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks
from functools import lru_cache  # ここではキャッシュは使わない（引数に非ハッシュ可能な ds_data が入るため）
import concurrent.futures
import imageio

# 出力用フォルダの定義（存在しない場合は作成）
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ページ全体の設定
st.set_page_config(page_title="Force Plate Analysis", layout="wide")
st.title("Force Plate Analysis")

# ─────────────────────────────
# サイドバー：入力パラメータ
# ─────────────────────────────
st.sidebar.header("Input Parameters")

# CSVファイルアップロード（skiprows=9 を想定）
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload a CSV file.")
    st.stop()

# CSV ファイルの読み込み（skiprows=9）
try:
    new_data = pd.read_csv(uploaded_file, skiprows=9)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# MaxF の入力（任意；空欄ならCSVから計算した全体最大値を使用）
maxF_input = st.sidebar.text_input("MaxF (optional, in Newton)", value="")
if maxF_input.strip() == "":
    MaxF = None
else:
    try:
        MaxF = float(maxF_input)
    except Exception as e:
        st.error(f"MaxF の値が不正です: {e}")
        MaxF = None

# CSV読み込み後に、時間の最小値・最大値を取得
min_sec = new_data.iloc[:, 0].min()
max_sec = new_data.iloc[:, 0].max()

# 開始秒、終了秒の入力（単位：秒）
start_sec = st.sidebar.number_input("Start Second", value=min_sec, step=0.1)
end_sec = st.sidebar.number_input("End Second", value=max_sec, step=0.1)

# ファイル名（アップロードされたファイル名）と拡張子除去
file_name = uploaded_file.name
file_name_without_ext = file_name.replace(".csv", "")

# 時間フィルタリング（CSV の 1列目が時間と仮定）
if start_sec is not None and end_sec is not None:
    filtered_data = new_data[(new_data.iloc[:, 0] >= float(start_sec)) & (new_data.iloc[:, 0] <= float(end_sec))]
    title_time_range = f"(Time: {start_sec}s - {end_sec}s)"
else:
    filtered_data = new_data
    title_time_range = "(Full Range)"

# ─────────────────────────────
# プレート設定の入力
# ─────────────────────────────
st.sidebar.header("Plate Configuration")
num_plates = st.sidebar.number_input("Number of Plates", min_value=1, max_value=10, value=4, step=1)
plate_centers = []
for i in range(int(num_plates)):
    st.sidebar.write(f"Plate {i+1} Center Coordinates:")
    default_x = -200 if i % 2 == 0 else 200
    default_y = -300 if i < 2 else 300
    center_x = st.sidebar.number_input(f"Plate {i+1} Center X", value=default_x, step=1, key=f"center_x_{i}")
    center_y = st.sidebar.number_input(f"Plate {i+1} Center Y", value=default_y, step=1, key=f"center_y_{i}")
    plate_centers.append((center_x, center_y))
invert_x = st.sidebar.checkbox("Invert X values", value=False)
invert_y = st.sidebar.checkbox("Invert Y values", value=True)



# ─────────────────────────────
# 関数定義（歩行開始解析用）
# ─────────────────────────────
def calculate_average_for_plate(norm_f_custom_all, plate_column, time_column, start_sec, duration_sec):
    """指定時間範囲の平均値を計算"""
    filtered = norm_f_custom_all[
        (norm_f_custom_all[time_column] >= start_sec - duration_sec) &
        (norm_f_custom_all[time_column] < start_sec)
    ]
    avg = filtered[plate_column].mean()
    return avg

def analyze_norm_f_custom_with_averages(filtered_data, normalization_value, prominence=5, distance=100, duration_sec=0.5):
    """
    各プレートの正規化データ norm_f_custom を計算し、
    ピーク・谷の検出とその直前の平均値との差（変化量）を算出する。
    """
    results = []
    norm_f_custom_all = pd.DataFrame()
    norm_f_custom_all['Time'] = filtered_data.iloc[:, 0]
    time_lag = 0.5  # 平均値計算に用いる時間のずれ

    for plate in range(4):
        fx = filtered_data.iloc[:, plate * 6 + 1]
        fy = filtered_data.iloc[:, plate * 6 + 2]
        fz = filtered_data.iloc[:, plate * 6 + 3]
        norm_f = np.sqrt(fx**2 + fy**2 + fz**2)
        norm_f_custom = (norm_f / normalization_value) * 100
        norm_f_custom_all[f'Plate_{plate + 1}'] = norm_f_custom

        # ピーク（山）と谷の検出
        peaks, _ = find_peaks(norm_f_custom, prominence=prominence, distance=distance)
        troughs, _ = find_peaks(-norm_f_custom, prominence=prominence, distance=distance)

        peak_values = norm_f_custom.iloc[peaks]
        peak_times = filtered_data.iloc[peaks, 0]
        trough_values = norm_f_custom.iloc[troughs]
        trough_times = filtered_data.iloc[troughs, 0]

        for time_val, value in zip(peak_times, peak_values):
            avg = calculate_average_for_plate(norm_f_custom_all, f'Plate_{plate + 1}', 'Time', time_val - time_lag, duration_sec)
            results.append({
                "Plate": plate + 1,
                "Type": "Peak",
                "Time (s)": time_val,
                "Value (%)": value,
                "Change (%)": value - avg
            })

        for time_val, value in zip(trough_times, trough_values):
            avg = calculate_average_for_plate(norm_f_custom_all, f'Plate_{plate + 1}', 'Time', time_val - time_lag, duration_sec)
            results.append({
                "Plate": plate + 1,
                "Type": "Trough",
                "Time (s)": time_val,
                "Value (%)": value,
                "Change (%)": value - avg
            })

    results_df = pd.DataFrame(results)
    return results_df, norm_f_custom_all

# ─────────────────────────────
# 各プロット・解析処理をタブで表示
# ─────────────────────────────
tabs = st.tabs([
    "6x4 Plot", 
    "6x4 Plot Y",
    "Force Trajectories", 
    "Force Trajectories w/ Center", 
    "Normalized Force", 
    "Plate Comparison", 
    "Analysis",
    "Plate Config Plot",
    "Movie"
])

# 各プレートに共通の測定項目・色設定
measurements = ['Fx', 'Fy', 'Fz', 'COPx', 'COPy', 'Tz']
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# ── タブ1: 6x4 プロット ─────────────────────────────
with tabs[0]:
    st.subheader("6x4 Plot of Force Plate Data")
    fig, axs = plt.subplots(6, 4, figsize=(16, 18), sharex=True)
    for plate in range(4):
        for i, measure in enumerate(measurements):
            column_index = plate * 6 + i + 1  # 時間列を除くため +1
            data = filtered_data.iloc[:, column_index]
            # Plate 1,3 の Fx, COPx を反転
            if measure in ['Fx', 'COPx'] and plate in [0, 2]:
                data = -data
            # 全プレートで Fy, COPy を反転
            if measure in ['Fy', 'COPy']:
                data = -data
            axs[i, plate].plot(filtered_data.iloc[:, 0], data, color=colors[i])
            axs[i, plate].set_title(f'{measure} (Ch {plate + 1})', fontsize=10)
            if plate == 0:
                axs[i, plate].set_ylabel(measure)
            if measure == 'COPx':
                axs[i, plate].set_ylim(-200, 200)
            elif measure == 'COPy':
                axs[i, plate].set_ylim(-300, 300)
            axs[i, plate].grid(True)
    for ax in axs[-1, :]:
        ax.set_xlabel('Time (s)')
    fig.suptitle(f'{file_name_without_ext} {title_time_range}', fontsize=16, weight='bold', y=0.98)
    fig.text(0.5, 0.96, '(Fx, COPX: Out+, In- ;  Fy, COPy: Front+, Back-)', ha='center', fontsize=10)
    plt.subplots_adjust(top=0.94, hspace=0.2)
    st.pyplot(fig)
    plt.close(fig)

# ── タブ2: Y軸範囲を統一した 6x4 プロット ─────────────────────────────
with tabs[1]:
    st.subheader("Unified Y-axis 6x4 Plot")
    channel_min_max = {measure: {'min': float('inf'), 'max': float('-inf')} for measure in measurements}
    for plate in range(4):
        for i, measure in enumerate(measurements):
            column_index = plate * 6 + i + 1
            data = filtered_data.iloc[:, column_index]
            if measure in ['Fx', 'COPx'] and plate in [0, 2]:
                data = -data
            if measure in ['Fy', 'COPy']:
                data = -data
            if measure == 'COPx':
                channel_min_max[measure]['min'] = -200
                channel_min_max[measure]['max'] = 200
            elif measure == 'COPy':
                channel_min_max[measure]['min'] = -300
                channel_min_max[measure]['max'] = 300
            else:
                channel_min_max[measure]['min'] = min(channel_min_max[measure]['min'], data.min())
                channel_min_max[measure]['max'] = max(channel_min_max[measure]['max'], data.max())
    fig, axs = plt.subplots(6, 4, figsize=(16, 18), sharex=True)
    for plate in range(4):
        for i, measure in enumerate(measurements):
            column_index = plate * 6 + i + 1
            data = filtered_data.iloc[:, column_index]
            if measure in ['Fx', 'COPx'] and plate in [0, 2]:
                data = -data
            if measure in ['Fy', 'COPy']:
                data = -data
            axs[i, plate].plot(filtered_data.iloc[:, 0], data, color=colors[i])
            axs[i, plate].set_title(f'{measure} (Ch {plate + 1})', fontsize=10)
            if measure in ['COPx', 'COPy']:
                axs[i, plate].set_ylim(channel_min_max[measure]['min'], channel_min_max[measure]['max'])
            else:
                y_margin = (channel_min_max[measure]['max'] - channel_min_max[measure]['min']) * 0.1
                axs[i, plate].set_ylim(channel_min_max[measure]['min'] - y_margin, channel_min_max[measure]['max'] + y_margin)
            if plate == 0:
                axs[i, plate].set_ylabel(measure)
            axs[i, plate].grid(True)
    for ax in axs[-1, :]:
        ax.set_xlabel('Time (s)')
    fig.suptitle(f'{file_name_without_ext} {title_time_range}', fontsize=16, weight='bold', y=0.98)
    fig.text(0.5, 0.96, '(Fx, COPX: Out+, In- ;  Fy, COPy: Front+, Back-)', ha='center', fontsize=10)
    plt.subplots_adjust(top=0.94, hspace=0.2)
    st.pyplot(fig)
    plt.close(fig)

# ── タブ3: 軌跡のプロット（正規化：MaxF 使用） ─────────────────────────────
with tabs[2]:
    st.subheader("Force Trajectories")
    # 100Hz にダウンサンプリング（例：10ms間隔）
    downsampled_data = filtered_data.iloc[::10, :].reset_index(drop=True)
    # 各プレートのデータ列オフセット（タプル内：(COPx, COPy, Fx, Fy, Fz)）
    plate_indices = [(4, 5, 1, 2, 3), (10, 11, 7, 8, 9), (16, 17, 13, 14, 15), (22, 23, 19, 20, 21)]
    base_colors = ['Reds', 'Greens', 'Blues', 'Purples']
    cmaps = [LinearSegmentedColormap.from_list(f"custom_{color}", [plt.get_cmap(color)(0.2), plt.get_cmap(color)(0.8)])
             for color in base_colors]
    cop_cmaps = [LinearSegmentedColormap.from_list(f"custom_cop_{color}", [plt.get_cmap(color)(0.8), plt.get_cmap(color)(1.0)])
                 for color in base_colors]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-400, 400)
    ax.set_ylim(-600, 600)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('COPx (mm)', fontsize=8)
    ax.set_ylabel('COPy (mm)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    # 軌跡データ格納用リスト
    all_trace_x = [[] for _ in range(len(plate_indices))]
    all_trace_y = [[] for _ in range(len(plate_indices))]
    all_trace_time = [[] for _ in range(len(plate_indices))]
    all_copx = [[] for _ in range(len(plate_indices))]
    all_copy = [[] for _ in range(len(plate_indices))]
    
    # 各プレートの最大力（スムージング後）を求める
    window = 25
    max_smoothed_norms = []
    for plate in range(4):
        fx = downsampled_data.iloc[:, plate * 6 + 1]
        fy = downsampled_data.iloc[:, plate * 6 + 2]
        fz = downsampled_data.iloc[:, plate * 6 + 3]
        norm_f = np.sqrt(fx**2 + fy**2 + fz**2)
        smoothed_norm_f = norm_f.rolling(window=window, center=True).mean()
        max_smoothed_norms.append(smoothed_norm_f.max())
    overall_max_norm = max(max_smoothed_norms)
    if MaxF is None or MaxF <= 0:
        normalization_value = overall_max_norm
    else:
        normalization_value = MaxF
    normalization_value = math.ceil(normalization_value)
    st.write(f'Normalization value: {normalization_value}')
    
    # 各時刻ごとに軌跡を取得
    for num, time_val in enumerate(downsampled_data.iloc[:, 0]):
        for idx, (copx_idx, copy_idx, fx_idx, fy_idx, fz_idx) in enumerate(plate_indices):
            copx = downsampled_data.iloc[num, copx_idx]
            copy = downsampled_data.iloc[num, copy_idx]
            fx = downsampled_data.iloc[num, fx_idx]
            fy = downsampled_data.iloc[num, fy_idx]
            fz = downsampled_data.iloc[num, fz_idx]
            
            # 必要に応じて X, Y の符号を反転
            if invert_x:
                copx = -copx
                fx = -fx

            if invert_y:
                copy = -copy
                fy = -fy
            
            # プレートごとの中心座標を取得し、位置調整
            # ここで plate_centers リストのインデックス idx を利用
            if copx == 0 and copy == 0:
                continue
            center_x, center_y = plate_centers[idx]
            copx += center_x
            copy += center_y




            force_magnitude = math.sqrt(fx**2 + fy**2 + fz**2) / (math.sqrt(fx**2 + fy**2) if (fx**2+fy**2)!=0 else 1)
            length = force_magnitude / normalization_value * 100
            end_x = copx + length * fx
            end_y = copy + length * fy
            all_trace_x[idx].extend([copx, end_x])
            all_trace_y[idx].extend([copy, end_y])
            all_trace_time[idx].extend([time_val, time_val])
            all_copx[idx].append(copx)
            all_copy[idx].append(copy)
    
    # 軌跡の描画
    for idx in range(len(plate_indices)):
        if len(all_trace_time[idx]) == 0:
            continue
        points = np.array([all_trace_x[idx], all_trace_y[idx]]).T.reshape(-1, 2, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        times_arr = np.array(all_trace_time[idx])
        norm_obj = plt.Normalize(times_arr.min(), times_arr.max())
        lc = LineCollection(segments, cmap=cmaps[idx], norm=norm_obj)
        lc.set_array(times_arr)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        cop_norm = plt.Normalize(times_arr.min(), times_arr.max())
        ax.scatter(all_copx[idx], all_copy[idx], c=times_arr[::2], cmap=cop_cmaps[idx], norm=cop_norm, s=15, zorder=3)
    ax.set_title(f'{file_name_without_ext} - Overall Force Trajectories', fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

# ── タブ4: 軌跡のプロット＋圧中心プロット ─────────────────────────────
with tabs[3]:
    st.subheader("Force Trajectories with Center of Force")
    downsampled_data = filtered_data.iloc[::10, :].reset_index(drop=True)
    plate_indices = [(4, 5, 1, 2, 3), (10, 11, 7, 8, 9), (16, 17, 13, 14, 15), (22, 23, 19, 20, 21)]
    base_colors = ['Reds', 'Greens', 'Blues', 'Purples']
    cmaps = [LinearSegmentedColormap.from_list(f"custom_{color}", [plt.get_cmap(color)(0.2), plt.get_cmap(color)(0.8)])
             for color in base_colors]
    cop_cmaps = [LinearSegmentedColormap.from_list(f"custom_cop_{color}", [plt.get_cmap(color)(0.8), plt.get_cmap(color)(1.0)])
                 for color in base_colors]
    
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(-400, 400)
    ax.set_ylim(-600, 600)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('COPx (mm)', fontsize=8)
    ax.set_ylabel('COPy (mm)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    all_trace_x = [[] for _ in range(len(plate_indices))]
    all_trace_y = [[] for _ in range(len(plate_indices))]
    all_trace_time = [[] for _ in range(len(plate_indices))]
    all_copx = [[] for _ in range(len(plate_indices))]
    all_copy = [[] for _ in range(len(plate_indices))]
    
    window = 25
    max_smoothed_norms = []
    for plate in range(4):
        fx = downsampled_data.iloc[:, plate * 6 + 1]
        fy = downsampled_data.iloc[:, plate * 6 + 2]
        fz = downsampled_data.iloc[:, plate * 6 + 3]
        norm_f = np.sqrt(fx**2 + fy**2 + fz**2)
        smoothed_norm_f = norm_f.rolling(window=window, center=True).mean()
        max_smoothed_norms.append(smoothed_norm_f.max())
    overall_max_norm = max(max_smoothed_norms)
    if MaxF is None or MaxF <= 0:
        normalization_value = overall_max_norm
    else:
        normalization_value = MaxF
    normalization_value = math.ceil(normalization_value)
    st.write(f'Normalization value: {normalization_value}')
    
    center_x_data = []
    center_y_data = []
    
    for num, time_val in enumerate(downsampled_data.iloc[:, 0]):
        total_copx = 0
        total_copy = 0
        total_fz = 0
        for idx, (copx_idx, copy_idx, fx_idx, fy_idx, fz_idx) in enumerate(plate_indices):
            copx = downsampled_data.iloc[num, copx_idx]
            copy = downsampled_data.iloc[num, copy_idx]
            fx = downsampled_data.iloc[num, fx_idx]
            fy = downsampled_data.iloc[num, fy_idx]
            fz = downsampled_data.iloc[num, fz_idx]
            copy = -copy
            fy = -fy
            if copx == 0 and copy == 0:
                continue
            if idx == 0:
                copx -= 200
                copy -= 300
            elif idx == 1:
                copx += 200
                copy -= 300
            elif idx == 2:
                copx -= 200
                copy += 300
            elif idx == 3:
                copx += 200
                copy += 300
            force_magnitude = math.sqrt(fx**2 + fy**2 + fz**2) / (math.sqrt(fx**2 + fy**2) if (fx**2+fy**2)!=0 else 1)
            length = force_magnitude / normalization_value * 100
            end_x = copx + length * fx
            end_y = copy + length * fy
            all_trace_x[idx].extend([copx, end_x])
            all_trace_y[idx].extend([copy, end_y])
            all_trace_time[idx].extend([time_val, time_val])
            all_copx[idx].append(copx)
            all_copy[idx].append(copy)
            total_copx += copx * fz
            total_copy += copy * fz
            total_fz += fz
        if total_fz > 0:
            center_copx = total_copx / total_fz
            center_copy = total_copy / total_fz
            center_x_data.append(center_copx)
            center_y_data.append(center_copy)
            
    for idx in range(len(plate_indices)):
        if len(all_trace_time[idx]) == 0:
            continue
        points = np.array([all_trace_x[idx], all_trace_y[idx]]).T.reshape(-1, 2, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        times_arr = np.array(all_trace_time[idx])
        norm_obj = plt.Normalize(times_arr.min(), times_arr.max())
        lc = LineCollection(segments, cmap=cmaps[idx], norm=norm_obj)
        lc.set_array(times_arr)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        cop_norm = plt.Normalize(times_arr.min(), times_arr.max())
        ax.scatter(all_copx[idx], all_copy[idx], c=times_arr[::2], cmap=cop_cmaps[idx], norm=cop_norm, s=15, zorder=3)
    ax.plot(center_x_data, center_y_data, color='gray', alpha=0.5, linewidth=2)
    ax.set_title(f'{file_name_without_ext} - Overall Force Trajectories with Center of Force', fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

# ── タブ5: 合力の正規化プロット ─────────────────────────────
with tabs[4]:
    st.subheader("Normalized Force Plot")
    window = 250
    max_smoothed_norms = []
    for plate in range(4):
        fx = filtered_data.iloc[:, plate * 6 + 1]
        fy = filtered_data.iloc[:, plate * 6 + 2]
        fz = filtered_data.iloc[:, plate * 6 + 3]
        norm_f = np.sqrt(fx**2 + fy**2 + fz**2)
        smoothed_norm_f = norm_f.rolling(window=window, center=True).mean()
        max_smoothed_norms.append(smoothed_norm_f.max())
    overall_max_norm = max(max_smoothed_norms)
    if MaxF is None or MaxF <= 0:
        normalization_value = overall_max_norm
    else:
        normalization_value = MaxF
    normalization_value = math.ceil(normalization_value)
    st.write(f"Normalization value: {normalization_value}")
    
    fig, axs = plt.subplots(4, 4, figsize=(16, 14))
    for plate in range(4):
        fx = filtered_data.iloc[:, plate * 6 + 1]
        fy = filtered_data.iloc[:, plate * 6 + 2]
        fz = filtered_data.iloc[:, plate * 6 + 3]
        norm_f = np.sqrt(fx**2 + fy**2 + fz**2)
        norm_f_percent = (norm_f / max_smoothed_norms[plate]) * 100
        norm_f_custom = (norm_f / normalization_value) * 100
        col_idx = plate
        axs[0, col_idx].plot(filtered_data.iloc[:, 0], norm_f, color='blue', label='|F| (N)')
        axs[0, col_idx].set_title(f'Plate {plate + 1} |F|', fontsize=10)
        axs[0, col_idx].grid(True)
        if plate == 0:
            axs[0, col_idx].set_ylabel('|F| (N)', fontsize=10)
        axs[0, col_idx].set_ylim(0, overall_max_norm * 1.1)
        
        axs[1, col_idx].plot(filtered_data.iloc[:, 0], norm_f_percent, color='green', label='% of Max |F|')
        axs[1, col_idx].set_title(f'Plate {plate + 1} % of Max |F|', fontsize=10)
        axs[1, col_idx].grid(True)
        if plate == 0:
            axs[1, col_idx].set_ylabel('% of Max |F|', fontsize=10)
        axs[1, col_idx].set_ylim(0, 110)
        
        axs[2, col_idx].plot(filtered_data.iloc[:, 0], norm_f_custom, color='purple', label=f'% of {normalization_value} N')
        axs[2, col_idx].set_title(f'Plate {plate + 1} % of {normalization_value} N', fontsize=10)
        axs[2, col_idx].grid(True)
        if plate == 0:
            axs[2, col_idx].set_ylabel(f'% of {normalization_value} N', fontsize=10)
        axs[2, col_idx].set_ylim(0, 110)
    axs[3, 0].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 1]**2 + filtered_data.iloc[:, 2]**2 + filtered_data.iloc[:, 3]**2)**0.5 / normalization_value * 100,
                   color='blue', label='Plate 1')
    axs[3, 0].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 7]**2 + filtered_data.iloc[:, 8]**2 + filtered_data.iloc[:, 9]**2)**0.5 / normalization_value * 100,
                   color='green', label='Plate 2')
    axs[3, 0].set_title('Plate 1 & 2', fontsize=10)
    axs[3, 0].set_ylabel(f'% of {normalization_value} N', fontsize=10)
    axs[3, 0].set_ylim(0, 110)
    axs[3, 0].grid(True)
    axs[3, 0].legend(fontsize=8)
    
    axs[3, 1].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 1]**2 + filtered_data.iloc[:, 2]**2 + filtered_data.iloc[:, 3]**2)**0.5 / normalization_value * 100,
                   color='blue', label='Plate 1')
    axs[3, 1].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 7]**2 + filtered_data.iloc[:, 8]**2 + filtered_data.iloc[:, 9]**2)**0.5 / normalization_value * 100,
                   color='green', label='Plate 2')
    axs[3, 1].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 19]**2 + filtered_data.iloc[:, 20]**2 + filtered_data.iloc[:, 21]**2)**0.5 / normalization_value * 100,
                   color='red', label='Plate 4')
    axs[3, 1].set_title('Plate 1, 2 & 4', fontsize=10)
    axs[3, 1].set_ylim(0, 110)
    axs[3, 1].grid(True)
    axs[3, 1].legend(fontsize=8)
    
    axs[3, 2].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 1]**2 + filtered_data.iloc[:, 2]**2 + filtered_data.iloc[:, 3]**2)**0.5 / normalization_value * 100,
                   color='blue', label='Plate 1')
    axs[3, 2].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 7]**2 + filtered_data.iloc[:, 8]**2 + filtered_data.iloc[:, 9]**2)**0.5 / normalization_value * 100,
                   color='green', label='Plate 2')
    axs[3, 2].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 13]**2 + filtered_data.iloc[:, 14]**2 + filtered_data.iloc[:, 15]**2)**0.5 / normalization_value * 100,
                   color='red', label='Plate 3')
    axs[3, 2].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 19]**2 + filtered_data.iloc[:, 20]**2 + filtered_data.iloc[:, 21]**2)**0.5 / normalization_value * 100,
                   color='orange', label='Plate 4')
    axs[3, 2].set_title('All Plates', fontsize=10)
    axs[3, 2].set_ylim(0, 110)
    axs[3, 2].grid(True)
    axs[3, 2].legend(fontsize=8)
    
    axs[3, 3].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 1]**2 + filtered_data.iloc[:, 2]**2 + filtered_data.iloc[:, 3]**2)**0.5 / normalization_value * 100,
                   color='blue')
    axs[3, 3].plot(filtered_data.iloc[:, 0],
                   (filtered_data.iloc[:, 7]**2 + filtered_data.iloc[:, 8]**2 + filtered_data.iloc[:, 9]**2)**0.5 / normalization_value * 100,
                   color='green')
    axs[3, 3].set_ylabel('%', fontsize=10)
    axs[3, 3].set_ylim(0, 110)
    axs[3, 3].grid(True)
    
    for ax in axs[3, :]:
        ax.set_xlabel('Time (s)', fontsize=10)
    for plate in range(4):
        fig.text(0.1 + 0.25 * plate, 0.02, f'Plate {plate + 1} Max |F|: {max_smoothed_norms[plate]:.2f} N', ha='center', fontsize=10)
    fig.suptitle(f'{file_name_without_ext} {title_time_range}', fontsize=16, weight='bold', y=0.98)
    st.pyplot(fig)
    plt.close(fig)

# ── タブ6: Plate 1 & Plate 2 の正規化プロット ─────────────────────────────
with tabs[5]:
    st.subheader("Plate 1 & 2 Normalized Force Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered_data.iloc[:, 0],
            (filtered_data.iloc[:, 1]**2 + filtered_data.iloc[:, 2]**2 + filtered_data.iloc[:, 3]**2)**0.5 / normalization_value * 100,
            color='blue', label='Plate 1', linewidth=2)
    ax.plot(filtered_data.iloc[:, 0],
            (filtered_data.iloc[:, 7]**2 + filtered_data.iloc[:, 8]**2 + filtered_data.iloc[:, 9]**2)**0.5 / normalization_value * 100,
            color='green', label='Plate 2', linewidth=2)
    ax.set_ylabel(f'% of {normalization_value} N', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylim(0, 130)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.set_title(f'{file_name_without_ext} {title_time_range}', fontsize=12, weight='bold')
    st.pyplot(fig)
    plt.close(fig)

# ── タブ7: 歩行開始時の重心移動の解析 ─────────────────────────────
with tabs[6]:
    st.subheader("Walking Start Analysis")
    if st.button("Run Walking Start Analysis"):
        results_df, norm_f_custom_all = analyze_norm_f_custom_with_averages(filtered_data, normalization_value, prominence=5, distance=100, duration_sec=0.5)
        st.write("Peak and Trough Analysis Results:")
        st.dataframe(results_df)
        # CSV ダウンロードボタン
        csv_results = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Analysis Results CSV", data=csv_results,
                           file_name=f"{file_name_without_ext}_analysis_results.csv", mime="text/csv")
        csv_norm = norm_f_custom_all.to_csv(index=False).encode('utf-8')
        st.download_button("Download Norm(F) Custom Data CSV", data=csv_norm,
                           file_name=f"{file_name_without_ext}_norm_f_custom.csv", mime="text/csv")


# ─────────────────────────────
# Plate Configuration の 2D プロット表示
# ─────────────────────────────
with tabs[7]:
    st.subheader("Plate Configuration Plot")
    
    # サイドバーで設定した各プレートの中心座標に反転設定を適用
    plot_centers = []
    for (cx, cy) in plate_centers:
        plot_cx = -cx if invert_x else cx
        plot_cy = -cy if invert_y else cy
        plot_centers.append((plot_cx, plot_cy))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Plate Centers and Directions")
    
    arrow_length = 50  # 矢印の長さ（必要に応じて調整）
    for idx, (px, py) in enumerate(plot_centers):
        ax.plot(px, py, 'o', markersize=8, label=f'Plate {idx+1}')
        ax.text(px, py, f' Plate {idx+1}', fontsize=10, verticalalignment='bottom')
        
        # 反転設定に合わせた矢印方向（X: 赤、Y: 緑）
        x_dir = -arrow_length if invert_x else arrow_length
        y_dir = -arrow_length if invert_y else arrow_length
        ax.arrow(px, py, x_dir, 0, head_width=5, head_length=5, fc='red', ec='red')
        ax.arrow(px, py, 0, y_dir, head_width=5, head_length=5, fc='green', ec='green')
    
    # 各プレートの座標の最大値を取得し、軸範囲をその2倍に設定
    xs = [pt[0] for pt in plot_centers]
    ys = [pt[1] for pt in plot_centers]
    max_x = max(abs(x) for x in xs)
    max_y = max(abs(y) for y in ys)
    ax.set_xlim(-2 * max_x, 2 * max_x)
    ax.set_ylim(-2 * max_y, 2 * max_y)
    
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize=8)
    
    st.pyplot(fig)




# ─────────────────────────────
# Movie 作成
# ─────────────────────────────

# まず、filtered_data からダウンサンプリングしたデータを作成し、グローバル変数として設定
ds_data = filtered_data.iloc[::20, :].reset_index(drop=True)
global_ds_data = ds_data  # スレッド間で共有するためのグローバル変数
num_frames = len(ds_data)
base_name = os.path.splitext(file_name)[0]

# グローバル変数で累積データを保持する
cumulative_trace_x = [[] for _ in range(4)]
cumulative_trace_y = [[] for _ in range(4)]

if os.name == "nt":  # Windows
    output_folder = "C:\\temp"
else:  # Linux/macOS
    output_folder = "/tmp"

output_path = os.path.join(output_folder, f"{base_name}_animation.mp4")
os.makedirs(output_folder, exist_ok=True)  # フォルダがなければ作成


def render_frame(frame_index):
    ds = global_ds_data  # グローバルなダウンサンプリング済みデータ
    fig, ax = plt.subplots(figsize=(4, 6.08))
    ax.set_xlim(-400, 400)
    ax.set_ylim(-600, 600)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('COPx (mm)', fontsize=8)
    ax.set_ylabel('COPy (mm)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    plate_indices_csv = [(4, 5, 1, 2, 3),
                           (10, 11, 7, 8, 9),
                           (16, 17, 13, 14, 15),
                           (22, 23, 19, 20, 21)]
    colors_csv = ['r', 'g', 'b', 'm']
    
    # 各プレートごとの処理
    for idx, (copx_idx, copy_idx, fx_idx, fy_idx, fz_idx) in enumerate(plate_indices_csv):
        try:
            copx = ds.iloc[frame_index, copx_idx]
            copy = ds.iloc[frame_index, copy_idx]
            fx = ds.iloc[frame_index, fx_idx]
            fy = ds.iloc[frame_index, fy_idx]
            fz = ds.iloc[frame_index, fz_idx]
        except Exception:
            continue
        
        if invert_x:
            copx = -copx
            fx = -fx
        if invert_y:
            copy = -copy
            fy = -fy
        if copx == 0 and copy == 0:
            continue
        
        center_x, center_y = plate_centers[idx]
        copx += center_x
        copy += center_y
        
        force_magnitude = math.sqrt(fx**2 + fy**2 + fz**2) / (math.sqrt(fx**2 + fy**2) if (fx**2+fy**2) != 0 else 1)
        length = force_magnitude / normalization_value * 100
        end_x = copx + length * fx
        end_y = copy + length * fy
        
        # 累積データに追加
        cumulative_trace_x[idx].extend([copx, end_x])
        cumulative_trace_y[idx].extend([copy, end_y])
        
        # 現在のフレームの点と矢印を描画
        ax.plot(copx, copy, 'o', color=colors_csv[idx], markersize=3)
        ax.plot([copx, end_x], [copy, end_y], color=colors_csv[idx], linewidth=1.5)
    
    # 各プレートごとに累積トレースを一度だけ描画（内側のループで変数名を上書きしないように別の変数名を使用）
    for j in range(len(plate_indices_csv)):
        if cumulative_trace_x[j] and cumulative_trace_y[j]:
            ax.plot(cumulative_trace_x[j], cumulative_trace_y[j], color=colors_csv[j], linewidth=1, alpha=0.5)
    
    current_time = ds.iloc[frame_index, 0]
    ax.set_title(f"{base_name}\nTime = {current_time:.2f} s", fontsize=8)
    
    # BytesIO に保存して画像を取得
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = imageio.imread(buf)
    plt.close(fig)
    return img



def render_all_frames(num_frames, max_workers=4):
    """
    ThreadPoolExecutor を用いて、各フレームの画像を並列処理で生成します。
    """
    frames = [None] * num_frames
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {executor.submit(render_frame, i): i for i in range(num_frames)}
        for future in concurrent.futures.as_completed(future_to_frame):
            i = future_to_frame[future]
            try:
                frames[i] = future.result()
            except Exception as exc:
                st.error(f"Frame {i} generated an exception: {exc}")
    return frames

def create_csv_animation(filtered_data, file_name, normalization_value, output_folder=output_folder, fps=25):
    """
    CSVデータからアニメーションを生成し、動画ファイルとして保存する関数。
    """
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}_animation.mp4")
    
    # 画像サイズ調整
    global_ds_data["frame_height"] = 608  # 16の倍数に調整
    global_ds_data["frame_width"] = 400

    start_time = time.time()
    st.info("Generating frames (this may take a while)...")
    frames = render_all_frames(num_frames, max_workers=4)
    elapsed = time.time() - start_time
    st.write(f"Frames generated in {elapsed:.2f} seconds.")
    
    st.info("Encoding video using imageio and ffmpeg...")
    try:
        imageio.mimwrite(output_path, frames, fps=fps, quality=8, macro_block_size=1)
        st.success("Video encoded successfully!")
    except Exception as e:
        st.error(f"Video encoding failed: {e}")
    
    total_time = time.time() - start_time
    st.write(f"Total movie creation time: {total_time:.2f} seconds.")
    return output_path

# ─────────────────────────────
# Movie タブ：CSVデータから生成したムービーの作成と表示
# ─────────────────────────────
with tabs[8]:
    st.subheader("Movie Creation and Display")
    if st.button("Create Animation"):
        with st.spinner("Creating animation from CSV data..."):
            # 正規化値の決定（従来の処理）
            window = 25
            max_smoothed_norms = []
            for plate in range(4):
                fx = filtered_data.iloc[:, plate * 6 + 1]
                fy = filtered_data.iloc[:, plate * 6 + 2]
                fz = filtered_data.iloc[:, plate * 6 + 3]
                norm_f = np.sqrt(fx**2 + fy**2 + fz**2)
                smoothed_norm_f = norm_f.rolling(window=window, center=True).mean()
                max_smoothed_norms.append(smoothed_norm_f.max())
            overall_max_norm = max(max_smoothed_norms)
            if MaxF is None or MaxF <= 0:
                norm_value = overall_max_norm
            else:
                norm_value = MaxF
            norm_value = math.ceil(norm_value)
            
            output_video_path = create_csv_animation(filtered_data, file_name, norm_value)
        st.success("Animation created successfully!")
        st.write("Video saved at:", output_video_path)
        
        try:
            file_size = os.path.getsize(output_video_path)
            st.write("File size (bytes):", file_size)
        except Exception as e:
            st.error(f"Could not determine file size: {e}")
        
        st.video(output_video_path)