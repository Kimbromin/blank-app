#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 온도 분석 웹 애플리케이션 (Streamlit)
Integrated Temperature Analysis System - Web Version

주요 기능:
- 히트맵 생성 및 분석 (create_heatmap 기능)
- Body Tip 온도 분석 (Body_Tip_gui 기능)
  - 트렌드 분석 (시간-온도 그래프, 평탄 구간 검출)
  - 온도 분포도 분석 (x축-온도 그래프)
  - 정규분포 피팅 및 합산 시뮬레이션
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import io

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="통합 온도 분석 시스템",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 통합 온도 분석 시스템")
st.markdown("---")

# ==================== 공통 유틸리티 함수 ====================

def detect_left_peak_and_fit_gaussian(x, y):
    """맨 왼쪽 봉우리 검출 및 정규 분포 피팅"""
    if len(x) < 3 or len(y) < 3:
        return None
    
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]
    
    left_half_idx = len(x_sorted) // 2
    x_left = x_sorted[:left_half_idx]
    y_left = y_sorted[:left_half_idx]
    
    if len(x_left) < 3:
        return None
    
    try:
        y_range = np.max(y_sorted) - np.min(y_sorted)
        prominence_threshold = y_range * 0.05
        
        peaks, properties = find_peaks(y_left, prominence=prominence_threshold)
        
        if len(peaks) == 0:
            return None
        
        peak_idx = peaks[0]
        peak_x = x_left[peak_idx]
        peak_y = y_left[peak_idx]
        
        x_range = np.max(x_sorted) - np.min(x_sorted)
        window_size = x_range * 0.15
        
        mask = (x_sorted >= peak_x - window_size) & (x_sorted <= peak_x + window_size)
        
        if np.sum(mask) < 3:
            return None
        
        x_window = x_sorted[mask]
        y_window = y_sorted[mask]
        
        try:
            baseline = np.min(y_window)
            amplitude = peak_y - baseline
            mu_init = peak_x
            sigma_init = window_size / 3
            
            def gaussian_func(x_data, mu, sigma, amp, base):
                return amp * np.exp(-0.5 * ((x_data - mu) / sigma) ** 2) + base
            
            p0 = [mu_init, sigma_init, amplitude, baseline]
            bounds = ([peak_x - window_size, sigma_init * 0.1, amplitude * 0.1, baseline * 0.9],
                      [peak_x + window_size, window_size, amplitude * 2, baseline * 1.5])
            
            popt, _ = curve_fit(gaussian_func, x_window, y_window, p0=p0, bounds=bounds, maxfev=5000)
            
            mu, sigma, amplitude, baseline = popt
            
            if sigma > 0 and sigma < window_size and abs(mu - peak_x) < window_size:
                return {
                    'mu': mu,
                    'sigma': sigma,
                    'amplitude': amplitude,
                    'baseline': baseline,
                    'peak_x': peak_x,
                    'peak_y': peak_y
                }
                
        except Exception:
            return None
            
    except Exception:
        return None
    
    return None

def detect_plateaus(time_data, temp_data, num_plateaus=10):
    """평탄 구간 검출"""
    if len(time_data) < 3 or len(temp_data) < 3:
        return []
    
    smoothed = gaussian_filter1d(temp_data, sigma=2)
    diff = np.diff(smoothed)
    threshold = np.std(diff) * 0.3
    
    plateaus = []
    plateau_start = None
    
    for i in range(len(diff)):
        if abs(diff[i]) < threshold:
            if plateau_start is None:
                plateau_start = i
        else:
            if plateau_start is not None:
                plateau_length = i - plateau_start
                if plateau_length > len(time_data) * 0.02:
                    plateau_end = i
                    plateau_time_start = time_data[plateau_start]
                    plateau_time_end = time_data[plateau_end]
                    plateau_temp = np.mean(temp_data[plateau_start:plateau_end])
                    plateaus.append({
                        'start_idx': plateau_start,
                        'end_idx': plateau_end,
                        'time_start': plateau_time_start,
                        'time_end': plateau_time_end,
                        'temperature': plateau_temp
                    })
                plateau_start = None
    
    plateaus.sort(key=lambda x: x['temperature'], reverse=True)
    return plateaus[:num_plateaus]

def prepare_heatmap_data(df, data_start_row=9, use_smoothing=False, sigma_value=1.0):
    """히트맵 데이터 준비"""
    data_without_first_col = df.iloc[data_start_row:, 1:]
    numeric_df = data_without_first_col.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return None, None, None
    
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    row_indices = np.arange(len(numeric_df))
    y_mm_values = pd.Series(row_indices / 3.8)
    
    col_indices = np.arange(len(numeric_df.columns))
    x_mm_values = pd.Series(col_indices / 3.8)
    
    max_rows = 500
    if len(numeric_df) > max_rows:
        step = len(numeric_df) // max_rows
        numeric_df = numeric_df.iloc[::step].head(max_rows)
        y_mm_values = y_mm_values.iloc[::step].head(max_rows).reset_index(drop=True)
    
    if len(numeric_df.columns) > max_rows:
        step_col = len(numeric_df.columns) // max_rows
        numeric_df = numeric_df.iloc[:, ::step_col].iloc[:, :max_rows]
        x_mm_values = x_mm_values.iloc[::step_col].head(max_rows).reset_index(drop=True)
    
    if use_smoothing:
        numeric_df = pd.DataFrame(
            gaussian_filter(numeric_df.values, sigma=sigma_value),
            index=numeric_df.index,
            columns=numeric_df.columns
        )
    
    return numeric_df, x_mm_values, y_mm_values

# ==================== 세션 상태 초기화 ====================

if 'excel_data' not in st.session_state:
    st.session_state.excel_data = {}
if 'current_numeric_df' not in st.session_state:
    st.session_state.current_numeric_df = None
if 'current_x_mm_values' not in st.session_state:
    st.session_state.current_x_mm_values = None
if 'current_y_mm_values' not in st.session_state:
    st.session_state.current_y_mm_values = None
if 'distribution_data_list' not in st.session_state:
    st.session_state.distribution_data_list = []
if 'distribution_id_counter' not in st.session_state:
    st.session_state.distribution_id_counter = 0
if 'saved_sum_results' not in st.session_state:
    st.session_state.saved_sum_results = {}
if 'sum_result_id_counter' not in st.session_state:
    st.session_state.sum_result_id_counter = 0
if 'trend_data' not in st.session_state:
    st.session_state.trend_data = None
if 'distribution_files' not in st.session_state:
    st.session_state.distribution_files = {}
if 'normal_dist_params_b' not in st.session_state:
    st.session_state.normal_dist_params_b = []
if 'normal_dist_params_c' not in st.session_state:
    st.session_state.normal_dist_params_c = []
if 'left_peak_params_b' not in st.session_state:
    st.session_state.left_peak_params_b = None
if 'left_peak_params_c' not in st.session_state:
    st.session_state.left_peak_params_c = None

# ==================== 메인 탭 ====================

tab1, tab2, tab3, tab4 = st.tabs(["🔥 히트맵 분석", "📈 트렌드 분석", "📊 분포도 분석", "⚙️ 정규분포 시뮬레이션"])

# ==================== 탭 1: 히트맵 분석 ====================

with tab1:
    st.header("🔥 히트맵 생성 및 분석")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 파일 업로드")
        uploaded_files = st.file_uploader(
            "Excel 파일 선택",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            key="heatmap_files"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.excel_data:
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name='영역 데이터1', header=None)
                        st.session_state.excel_data[uploaded_file.name] = df
                        st.success(f"✅ {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")
        
        st.markdown("---")
        st.subheader("⚙️ 설정")
        
        if st.session_state.excel_data:
            analysis_mode = st.radio(
                "분석 모드",
                ["일반 히트맵", "델타 히트맵"],
                index=0
            )
            
            file_list = list(st.session_state.excel_data.keys())
            
            if analysis_mode == "델타 히트맵":
                sheet1_name = st.selectbox("첫 번째 파일", options=file_list, index=0)
                sheet2_name = st.selectbox("두 번째 파일", options=file_list, index=min(1, len(file_list)-1))
            else:
                selected_file = st.selectbox("분석할 파일", options=file_list)
                sheet1_name = None
                sheet2_name = None
        else:
            selected_file = None
            sheet1_name = None
            sheet2_name = None
        
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 
                     'RdYlBu', 'RdYlGn', 'Spectral', 'hot', 'cool']
        selected_cmap = st.selectbox("컬러맵", options=colormaps, index=0)
        
        use_smoothing = st.checkbox("스무딩 적용", value=False)
        if use_smoothing:
            sigma_value = st.slider("Sigma 값", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        else:
            sigma_value = 1.0
        
        st.markdown("---")
        st.subheader("컬러바 범위")
        auto_range = st.checkbox("자동 범위", value=True)
        
        if not auto_range:
            cbar_min = st.number_input("최소값", value=20.0, step=0.1)
            cbar_max = st.number_input("최대값", value=40.0, step=0.1)
        else:
            cbar_min = None
            cbar_max = None
        
        st.markdown("---")
        st.subheader("📊 분포도 설정")
        y_coord = st.number_input("Y 좌표 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, key="heatmap_y")
        show_y_dist = st.button("Y축 분포도", key="heatmap_y_btn")
        x_coord = st.number_input("X 좌표 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, key="heatmap_x")
        show_x_dist = st.button("X축 분포도", key="heatmap_x_btn")
    
    with col2:
        if analysis_mode == "델타 히트맵":
            if sheet1_name and sheet2_name and sheet1_name != sheet2_name:
                df1 = st.session_state.excel_data[sheet1_name]
                df2 = st.session_state.excel_data[sheet2_name]
                
                numeric_df1, x_mm1, y_mm1 = prepare_heatmap_data(df1, use_smoothing=use_smoothing, sigma_value=sigma_value)
                numeric_df2, x_mm2, y_mm2 = prepare_heatmap_data(df2, use_smoothing=use_smoothing, sigma_value=sigma_value)
                
                if numeric_df1 is not None and numeric_df2 is not None:
                    min_rows = min(len(numeric_df1), len(numeric_df2))
                    min_cols = min(len(numeric_df1.columns), len(numeric_df2.columns))
                    
                    numeric_df1 = numeric_df1.iloc[:min_rows, :min_cols].reset_index(drop=True)
                    numeric_df1.columns = range(len(numeric_df1.columns))
                    numeric_df2 = numeric_df2.iloc[:min_rows, :min_cols].reset_index(drop=True)
                    numeric_df2.columns = range(len(numeric_df2.columns))
                    
                    delta_df = numeric_df2 - numeric_df1
                    delta_df = delta_df.fillna(delta_df.mean()).fillna(0)
                    
                    row_indices = np.arange(len(delta_df))
                    y_mm_values = pd.Series(row_indices / 3.8)
                    col_indices = np.arange(len(delta_df.columns))
                    x_mm_values = pd.Series(col_indices / 3.8)
                    
                    st.session_state.current_numeric_df = delta_df
                    st.session_state.current_x_mm_values = x_mm_values
                    st.session_state.current_y_mm_values = y_mm_values
                    
                    if auto_range:
                        vmin = float(delta_df.min().min())
                        vmax = float(delta_df.max().max())
                    else:
                        vmin = cbar_min
                        vmax = cbar_max
                    
                    fig, ax = plt.subplots(figsize=(16, 10))
                    
                    if len(delta_df.columns) <= 30:
                        x_labels = [f'{x_mm_values.iloc[i]:.2f}' for i in range(len(delta_df.columns))]
                    else:
                        step_label_x = max(1, len(delta_df.columns) // 15)
                        x_labels = [f'{x_mm_values.iloc[i]:.2f}' if i % step_label_x == 0 else '' 
                                   for i in range(len(delta_df.columns))]
                    
                    if len(delta_df) <= 30:
                        y_labels = [f'{y_mm_values.iloc[i]:.2f}' for i in range(len(delta_df))]
                    else:
                        step_label_y = max(1, len(delta_df) // 15)
                        y_labels = [f'{y_mm_values.iloc[i]:.2f}' if i % step_label_y == 0 else '' 
                                   for i in range(len(delta_df))]
                    
                    heatmap = sns.heatmap(
                        delta_df,
                        cmap=selected_cmap,
                        square=True,
                        linewidths=0,
                        cbar_kws={'label': 'ΔT [°C]'},
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=x_labels if len(delta_df.columns) <= 30 else False,
                        yticklabels=y_labels if len(delta_df) <= 30 else False,
                        ax=ax
                    )
                    
                    ax.set_title(f'Delta Heat map: {sheet2_name} - {sheet1_name}', fontsize=14, pad=10)
                    ax.set_xlabel('X axis [mm]', fontsize=12)
                    ax.set_ylabel('Y axis [mm]', fontsize=12)
                    
                    cbar = heatmap.collections[0].colorbar
                    cbar.ax.set_ylabel('ΔT [°C]', rotation=270, labelpad=20)
                    cbar.ax.yaxis.label.set_rotation(270)
                    cbar.ax.yaxis.label.set_x(1.4)
                    cbar.ax.yaxis.label.set_va('center')
                    cbar.ax.yaxis.label.set_ha('left')
                    cbar.ax.tick_params(pad=15)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format='png', dpi=150, bbox_inches='tight')
                        buf_png.seek(0)
                        st.download_button("📥 PNG 다운로드", data=buf_png, 
                                         file_name=f"delta_heatmap_{sheet2_name}_{sheet1_name}.png", 
                                         mime="image/png")
                    with col_dl2:
                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                        buf_svg.seek(0)
                        st.download_button("📥 SVG 다운로드", data=buf_svg, 
                                         file_name=f"delta_heatmap_{sheet2_name}_{sheet1_name}.svg", 
                                         mime="image/svg+xml")
        
        elif selected_file and selected_file in st.session_state.excel_data:
            df = st.session_state.excel_data[selected_file]
            numeric_df, x_mm_values, y_mm_values = prepare_heatmap_data(
                df, use_smoothing=use_smoothing, sigma_value=sigma_value
            )
            
            if numeric_df is not None:
                st.session_state.current_numeric_df = numeric_df
                st.session_state.current_x_mm_values = x_mm_values
                st.session_state.current_y_mm_values = y_mm_values
                
                if auto_range:
                    vmin = float(numeric_df.min().min())
                    vmax = float(numeric_df.max().max())
                else:
                    vmin = cbar_min
                    vmax = cbar_max
                
                fig, ax = plt.subplots(figsize=(16, 10))
                
                if len(numeric_df.columns) <= 30:
                    x_labels = [f'{x_mm_values.iloc[i]:.2f}' for i in range(len(numeric_df.columns))]
                else:
                    step_label_x = max(1, len(numeric_df.columns) // 15)
                    x_labels = [f'{x_mm_values.iloc[i]:.2f}' if i % step_label_x == 0 else '' 
                               for i in range(len(numeric_df.columns))]
                
                if len(numeric_df) <= 30:
                    y_labels = [f'{y_mm_values.iloc[i]:.2f}' for i in range(len(numeric_df))]
                else:
                    step_label_y = max(1, len(numeric_df) // 15)
                    y_labels = [f'{y_mm_values.iloc[i]:.2f}' if i % step_label_y == 0 else '' 
                               for i in range(len(numeric_df))]
                
                heatmap = sns.heatmap(
                    numeric_df,
                    cmap=selected_cmap,
                    square=True,
                    linewidths=0,
                    cbar_kws={'label': 'Temperature [°C]'},
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=x_labels if len(numeric_df.columns) <= 30 else False,
                    yticklabels=y_labels if len(numeric_df) <= 30 else False,
                    ax=ax
                )
                
                ax.set_title(f'{selected_file} Heat map', fontsize=14, pad=10)
                ax.set_xlabel('X axis [mm]', fontsize=12)
                ax.set_ylabel('Y axis [mm]', fontsize=12)
                
                cbar = heatmap.collections[0].colorbar
                cbar.ax.set_ylabel('Temperature [°C]', rotation=270, labelpad=20)
                cbar.ax.yaxis.label.set_rotation(270)
                cbar.ax.yaxis.label.set_x(1.4)
                cbar.ax.yaxis.label.set_va('center')
                cbar.ax.yaxis.label.set_ha('left')
                cbar.ax.tick_params(pad=15)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format='png', dpi=150, bbox_inches='tight')
                    buf_png.seek(0)
                    st.download_button("📥 PNG 다운로드", data=buf_png, 
                                     file_name=f"{selected_file}_heatmap.png", 
                                     mime="image/png")
                with col_dl2:
                    buf_svg = io.BytesIO()
                    fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                    buf_svg.seek(0)
                    st.download_button("📥 SVG 다운로드", data=buf_svg, 
                                     file_name=f"{selected_file}_heatmap.svg", 
                                     mime="image/svg+xml")
                
                # 분포도 그래프
                if show_y_dist or show_x_dist:
                    st.markdown("---")
                    st.subheader("📈 분포도 그래프")
                    
                    if show_y_dist:
                        y_mm_array = y_mm_values.values
                        row_idx = np.argmin(np.abs(y_mm_array - y_coord))
                        
                        if 0 <= row_idx < len(numeric_df):
                            row_data = numeric_df.iloc[row_idx, :].values
                            x_coords = x_mm_values.values
                            
                            fig_dist, ax_dist = plt.subplots(figsize=(12, 5))
                            ax_dist.plot(x_coords, row_data, 'b-', linewidth=2, marker='o', markersize=3, label='Data')
                            
                            fitted_params = detect_left_peak_and_fit_gaussian(x_coords, row_data)
                            
                            ax_dist.set_xlabel('X axis [mm]', fontsize=12)
                            ax_dist.set_ylabel('Temperature [°C]', fontsize=12)
                            ax_dist.set_title(f'Y-axis Distribution (Row {row_idx}, Y={y_mm_array[row_idx]:.2f}mm)', 
                                             fontsize=14, pad=15)
                            ax_dist.grid(True, alpha=0.3)
                            ax_dist.legend()
                            plt.tight_layout()
                            st.pyplot(fig_dist)
                    
                    if show_x_dist:
                        x_mm_array = x_mm_values.values
                        col_idx = np.argmin(np.abs(x_mm_array - x_coord))
                        
                        if 0 <= col_idx < len(numeric_df.columns):
                            col_data = numeric_df.iloc[:, col_idx].values
                            y_coords = y_mm_values.values
                            
                            fig_dist, ax_dist = plt.subplots(figsize=(12, 5))
                            ax_dist.plot(y_coords, col_data, 'r-', linewidth=2, marker='o', markersize=3, label='Data')
                            
                            ax_dist.set_xlabel('Y axis [mm]', fontsize=12)
                            ax_dist.set_ylabel('Temperature [°C]', fontsize=12)
                            ax_dist.set_title(f'X-axis Distribution (Column {col_idx}, X={x_mm_array[col_idx]:.2f}mm)', 
                                             fontsize=14, pad=15)
                            ax_dist.grid(True, alpha=0.3)
                            ax_dist.legend()
                            plt.tight_layout()
                            st.pyplot(fig_dist)
            else:
                st.warning("⚠️ 숫자 데이터를 찾을 수 없습니다.")
        else:
            st.info("👈 사이드바에서 Excel 파일을 업로드하세요.")

# ==================== 탭 2: 트렌드 분석 ====================

with tab2:
    st.header("📈 트렌드 분석")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 파일 업로드")
        uploaded_file = st.file_uploader("Excel 파일 선택", type=['xlsx', 'xls'], key="trend_file")
        
        st.markdown("---")
        st.subheader("⚙️ 설정")
        num_plateaus = st.number_input("평탄 구간 개수", min_value=1, max_value=20, value=10)
        use_smoothing = st.checkbox("스무딩 적용", value=True)
        if use_smoothing:
            smoothing_sigma = st.slider("스무딩 강도", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
        else:
            smoothing_sigma = 2.0
    
    with col2:
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                
                time_col = None
                temp_cols = []
                
                for col in df.columns:
                    col_str = str(col).lower()
                    if 'time' in col_str or '시간' in col_str or 't' == col_str:
                        time_col = col
                    elif 'temp' in col_str or '온도' in col_str or 'temperature' in col_str:
                        temp_cols.append(col)
                
                if time_col is None:
                    time_col = df.columns[0]
                if not temp_cols:
                    temp_cols = [df.columns[1]] if len(df.columns) > 1 else [df.columns[0]]
                
                time_data = pd.to_numeric(df[time_col], errors='coerce')
                temp_data = pd.to_numeric(df[temp_cols[0]], errors='coerce')
                
                valid_mask = ~(time_data.isna() | temp_data.isna())
                time_data = time_data[valid_mask].values
                temp_data = temp_data[valid_mask].values
                
                if len(time_data) > 0:
                    st.session_state.trend_data = {'time': time_data, 'temp': temp_data}
                    
                    if use_smoothing:
                        temp_data_smooth = gaussian_filter1d(temp_data, sigma=smoothing_sigma)
                    else:
                        temp_data_smooth = temp_data
                    
                    plateaus = detect_plateaus(time_data, temp_data_smooth, num_plateaus)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(time_data, temp_data, 'b-', linewidth=1, alpha=0.5, label='원본 데이터')
                    ax.plot(time_data, temp_data_smooth, 'r-', linewidth=2, label='스무딩 데이터')
                    
                    colors = plt.cm.tab10(np.linspace(0, 1, len(plateaus)))
                    for i, plateau in enumerate(plateaus):
                        ax.axhspan(plateau['temperature'] - 0.5, plateau['temperature'] + 0.5,
                                  xmin=(plateau['time_start'] - time_data.min()) / (time_data.max() - time_data.min()),
                                  xmax=(plateau['time_end'] - time_data.min()) / (time_data.max() - time_data.min()),
                                  alpha=0.3, color=colors[i], label=f"Plateau {i+1}")
                    
                    ax.set_xlabel('Time', fontsize=12)
                    ax.set_ylabel('Temperature [°C]', fontsize=12)
                    ax.set_title('Temperature Trend Analysis', fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    if plateaus:
                        st.subheader("평탄 구간 정보")
                        plateau_df = pd.DataFrame([
                            {
                                'Plateau': i+1,
                                'Time Start': f"{plateau['time_start']:.2f}",
                                'Time End': f"{plateau['time_end']:.2f}",
                                'Temperature [°C]': f"{plateau['temperature']:.2f}"
                            }
                            for i, plateau in enumerate(plateaus)
                        ])
                        st.dataframe(plateau_df, use_container_width=True)
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("📥 그래프 다운로드", data=buf, 
                                     file_name="trend_analysis.png", mime="image/png")
                else:
                    st.error("유효한 데이터를 찾을 수 없습니다.")
            except Exception as e:
                st.error(f"파일 읽기 오류: {str(e)}")
        else:
            st.info("Excel 파일을 업로드하세요.")

# ==================== 탭 3: 분포도 분석 ====================

with tab3:
    st.header("📊 온도 분포도 분석")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 파일 업로드")
        uploaded_files = st.file_uploader("Excel 파일 선택 (여러 파일 가능)", 
                                         type=['xlsx', 'xls'], 
                                         accept_multiple_files=True,
                                         key="dist_files")
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.distribution_files:
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name='직선분석', header=None)
                        data_start_row = 45
                        
                        if len(df.columns) >= 3:
                            x_coords = pd.to_numeric(df.iloc[data_start_row:, 0], errors='coerce')
                            b_temp = pd.to_numeric(df.iloc[data_start_row:, 1], errors='coerce')
                            c_temp = pd.to_numeric(df.iloc[data_start_row:, 2], errors='coerce')
                            
                            valid_mask = ~(x_coords.isna() | b_temp.isna() | c_temp.isna())
                            distribution_df = pd.DataFrame({
                                'x좌표': x_coords[valid_mask].values,
                                'B영역_온도': b_temp[valid_mask].values,
                                'C영역_온도': c_temp[valid_mask].values
                            })
                            
                            st.session_state.distribution_files[uploaded_file.name] = distribution_df
                            st.success(f"✅ {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")
        
        st.markdown("---")
        st.subheader("⚙️ 설정")
        
        if st.session_state.distribution_files:
            selected_files = st.multiselect(
                "분석할 파일 선택",
                options=list(st.session_state.distribution_files.keys()),
                default=list(st.session_state.distribution_files.keys())[:1] if st.session_state.distribution_files else []
            )
            
            show_b_region = st.checkbox("B영역 표시", value=True)
            show_c_region = st.checkbox("C영역 표시", value=True)
            use_delta_t = st.checkbox("ΔT 모드", value=False)
            
            if use_delta_t:
                reference_file = st.selectbox("기준 파일", 
                                             options=list(st.session_state.distribution_files.keys()),
                                             index=0)
            
            coord_conversion = st.checkbox("좌표 변환 (px → mm)", value=False)
            if coord_conversion:
                mm_value = st.number_input("mm 값", value=10.0, step=0.1)
                px_value = st.number_input("px 값", value=38.0, step=0.1)
                conversion_ratio = mm_value / px_value if px_value > 0 else 1.0
            else:
                conversion_ratio = 1.0
        else:
            selected_files = []
            show_b_region = True
            show_c_region = True
            use_delta_t = False
            reference_file = None
            conversion_ratio = 1.0
    
    with col2:
        if selected_files:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_files)))
            
            for idx, file_name in enumerate(selected_files):
                df = st.session_state.distribution_files[file_name]
                
                if coord_conversion:
                    df = df.copy()
                    df['x좌표'] = df['x좌표'] * conversion_ratio
                
                if use_delta_t and reference_file in st.session_state.distribution_files:
                    ref_df = st.session_state.distribution_files[reference_file]
                    if coord_conversion:
                        ref_df = ref_df.copy()
                        ref_df['x좌표'] = ref_df['x좌표'] * conversion_ratio
                    
                    if show_b_region:
                        merged_b = pd.merge(df[['x좌표', 'B영역_온도']], 
                                           ref_df[['x좌표', 'B영역_온도']],
                                           on='x좌표', how='inner', suffixes=('', '_ref'))
                        if len(merged_b) > 0:
                            delta_b = merged_b['B영역_온도'] - merged_b['B영역_온도_ref']
                            ax.plot(merged_b['x좌표'], delta_b, '-', linewidth=2, 
                                   color=colors[idx], label=f"{file_name} - B영역 ΔT")
                    
                    if show_c_region:
                        merged_c = pd.merge(df[['x좌표', 'C영역_온도']], 
                                           ref_df[['x좌표', 'C영역_온도']],
                                           on='x좌표', how='inner', suffixes=('', '_ref'))
                        if len(merged_c) > 0:
                            delta_c = merged_c['C영역_온도'] - merged_c['C영역_온도_ref']
                            ax.plot(merged_c['x좌표'], delta_c, '--', linewidth=2, 
                                   color=colors[idx], label=f"{file_name} - C영역 ΔT")
                else:
                    if show_b_region:
                        ax.plot(df['x좌표'], df['B영역_온도'], '-', linewidth=2, 
                               color=colors[idx], label=f"{file_name} - B영역")
                    if show_c_region:
                        ax.plot(df['x좌표'], df['C영역_온도'], '--', linewidth=2, 
                               color=colors[idx], label=f"{file_name} - C영역")
            
            ax.set_xlabel('X 좌표 [mm]' if coord_conversion else 'X 좌표 [px]', fontsize=12)
            ax.set_ylabel('Temperature [°C]', fontsize=12)
            ax.set_title('Temperature Distribution Analysis', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button("📥 그래프 다운로드", data=buf, 
                             file_name="distribution_analysis.png", mime="image/png")
        else:
            st.info("Excel 파일을 업로드하세요.")

# ==================== 탭 4: 정규분포 시뮬레이션 ====================

with tab4:
    st.header("⚙️ 정규분포 시뮬레이션")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 파일 업로드")
        uploaded_files = st.file_uploader("Excel 파일 선택 (여러 파일 가능)", 
                                         type=['xlsx', 'xls'], 
                                         accept_multiple_files=True,
                                         key="sim_files")
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.distribution_files:
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name='직선분석', header=None)
                        data_start_row = 45
                        
                        if len(df.columns) >= 3:
                            x_coords = pd.to_numeric(df.iloc[data_start_row:, 0], errors='coerce')
                            b_temp = pd.to_numeric(df.iloc[data_start_row:, 1], errors='coerce')
                            c_temp = pd.to_numeric(df.iloc[data_start_row:, 2], errors='coerce')
                            
                            valid_mask = ~(x_coords.isna() | b_temp.isna() | c_temp.isna())
                            distribution_df = pd.DataFrame({
                                'x좌표': x_coords[valid_mask].values,
                                'B영역_온도': b_temp[valid_mask].values,
                                'C영역_온도': c_temp[valid_mask].values
                            })
                            
                            st.session_state.distribution_files[uploaded_file.name] = distribution_df
                            st.success(f"✅ {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")
        
        st.markdown("---")
        st.subheader("⚙️ 설정")
        
        if st.session_state.distribution_files:
            selected_file = st.selectbox("분석할 파일", 
                                         options=list(st.session_state.distribution_files.keys()))
            
            active_region = st.radio("분석 영역", ["B영역", "C영역"], index=0)
            show_left_peak = st.checkbox("왼쪽 봉우리 표시", value=True)
            show_sum = st.checkbox("합산 결과 표시", value=False)
            
            num_distributions = st.number_input("정규분포 개수", min_value=0, max_value=10, value=0)
        else:
            selected_file = None
            active_region = "B영역"
            show_left_peak = True
            show_sum = False
            num_distributions = 0
    
    with col2:
        if selected_file and selected_file in st.session_state.distribution_files:
            df = st.session_state.distribution_files[selected_file]
            
            if active_region == "B영역":
                x_data = df['x좌표'].values
                y_data = df['B영역_온도'].values
                normal_dist_params = st.session_state.normal_dist_params_b
                left_peak_params = st.session_state.left_peak_params_b
            else:
                x_data = df['x좌표'].values
                y_data = df['C영역_온도'].values
                normal_dist_params = st.session_state.normal_dist_params_c
                left_peak_params = st.session_state.left_peak_params_c
            
            if left_peak_params is None:
                left_peak_params = detect_left_peak_and_fit_gaussian(x_data, y_data)
                if active_region == "B영역":
                    st.session_state.left_peak_params_b = left_peak_params
                else:
                    st.session_state.left_peak_params_c = left_peak_params
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_data, y_data, 'b-', linewidth=2, label='Data')
            
            if show_left_peak and left_peak_params:
                x_extended = np.linspace(x_data.min(), x_data.max(), 1000)
                y_gaussian = left_peak_params['amplitude'] * np.exp(-0.5 * ((x_extended - left_peak_params['mu']) / left_peak_params['sigma']) ** 2) + left_peak_params['baseline']
                ax.plot(x_extended, y_gaussian, '--', linewidth=2, alpha=0.7, color='red', label='Left Peak')
            
            if len(normal_dist_params) > 0:
                x_extended = np.linspace(x_data.min(), x_data.max(), 1000)
                for i, params in enumerate(normal_dist_params):
                    y_gaussian = params['amplitude'] * np.exp(-0.5 * ((x_extended - params['mu']) / params['sigma']) ** 2) + params['baseline']
                    ax.plot(x_extended, y_gaussian, '--', linewidth=2, alpha=0.7, label=f'Normal Dist {i+1}')
            
            if show_sum:
                x_extended = np.linspace(x_data.min(), x_data.max(), 1000)
                y_sum = np.zeros_like(x_extended)
                baseline_sum = 0
                count = 0
                
                if left_peak_params:
                    baseline_sum += left_peak_params['baseline']
                    count += 1
                    y_sum += left_peak_params['amplitude'] * np.exp(-0.5 * ((x_extended - left_peak_params['mu']) / left_peak_params['sigma']) ** 2)
                
                for params in normal_dist_params:
                    baseline_sum += params['baseline']
                    count += 1
                    y_sum += params['amplitude'] * np.exp(-0.5 * ((x_extended - params['mu']) / params['sigma']) ** 2)
                
                if count > 0:
                    avg_baseline = baseline_sum / count
                    y_sum_total = y_sum + avg_baseline
                    ax.plot(x_extended, y_sum_total, '-', linewidth=3, alpha=0.8, color='purple', label='Sum Result')
            
            ax.set_xlabel('X 좌표 [px]', fontsize=12)
            ax.set_ylabel('Temperature [°C]', fontsize=12)
            ax.set_title(f'Normal Distribution Simulation - {active_region}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            if num_distributions > 0:
                st.subheader("정규분포 파라미터 설정")
                new_params = []
                for i in range(num_distributions):
                    with st.expander(f"정규분포 {i+1}"):
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            mu = st.number_input(f"중심 위치 μ", value=left_peak_params['mu'] if left_peak_params and i == 0 else 0.0, key=f"mu_{i}")
                            sigma = st.number_input(f"표준편차 σ", value=left_peak_params['sigma'] if left_peak_params and i == 0 else 2.0, min_value=0.1, key=f"sigma_{i}")
                        with col_p2:
                            amplitude = st.number_input(f"진폭", value=left_peak_params['amplitude'] if left_peak_params and i == 0 else 5.0, key=f"amp_{i}")
                            baseline = st.number_input(f"기준선", value=left_peak_params['baseline'] if left_peak_params and i == 0 else 20.0, key=f"base_{i}")
                        new_params.append({'mu': mu, 'sigma': sigma, 'amplitude': amplitude, 'baseline': baseline})
                
                if st.button("파라미터 적용"):
                    if active_region == "B영역":
                        st.session_state.normal_dist_params_b = new_params
                    else:
                        st.session_state.normal_dist_params_c = new_params
                    st.rerun()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button("📥 그래프 다운로드", data=buf, 
                             file_name="normal_dist_simulation.png", mime="image/png")
        else:
            st.info("Excel 파일을 업로드하세요.")

# 사용 방법 안내
with st.expander("📖 사용 방법"):
    st.markdown("""
    ### 통합 온도 분석 시스템 사용 방법
    
    #### 1. 히트맵 분석
    - Excel 파일의 '영역 데이터1' 시트에서 히트맵 생성
    - 일반 히트맵 및 델타 히트맵 지원
    - 다양한 컬러맵 선택 및 스무딩 적용 가능
    - Y축/X축 분포도 생성 가능
    
    #### 2. 트렌드 분석
    - 시간-온도 데이터 분석
    - 평탄 구간(plateau) 자동 검출
    - 스무딩을 통한 노이즈 제거
    
    #### 3. 분포도 분석
    - Excel 파일의 '직선분석' 시트에서 데이터 읽기
    - B영역과 C영역 온도 분포도 비교
    - ΔT 모드로 기준선 대비 차이 분석
    - 좌표 변환 (px → mm)
    
    #### 4. 정규분포 시뮬레이션
    - 왼쪽 봉우리 자동 검출 및 피팅
    - 여러 정규분포 수동 추가
    - 합산 결과 확인
    
    ### 주요 기능
    - ✅ 히트맵 생성 및 분석
    - ✅ 트렌드 분석 및 평탄 구간 검출
    - ✅ 온도 분포도 분석 (B영역/C영역)
    - ✅ ΔT 모드 (기준선 대비 차이)
    - ✅ 정규분포 피팅 및 합산 시뮬레이션
    - ✅ 좌표 변환 (px → mm)
    - ✅ 그래프 다운로드 (PNG, SVG)
    """)

