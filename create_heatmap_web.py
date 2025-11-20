#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
히트맵 생성 웹 애플리케이션 (Streamlit)
엑셀 파일을 히트맵으로 변환하는 웹 버전 - 모든 기능 포함
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import io

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="히트맵 생성기",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 히트맵 생성기")
st.markdown("---")

# 세션 상태 초기화
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

# 정규분포 피팅 함수
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

# 히트맵 데이터 준비 함수
def prepare_heatmap_data(df, data_start_row=9, use_smoothing=False, sigma_value=1.0):
    """히트맵 데이터 준비"""
    data_without_first_col = df.iloc[data_start_row:, 1:]  # A열 제외, 10행부터
    numeric_df = data_without_first_col.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return None, None, None
    
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    # 행/열 인덱스를 mm로 변환
    row_indices = np.arange(len(numeric_df))
    y_mm_values = pd.Series(row_indices / 3.8)
    
    col_indices = np.arange(len(numeric_df.columns))
    x_mm_values = pd.Series(col_indices / 3.8)
    
    # 샘플링
    max_rows = 500
    if len(numeric_df) > max_rows:
        step = len(numeric_df) // max_rows
        numeric_df = numeric_df.iloc[::step].head(max_rows)
        y_mm_values = y_mm_values.iloc[::step].head(max_rows).reset_index(drop=True)
    
    if len(numeric_df.columns) > max_rows:
        step_col = len(numeric_df.columns) // max_rows
        numeric_df = numeric_df.iloc[:, ::step_col].iloc[:, :max_rows]
        x_mm_values = x_mm_values.iloc[::step_col].head(max_rows).reset_index(drop=True)
    
    # 스무딩 적용
    if use_smoothing:
        numeric_df = pd.DataFrame(
            gaussian_filter(numeric_df.values, sigma=sigma_value),
            index=numeric_df.index,
            columns=numeric_df.columns
        )
    
    return numeric_df, x_mm_values, y_mm_values

# 사이드바 - 파일 업로드 및 설정
with st.sidebar:
    st.header("📁 파일 업로드")
    uploaded_files = st.file_uploader(
        "Excel 파일을 선택하세요",
        type=['xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.excel_data:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name='영역 데이터1', header=None)
                    st.session_state.excel_data[uploaded_file.name] = df
                    st.success(f"✅ {uploaded_file.name} 로드 완료")
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name} 로드 실패: {str(e)}")
    
    st.markdown("---")
    st.header("⚙️ 히트맵 설정")
    
    # 분석 모드 선택
    analysis_mode = st.radio(
        "분석 모드",
        ["일반 히트맵", "델타 히트맵"],
        index=0
    )
    
    # 파일 선택
    if st.session_state.excel_data:
        file_list = list(st.session_state.excel_data.keys())
        
        if analysis_mode == "델타 히트맵":
            st.subheader("델타 히트맵 설정")
            sheet1_name = st.selectbox("첫 번째 파일 (기준)", options=file_list, index=0)
            sheet2_name = st.selectbox("두 번째 파일 (비교)", options=file_list, index=min(1, len(file_list)-1))
        else:
            selected_file = st.selectbox("분석할 파일 선택", options=file_list)
            sheet1_name = None
            sheet2_name = None
    else:
        selected_file = None
        sheet1_name = None
        sheet2_name = None
    
    # 컬러맵 선택
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 
                 'RdYlBu', 'RdYlGn', 'Spectral', 'hot', 'cool']
    selected_cmap = st.selectbox("컬러맵", options=colormaps, index=0)
    
    # 스무딩 설정
    use_smoothing = st.checkbox("스무딩 적용", value=False)
    if use_smoothing:
        sigma_value = st.slider("Sigma 값", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    else:
        sigma_value = 1.0
    
    # 컬러바 범위 설정
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
    st.header("📊 분포도 설정")
    
    # Y축 분포도
    y_coord = st.number_input("Y 좌표 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    show_y_dist = st.button("Y축 분포도 생성")
    
    # X축 분포도
    x_coord = st.number_input("X 좌표 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    show_x_dist = st.button("X축 분포도 생성")
    
    st.markdown("---")
    st.header("📈 정규분포 피팅")
    
    show_normal_dist = st.checkbox("정규분포 표시", value=False)
    show_dist_sum = st.checkbox("합산 결과 표시", value=False)
    
    if st.button("현재 분포도 저장"):
        if st.session_state.current_numeric_df is not None:
            # 현재 표시된 분포도 저장 (간단한 구현)
            st.info("분포도를 생성한 후 저장하세요.")

# 메인 영역
tab1, tab2, tab3 = st.tabs(["🔥 히트맵", "📈 분포도", "⚙️ 정규분포 설정"])

with tab1:
    if analysis_mode == "델타 히트맵":
        if sheet1_name and sheet2_name and sheet1_name != sheet2_name:
            df1 = st.session_state.excel_data[sheet1_name]
            df2 = st.session_state.excel_data[sheet2_name]
            
            numeric_df1, x_mm1, y_mm1 = prepare_heatmap_data(df1, use_smoothing=use_smoothing, sigma_value=sigma_value)
            numeric_df2, x_mm2, y_mm2 = prepare_heatmap_data(df2, use_smoothing=use_smoothing, sigma_value=sigma_value)
            
            if numeric_df1 is not None and numeric_df2 is not None:
                # 두 데이터프레임의 크기를 맞춤
                min_rows = min(len(numeric_df1), len(numeric_df2))
                min_cols = min(len(numeric_df1.columns), len(numeric_df2.columns))
                
                numeric_df1 = numeric_df1.iloc[:min_rows, :min_cols].reset_index(drop=True)
                numeric_df1.columns = range(len(numeric_df1.columns))
                
                numeric_df2 = numeric_df2.iloc[:min_rows, :min_cols].reset_index(drop=True)
                numeric_df2.columns = range(len(numeric_df2.columns))
                
                # 델타 계산
                delta_df = numeric_df2 - numeric_df1
                delta_df = delta_df.fillna(delta_df.mean()).fillna(0)
                
                # 좌표값 계산
                row_indices = np.arange(len(delta_df))
                y_mm_values = pd.Series(row_indices / 3.8)
                col_indices = np.arange(len(delta_df.columns))
                x_mm_values = pd.Series(col_indices / 3.8)
                
                # 세션 상태에 저장
                st.session_state.current_numeric_df = delta_df
                st.session_state.current_x_mm_values = x_mm_values
                st.session_state.current_y_mm_values = y_mm_values
                
                # 컬러바 범위 설정
                if auto_range:
                    vmin = float(delta_df.min().min())
                    vmax = float(delta_df.max().max())
                else:
                    vmin = cbar_min
                    vmax = cbar_max
                
                # 히트맵 생성
                fig, ax = plt.subplots(figsize=(16, 10))
                
                # 레이블 생성
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
                
                # 히트맵 그리기
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
                
                # 컬러바 제목 설정
                cbar = heatmap.collections[0].colorbar
                cbar.ax.set_ylabel('ΔT [°C]', rotation=270, labelpad=20)
                cbar.ax.yaxis.label.set_rotation(270)
                cbar.ax.yaxis.label.set_x(1.4)
                cbar.ax.yaxis.label.set_va('center')
                cbar.ax.yaxis.label.set_ha('left')
                cbar.ax.tick_params(pad=15)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 다운로드 버튼
                col1, col2 = st.columns(2)
                with col1:
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format='png', dpi=150, bbox_inches='tight')
                    buf_png.seek(0)
                    st.download_button(
                        label="📥 히트맵 다운로드 (PNG)",
                        data=buf_png,
                        file_name=f"delta_heatmap_{sheet2_name}_{sheet1_name}.png",
                        mime="image/png"
                    )
                with col2:
                    buf_svg = io.BytesIO()
                    fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                    buf_svg.seek(0)
                    st.download_button(
                        label="📥 히트맵 다운로드 (SVG)",
                        data=buf_svg,
                        file_name=f"delta_heatmap_{sheet2_name}_{sheet1_name}.svg",
                        mime="image/svg+xml"
                    )
            else:
                st.warning("⚠️ 데이터를 준비할 수 없습니다.")
        else:
            st.warning("⚠️ 두 개의 서로 다른 파일을 선택하세요.")
    
    elif selected_file and selected_file in st.session_state.excel_data:
        df = st.session_state.excel_data[selected_file]
        
        numeric_df, x_mm_values, y_mm_values = prepare_heatmap_data(
            df, use_smoothing=use_smoothing, sigma_value=sigma_value
        )
        
        if numeric_df is not None:
            # 세션 상태에 저장
            st.session_state.current_numeric_df = numeric_df
            st.session_state.current_x_mm_values = x_mm_values
            st.session_state.current_y_mm_values = y_mm_values
            
            # 컬러바 범위 설정
            if auto_range:
                vmin = float(numeric_df.min().min())
                vmax = float(numeric_df.max().max())
            else:
                vmin = cbar_min
                vmax = cbar_max
            
            # 히트맵 생성
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # 레이블 생성
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
            
            # 히트맵 그리기
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
            
            # 컬러바 제목 설정
            cbar = heatmap.collections[0].colorbar
            cbar.ax.set_ylabel('Temperature [°C]', rotation=270, labelpad=20)
            cbar.ax.yaxis.label.set_rotation(270)
            cbar.ax.yaxis.label.set_x(1.4)
            cbar.ax.yaxis.label.set_va('center')
            cbar.ax.yaxis.label.set_ha('left')
            cbar.ax.tick_params(pad=15)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 다운로드 버튼
            col1, col2 = st.columns(2)
            with col1:
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format='png', dpi=150, bbox_inches='tight')
                buf_png.seek(0)
                st.download_button(
                    label="📥 히트맵 다운로드 (PNG)",
                    data=buf_png,
                    file_name=f"{selected_file}_heatmap.png",
                    mime="image/png"
                )
            with col2:
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                buf_svg.seek(0)
                st.download_button(
                    label="📥 히트맵 다운로드 (SVG)",
                    data=buf_svg,
                    file_name=f"{selected_file}_heatmap.svg",
                    mime="image/svg+xml"
                )
        else:
            st.warning("⚠️ 숫자 데이터를 찾을 수 없습니다.")
    else:
        st.info("👈 사이드바에서 Excel 파일을 업로드하세요.")

with tab2:
    st.header("📈 분포도 그래프")
    
    if st.session_state.current_numeric_df is None:
        st.info("먼저 히트맵을 생성하세요.")
    else:
        numeric_df = st.session_state.current_numeric_df
        x_mm_values = st.session_state.current_x_mm_values
        y_mm_values = st.session_state.current_y_mm_values
        
        # 분포도 생성
        if show_y_dist:
            y_mm_array = y_mm_values.values
            row_idx = np.argmin(np.abs(y_mm_array - y_coord))
            
            if 0 <= row_idx < len(numeric_df):
                row_data = numeric_df.iloc[row_idx, :].values
                x_coords = x_mm_values.values
                
                fig_dist, ax_dist = plt.subplots(figsize=(12, 5))
                ax_dist.plot(x_coords, row_data, 'b-', linewidth=2, marker='o', markersize=3, label='Data')
                
                # 정규분포 피팅
                fitted_params = detect_left_peak_and_fit_gaussian(x_coords, row_data)
                
                if show_normal_dist and fitted_params:
                    x_extended = np.linspace(x_coords.min(), x_coords.max(), 1000)
                    y_gaussian = fitted_params['amplitude'] * np.exp(-0.5 * ((x_extended - fitted_params['mu']) / fitted_params['sigma']) ** 2) + fitted_params['baseline']
                    ax_dist.plot(x_extended, y_gaussian, '--', linewidth=2, alpha=0.7, color='red', label='Fitted Gaussian')
                
                ax_dist.set_xlabel('X axis [mm]', fontsize=12)
                ax_dist.set_ylabel('Temperature [°C]', fontsize=12)
                ax_dist.set_title(f'Y-axis Distribution (Row {row_idx}, Y={y_mm_array[row_idx]:.2f}mm)', 
                                 fontsize=14, pad=15)
                ax_dist.grid(True, alpha=0.3)
                ax_dist.legend()
                plt.tight_layout()
                st.pyplot(fig_dist)
                
                # 분포도 저장
                if st.button("이 분포도 저장", key="save_y_dist"):
                    dist_id = st.session_state.distribution_id_counter
                    st.session_state.distribution_id_counter += 1
                    
                    st.session_state.distribution_data_list.append({
                        'id': dist_id,
                        'type': 'Y축',
                        'x': np.array(x_coords),
                        'y': np.array(row_data),
                        'label': f'Y-axis Distribution (Row {row_idx}, Y={y_mm_array[row_idx]:.2f}mm)',
                        'color': 'blue',
                        'fitted_params': fitted_params,
                        'normal_dist_params': []
                    })
                    st.success(f"분포도가 저장되었습니다! (ID: {dist_id})")
        
        if show_x_dist:
            x_mm_array = x_mm_values.values
            col_idx = np.argmin(np.abs(x_mm_array - x_coord))
            
            if 0 <= col_idx < len(numeric_df.columns):
                col_data = numeric_df.iloc[:, col_idx].values
                y_coords = y_mm_values.values
                
                fig_dist, ax_dist = plt.subplots(figsize=(12, 5))
                ax_dist.plot(y_coords, col_data, 'r-', linewidth=2, marker='o', markersize=3, label='Data')
                
                # 정규분포 피팅
                fitted_params = detect_left_peak_and_fit_gaussian(y_coords, col_data)
                
                if show_normal_dist and fitted_params:
                    y_extended = np.linspace(y_coords.min(), y_coords.max(), 1000)
                    y_gaussian = fitted_params['amplitude'] * np.exp(-0.5 * ((y_extended - fitted_params['mu']) / fitted_params['sigma']) ** 2) + fitted_params['baseline']
                    ax_dist.plot(y_extended, y_gaussian, '--', linewidth=2, alpha=0.7, color='green', label='Fitted Gaussian')
                
                ax_dist.set_xlabel('Y axis [mm]', fontsize=12)
                ax_dist.set_ylabel('Temperature [°C]', fontsize=12)
                ax_dist.set_title(f'X-axis Distribution (Column {col_idx}, X={x_mm_array[col_idx]:.2f}mm)', 
                                 fontsize=14, pad=15)
                ax_dist.grid(True, alpha=0.3)
                ax_dist.legend()
                plt.tight_layout()
                st.pyplot(fig_dist)
                
                # 분포도 저장
                if st.button("이 분포도 저장", key="save_x_dist"):
                    dist_id = st.session_state.distribution_id_counter
                    st.session_state.distribution_id_counter += 1
                    
                    st.session_state.distribution_data_list.append({
                        'id': dist_id,
                        'type': 'X축',
                        'x': np.array(y_coords),
                        'y': np.array(col_data),
                        'label': f'X-axis Distribution (Column {col_idx}, X={x_mm_array[col_idx]:.2f}mm)',
                        'color': 'red',
                        'fitted_params': fitted_params,
                        'normal_dist_params': []
                    })
                    st.success(f"분포도가 저장되었습니다! (ID: {dist_id})")
        
        # 저장된 분포도 목록
        if len(st.session_state.distribution_data_list) > 0:
            st.markdown("---")
            st.subheader("저장된 분포도")
            
            for idx, dist_data in enumerate(st.session_state.distribution_data_list):
                with st.expander(f"{dist_data['type']}: {dist_data['label']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"그래프 표시", key=f"show_dist_{idx}"):
                            fig_saved, ax_saved = plt.subplots(figsize=(12, 5))
                            ax_saved.plot(dist_data['x'], dist_data['y'], '-', linewidth=2, 
                                         color=dist_data['color'], label='Data')
                            
                            # 정규분포 표시
                            if show_normal_dist:
                                normal_dist_params = dist_data.get('normal_dist_params', [])
                                if len(normal_dist_params) > 0:
                                    for dist_idx, params in enumerate(normal_dist_params):
                                        x_extended = np.linspace(dist_data['x'].min(), dist_data['x'].max(), 1000)
                                        y_gaussian = params['amplitude'] * np.exp(-0.5 * ((x_extended - params['mu']) / params['sigma']) ** 2) + params['baseline']
                                        ax_saved.plot(x_extended, y_gaussian, '--', linewidth=2, alpha=0.7,
                                                     label=f"Normal Dist {dist_idx + 1}")
                                elif dist_data.get('fitted_params'):
                                    params = dist_data['fitted_params']
                                    x_extended = np.linspace(dist_data['x'].min(), dist_data['x'].max(), 1000)
                                    y_gaussian = params['amplitude'] * np.exp(-0.5 * ((x_extended - params['mu']) / params['sigma']) ** 2) + params['baseline']
                                    ax_saved.plot(x_extended, y_gaussian, '--', linewidth=2, alpha=0.7,
                                                 label='Fitted Gaussian')
                            
                            # 합산 결과 표시
                            if show_dist_sum and len(st.session_state.distribution_data_list) > 0:
                                all_x_min = min([d['x'].min() for d in st.session_state.distribution_data_list])
                                all_x_max = max([d['x'].max() for d in st.session_state.distribution_data_list])
                                x_range = all_x_max - all_x_min
                                x_extended = np.linspace(all_x_min - x_range * 0.1, all_x_max + x_range * 0.1, 1000)
                                
                                y_sum = np.zeros_like(x_extended)
                                baseline_sum = 0
                                count = 0
                                
                                for d in st.session_state.distribution_data_list:
                                    normal_dist_params = d.get('normal_dist_params', [])
                                    if len(normal_dist_params) > 0:
                                        for params in normal_dist_params:
                                            baseline_sum += params['baseline']
                                            count += 1
                                            y_sum += params['amplitude'] * np.exp(-0.5 * ((x_extended - params['mu']) / params['sigma']) ** 2)
                                    elif d.get('fitted_params'):
                                        params = d['fitted_params']
                                        baseline_sum += params['baseline']
                                        count += 1
                                        y_sum += params['amplitude'] * np.exp(-0.5 * ((x_extended - params['mu']) / params['sigma']) ** 2)
                                
                                if count > 0:
                                    avg_baseline = baseline_sum / count
                                    y_sum_total = y_sum + avg_baseline
                                    ax_saved.plot(x_extended, y_sum_total, '-', linewidth=3, alpha=0.8,
                                                 color='purple', label='Sum Result')
                            
                            ax_saved.set_xlabel('X axis [mm]' if dist_data['type'] == 'Y축' else 'Y axis [mm]', fontsize=12)
                            ax_saved.set_ylabel('Temperature [°C]', fontsize=12)
                            ax_saved.set_title(dist_data['label'], fontsize=14, pad=15)
                            ax_saved.grid(True, alpha=0.3)
                            ax_saved.legend()
                            plt.tight_layout()
                            st.pyplot(fig_saved)
                    
                    with col2:
                        if st.button(f"제거", key=f"remove_dist_{idx}"):
                            st.session_state.distribution_data_list.pop(idx)
                            st.rerun()

with tab3:
    st.header("⚙️ 정규분포 배치 설정")
    
    if len(st.session_state.distribution_data_list) == 0:
        st.info("먼저 분포도를 저장하세요.")
    else:
        dist_list = [f"{dist_data['type']}: {dist_data['label']}" for dist_data in st.session_state.distribution_data_list]
        selected_dist_idx = st.selectbox("분포도 선택", options=range(len(dist_list)), format_func=lambda x: dist_list[x])
        
        if selected_dist_idx is not None:
            dist_data = st.session_state.distribution_data_list[selected_dist_idx]
            
            # 왼쪽 봉우리 정보 표시
            if dist_data.get('fitted_params'):
                params = dist_data['fitted_params']
                st.info(f"**왼쪽 봉우리 (자동 검출)**: μ={params['mu']:.2f}mm, σ={params['sigma']:.2f}mm, "
                       f"진폭={params['amplitude']:.2f}℃, 기준선={params['baseline']:.2f}℃")
            
            # 정규분포 개수 설정
            num_distributions = st.number_input("정규분포 개수", min_value=0, max_value=10, value=len(dist_data.get('normal_dist_params', [])), step=1)
            
            # 정규분포 파라미터 입력
            normal_dist_params = []
            for i in range(num_distributions):
                st.subheader(f"정규분포 {i + 1}")
                col1, col2 = st.columns(2)
                with col1:
                    mu = st.number_input(f"중심 위치 μ (mm)", value=dist_data.get('fitted_params', {}).get('mu', 0.0) if i == 0 and dist_data.get('fitted_params') else 0.0, 
                                        key=f"mu_{selected_dist_idx}_{i}")
                    sigma = st.number_input(f"표준편차 σ (mm)", value=dist_data.get('fitted_params', {}).get('sigma', 2.0) if i == 0 and dist_data.get('fitted_params') else 2.0,
                                           min_value=0.1, step=0.1, key=f"sigma_{selected_dist_idx}_{i}")
                with col2:
                    amplitude = st.number_input(f"진폭 (℃)", value=dist_data.get('fitted_params', {}).get('amplitude', 5.0) if i == 0 and dist_data.get('fitted_params') else 5.0,
                                               step=0.1, key=f"amp_{selected_dist_idx}_{i}")
                    baseline = st.number_input(f"기준선 (℃)", value=dist_data.get('fitted_params', {}).get('baseline', 20.0) if i == 0 and dist_data.get('fitted_params') else 20.0,
                                               step=0.1, key=f"base_{selected_dist_idx}_{i}")
                
                normal_dist_params.append({
                    'mu': mu,
                    'sigma': sigma,
                    'amplitude': amplitude,
                    'baseline': baseline
                })
            
            if st.button("설정 적용"):
                st.session_state.distribution_data_list[selected_dist_idx]['normal_dist_params'] = normal_dist_params
                st.success("정규분포 배치 설정이 적용되었습니다!")
                st.rerun()

# 사용 방법 안내
with st.expander("📖 사용 방법"):
    st.markdown("""
    ### 히트맵 생성기 사용 방법
    
    1. **파일 업로드**
       - 사이드바에서 Excel 파일을 선택하세요
       - 여러 파일을 동시에 업로드할 수 있습니다
       - 파일은 '영역 데이터1' 시트에서 A열 10행부터 데이터를 읽습니다
    
    2. **히트맵 생성**
       - 일반 히트맵: 단일 파일의 히트맵 생성
       - 델타 히트맵: 두 파일의 차이값 히트맵 생성
       - 컬러맵, 스무딩, 컬러바 범위 등을 설정할 수 있습니다
    
    3. **분포도 생성**
       - Y 좌표 또는 X 좌표를 입력하고 버튼을 클릭하세요
       - 해당 좌표의 온도 분포도를 확인할 수 있습니다
       - 분포도를 저장하여 나중에 사용할 수 있습니다
    
    4. **정규분포 피팅**
       - 저장된 분포도에 대해 정규분포를 피팅할 수 있습니다
       - 여러 정규분포를 수동으로 추가하고 합산할 수 있습니다
    
    ### 주요 기능
    - ✅ 여러 파일 동시 업로드
    - ✅ 일반 히트맵 및 델타 히트맵 생성
    - ✅ 다양한 컬러맵 선택
    - ✅ 스무딩 필터 적용
    - ✅ 컬러바 범위 조정
    - ✅ Y축/X축 분포도 생성
    - ✅ 정규분포 피팅 및 합산
    - ✅ 히트맵 이미지 다운로드 (PNG, SVG)
    """)
