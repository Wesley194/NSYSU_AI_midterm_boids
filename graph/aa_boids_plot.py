import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# 檔案設定
file_name = """data/record/record_2025-12-10_17-42-35.json"""
OUTPUT_DIR = "graph"

# --- 指標列表設定 ---
# 嵌套指標：位於 'birds' 列表內，需要計算平均值
NESTED_BIRDS_METRICS = [
    'Size', 'MIN_Speed', 'MAX_Speed', 'Perception_Radius', 
    'Separation_Weight', 'Alignment_Weight', 'Cohesion_Weight', 
    'Flee_Weight', 'Alert_Radius', 'Fitness', 'MAX_Stamina', 
    'Survival_Time'
]

# 頂層指標：位於 JSON 結構的最外層，與 'time' 和 'birds' 平級
# 您可以將任何新的頂層指標（如環境溫度等）添加到此處
TOP_LEVEL_METRICS = ['Eat_Frequency'] 


# ----------------------------------------------------------------------
# 核心數據處理函數 (只讀取一次)
# ----------------------------------------------------------------------

def read_json_data(file_name):
    """
    只讀取一次 JSON 檔案，並將頂層數據轉換為 Pandas DataFrame。
    """
    print(f"--- 1. Reading file: {file_name} ---")
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_name}.")
        return None

    if not data:
        print("Error: JSON file contains no data.")
        return None
    
    # 轉換為 DataFrame，每個頂層物件成為一行 (包含 time, Eat_Frequency, birds...)
    df_raw = pd.DataFrame(data)
    
    # 預先計算相對時間
    if 'time' in df_raw.columns:
        initial_time = df_raw['time'].min()
        df_raw['relative_time_s'] = (df_raw['time'] - initial_time) / 1000
    else:
        print("Warning: 'time' column not found in data.")
        return None
        
    print("Data read and initial processing completed successfully.")
    return df_raw

# ----------------------------------------------------------------------
# 通用繪圖函數 1：處理頂層指標 (通用化)
# ----------------------------------------------------------------------

def plot_top_level_metric_trend(df_raw, metric_name, output_dir):
    """
    通用函式：繪製任何頂層指標（不需要分組平均）對 time 的趨勢圖。
    """
    
    # 檢查指定的指標欄位是否存在
    if metric_name not in df_raw.columns:
        print(f"--- ❌ Error Plotting Top-Level {metric_name} ---")
        print(f"Metric column '{metric_name}' not found in the top-level data.")
        return None
    
    # 創建輸出目錄 (如果不存在)
    os.makedirs(output_dir, exist_ok=True)
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.plot(df_raw['relative_time_s'], df_raw[metric_name], marker='o', linestyle='-', markersize=4)
    
    # --- 輸出欄位設定 ---
    plt.title(f"Time Series Trend of {metric_name}", fontsize=14)
    plt.xlabel("Relative Time (seconds)", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 優化 X 軸刻度
    x_ticks_count = len(df_raw['relative_time_s'])
    step = max(1, x_ticks_count // 10) 
    plt.xticks(df_raw['relative_time_s'][::step], rotation=45)
    
    # 儲存圖片
    output_filename = os.path.join(output_dir, f"{metric_name.lower()}_time_series_plot.png")
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved for Top-Level {metric_name} to: {output_filename}")
    return output_filename

# ----------------------------------------------------------------------
# 繪圖函數 2：處理嵌套的 'birds' 數據 (來自前一個回答)
# ----------------------------------------------------------------------

def flatten_and_prepare_birds_data(df_raw):
    """
    將 df_raw 中的 'birds' 欄位（嵌套數據）扁平化。
    """
    if 'birds' not in df_raw.columns:
        print("Warning: 'birds' column not found for nested analysis.")
        return None

    # 使用 json_normalize 處理每一行中的 'birds' 列表
    # meta=['time', 'relative_time_s'] 確保時間資訊傳遞給每一筆個體數據
    df_flat = pd.json_normalize(df_raw.to_dict('records'), 
                                record_path='birds', 
                                meta=['time', 'relative_time_s'])
    return df_flat

def plot_birds_metric_trend(df_flat, metric_name, output_dir):
    """
    繪製 birds 嵌套數據中，指定指標的平均值對 time 的趨勢圖。（需要分組平均）
    """
    
    # 檢查欄位是否存在
    if metric_name not in df_flat.columns:
         print(f"--- ❌ Error Plotting Nested {metric_name} ---")
         return None

    # 彙總數據：計算每個時間步長下該指標的平均值
    df_summary = df_flat.groupby('relative_time_s')[metric_name].mean().reset_index()

    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.plot(df_summary['relative_time_s'], df_summary[metric_name], marker='o', linestyle='-')
    
    # --- 輸出欄位設定 ---
    plt.title(f"Time Series Trend of Average {metric_name}", fontsize=14)
    plt.xlabel("Relative Time (seconds)", fontsize=12)
    plt.ylabel(f"Average {metric_name}", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 儲存圖片
    output_filename = os.path.join(output_dir, f"average_{metric_name.lower()}_time_series_plot.png")
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved for Average {metric_name} to: {output_filename}")
    return output_filename


# ----------------------------------------------------------------------
# 主執行流程
# ----------------------------------------------------------------------

def main():
    """
    主執行函式：控制流程，讀取一次數據後，分別處理頂層和嵌套指標。
    """
    
    # 步驟 1: 讀取所有 JSON 數據並生成頂層 DataFrame (只執行一次)
    df_raw_data = read_json_data(file_name)
    
    if df_raw_data is None:
        return

    # ------------------------------------------------------------------
    # 步驟 2: 處理頂層指標 (使用通用的 plot_top_level_metric_trend)
    # ------------------------------------------------------------------
    print("\n--- 2. Starting Plotting Top-Level Metrics ---")
    for metric in TOP_LEVEL_METRICS:
        plot_top_level_metric_trend(df_raw_data, metric, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 步驟 3: 處理嵌套指標 (Birds Data)
    # ------------------------------------------------------------------
    print("\n--- 3. Starting Plotting Nested Birds Metrics ---")
    
    # 扁平化嵌套數據 (如果需要)
    df_birds_flat = flatten_and_prepare_birds_data(df_raw_data)
    
    if df_birds_flat is not None:
        for metric in NESTED_BIRDS_METRICS:
            plot_birds_metric_trend(df_birds_flat, metric, OUTPUT_DIR)

    print("\n--- 4. All plots generated. ---")

# 程式入口點
if __name__ == "__main__":
    main()