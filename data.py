import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================
# 1. 讀取與初始化
# ==========================================
file_path = '山东数据.xls'
sheet_name = '负荷联络线新能源出力'  # 請確保這個 Sheet 名稱與你的 Excel 一致

try:
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    # 強行清理列名空格，防止 KeyError
    df_raw.columns = df_raw.columns.str.strip()
    print("✅ 成功讀取 Excel！")
except Exception as e:
    print(f"❌ 讀取失敗，請檢查 Sheet 名稱或路徑：{e}")
    exit()

# ==========================================
# 2. 提取核心列與數據清洗
# ==========================================
cols_map = {
    '日期': 'ts',
    '全网(山东).负荷.负荷': 'L',
    '全网(山东).风电.理论': 'W',
    '全网(山东).光伏(集中式).理论': 'S_c',
    '全网(山东).光伏(分布式).理论': 'S_d'
}

# 僅提取存在的列並重命名
df = df_raw[list(cols_map.keys())].rename(columns=cols_map)

# 合併光伏並填補空值（使用新版 Pandas 推薦語法 ffill/bfill 消除警告）
df['S'] = df['S_c'] + df['S_d']
df = df[['ts', 'L', 'W', 'S']].ffill().bfill()

# ==========================================
# 3. 計算標籤 (Labels)
# ==========================================
# A. 季節標籤：0-春, 1-夏, 2-秋, 3-冬
df['ts'] = pd.to_datetime(df['ts'])
df['month'] = df['ts'].dt.month


def get_season(m):
    if m in [3, 4, 5]: return 0
    if m in [6, 7, 8]: return 1
    if m in [9, 10, 11]: return 2
    return 3


df['season_label'] = df['month'].apply(get_season)

# B. 歸一化 (0-1) - GAN 訓練必需
scaler = MinMaxScaler()
df[['L', 'W', 'S']] = scaler.fit_transform(df[['L', 'W', 'S']])
# 保存縮放器，未來將生成出的 0-1 數據還原為 MW 時需要它
joblib.dump(scaler, 'shandong_scaler.pkl')

# ==========================================
# 4. 數據切片與壓平 (52週樣本製作)
# ==========================================
hours_per_week = 168
num_weeks = len(df) // hours_per_week
final_data = []

# 計算出力水平標籤所需的周平均值
df['total_re'] = df['W'] + df['S']  # 風+光總出力

for i in range(num_weeks):
    # 切出這一週的 168 小時數據
    week_chunk = df.iloc[i * hours_per_week: (i + 1) * hours_per_week]

    # 壓平三種曲線：L(168) + W(168) + S(168) = 504個數值
    flat_features = np.concatenate([
        week_chunk['L'].values,
        week_chunk['W'].values,
        week_chunk['S'].values
    ])

    # 獲取標籤
    season = week_chunk['season_label'].iloc[0]

    # 計算出力水平標籤 (基於週平均出力劃分 0-低, 1-中, 2-高)
    avg_re = week_chunk['total_re'].mean()
    if avg_re < 0.3:
        output_lvl = 0
    elif avg_re < 0.6:
        output_lvl = 1
    else:
        output_lvl = 2

    # 將數值與標籤合併
    full_row = np.append(flat_features, [season, output_lvl])
    final_data.append(full_row)

# ==========================================
# 5. 保存最終 CSV
# ==========================================
column_names = [f'L{i}' for i in range(168)] + \
               [f'W{i}' for i in range(168)] + \
               [f'S{i}' for i in range(168)] + \
               ['season_label', 'output_label']

df_final = pd.DataFrame(final_data, columns=column_names)
df_final.to_csv('shandong_gan_ready_final.csv', index=False)

print(f"🎉 預處理大功告成！")
print(f"生成的檔案：shandong_gan_ready_final.csv")
print(f"樣本總數：{len(df_final)} 週")
print(f"每行維度：{len(df_final.columns)} (504個數據點 + 2個標籤)")