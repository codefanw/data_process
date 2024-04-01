import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

data = {
    "NAME": ["InstructBLIP-8.2B", "IDEFICS-9B", "Qwen-VL-Chat-9.6B", "mPLUG-Owl2-8.2B", "LLaVA-1.5-7.2B", "LLaVA-HR-7.4B"],
    "GQA": [49.2, np.nan, 57.5, 56.1, 62.0, 64.2],
    "VQAv2": [np.nan, 50.9, 78.2, 79.4, 78.5, 81.9],
    "TextVQA": [50.1, 25.9, 61.5, 58.2, 58.2, 67.1],
    "VizWiz": [34.5, 35.5, 38.9, 54.5, 50.0, 48.7],
    "MME": [np.nan, np.nan, 1487.5, 1450.2, 1510.7, 1554.9],
    "OKVQA": [np.nan, 38.4, 56.6, 57.7, np.nan, 58.9],
    "POPE": [np.nan, np.nan, np.nan, np.nan, 85.9, 87.6],
    "SEED": [53.4, np.nan, 58.2, 57.8, 58.6, 64.2],
}

df = pd.DataFrame(data)

min_values = df.iloc[:, 1:].replace(0, np.nan).min()
baseline_values = min_values * 0.9
df_normalized = df.copy()
for column in df.columns[1:]:
    min_val = baseline_values[column]
    max_val = df[column].max()
    df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
df_normalized.fillna(np.nan, inplace=True)

#插值
def custom_interpolate_circular(row):
    n = len(row)
    for i in range(1, n):
        if pd.isna(row[i]):
            prev_val = next_val = None
            for j in range(1, n):
                prev_index = (i - j) % n
                next_index = (i + j) % n
                if prev_val is None and not pd.isna(row[prev_index]):
                    prev_val = row[prev_index]
                if next_val is None and not pd.isna(row[next_index]):
                    next_val = row[next_index]
                if prev_val is not None and next_val is not None:
                    break
            if prev_val is not None and next_val is not None:
                row[i] = (prev_val + next_val) / 2
    return row

df_values_only = df_normalized.iloc[:, 1:]  # 排除“NAME”列
df_values_only_interpolated_circular = df_values_only.apply(custom_interpolate_circular, axis=1)
df_normalized_custom_interpolated_circular_fixed = pd.concat([df_normalized[['NAME']], df_values_only_interpolated_circular], axis=1)

instructblip_vqav2_interp = (df_normalized_custom_interpolated_circular_fixed.loc[0, 'GQA'] + df_normalized_custom_interpolated_circular_fixed.loc[0, 'SEED']) / 2
df_normalized_custom_interpolated_circular_fixed.at[0, 'VQAv2'] = instructblip_vqav2_interp

# 绘制雷达图
def create_radar_chart_custom_fixed(df, df_normalized):
    labels = np.array(df.columns[1:])
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    # 更新颜色配置，确保LLaVA-HR-7.4B为红色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    texts = []
    for i, row in df_normalized.iterrows():
        data = row.drop('NAME').tolist()
        data += data[:1]
        if df['NAME'][i] == "LLaVA-HR-7.4B":
            linestyle = 'solid'  # 粗线
            linewidth = 4  # 线宽
            color = "#d62728"  # 红色
        else:
            linestyle = 'dashed'  # 虚线
            linewidth = 2  # 默认线宽
            color = colors[i % len(colors)]        
        ax.plot(angles, data, color=color, linewidth=linewidth, linestyle=linestyle, label=df['NAME'][i])
        ax.fill(angles, data, color=color, alpha=0.25)

        real_data = df.iloc[i, 1:].values
        for j, (angle, rd) in enumerate(zip(angles[:-1], real_data)):
            if pd.notna(rd):
                texts.append(ax.text(angle, data[j], f"{rd:.1f}", color='black', ha='center', va='center', fontsize=11))

    ax.set_yticklabels([])
    label_positions = np.degrees(angles[:-1])
    ax.set_thetagrids(label_positions, ['' for _ in labels], fontsize=12, weight='bold')  # 置空标签内容

    '''
    for label, rotation in zip(labels, label_positions):
        ax.text(np.radians(rotation), 1.12, label,  # 调整任务标签的放置位置
                color='black', ha='center', va='center', fontsize=15)
    '''

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    return fig, ax

fig, ax = create_radar_chart_custom_fixed(df, df_normalized_custom_interpolated_circular_fixed)
combined_file_path = './below_10B.png'
plt.savefig(combined_file_path, bbox_inches='tight')