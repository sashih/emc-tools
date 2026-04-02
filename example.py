import matplotlib.pyplot as plt
import numpy as np
from emc_tools import * # 假設你將所有函數存在 emc_tools.py

# 0. 設定檔案路徑 (建議使用 GLAD-M35 等全球模型)
fname = "./data/GLAD-M35.r0.1-n4.nc"

# 1. 快速檢視檔案結構 (Variables, Dimensions, Attributes)
print("Step 1: Inspecting NetCDF structure...")
inspect_netcdf(fname)

# 2. 提取特定深度與變數 (例如：核幔邊界 CMB 附近的 Vsh)
print("\nStep 2: Extracting data slice...")
target_depth = 2800  # km
lons, lats, data_slice, actual_depth = get_nc_slice(fname, 'vsh', target_depth)

# 3. 計算震波速度偏差 (Velocity Anomaly %)
# 基於球面面積加權平均：(x - mean)/mean * 100
print("\nStep 3: Calculating surface-weighted anomaly...")
dv_anomaly = get_velocity_anomaly(lons, lats, data_slice)

# 4. 繪製全球 2D 地圖 (Mapview)
print("\nStep 4: Plotting mapview...")
fig_map, m = plot_map_basemap(lons, lats, dv_anomaly, 
                               title=f"Vsh Anomaly at {actual_depth} km",
                               unit="%")
plt.savefig('test_map.jpg', dpi=300, bbox_inches='tight')

# 5. 重採樣至均勻分佈網格 (Fibonacci Sphere)
# 這是統計分析最重要的步驟，避免極區權重過高
print("\nStep 5: Resampling to even grid (nu=16)...")
even_lon, even_lat, even_dv = resample_tomography(lons, lats, dv_anomaly, nu=16)

# 6. 視覺化統計結果：比較原始網格與均勻網格的 PDF
print("\nStep 6: Comparing Statistics (PDF)...")
fig, ax = plt.subplots(figsize=(8, 5))

# 原始網格 (通常會因為極區過度採樣而使分佈稍微偏向某些值)
ax.hist(dv_anomaly.flatten(), bins=np.linspace(-5, 5, 51), 
        density=True, histtype='step', lw=2, label='Raw Lat-Lon Grid', color='C0')

# 均勻網格 (真實反映體積/面積比例的分佈)
ax.hist(even_dv, bins=np.linspace(-5, 5, 51), 
        density=True, histtype='step', lw=2, label='Fibonacci Even Grid', color='C1')

ax.set_xlabel('Velocity Anomaly dV (%)')
ax.set_ylabel('Probability Density')
ax.set_title(f'Statistical Distribution at {actual_depth} km')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('test_hist.jpg', dpi=300)

# 7. 檢查均勻網格的點分佈
print("\nStep 7: Checking even grid distribution...")
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(even_lon, even_lat, s=1, color='black', alpha=0.5)
ax.set_title(f'Fibonacci Grid Points (N={len(even_lat)})')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.savefig('test_grid.jpg', dpi=300)

print("\nAll tasks completed! Results saved as jpg files.")