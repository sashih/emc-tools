import matplotlib.pyplot as plt
import numpy as np
from emc_tools import * # 0. 設定檔案路徑
fname = "./data/GLAD-M35.r0.1-n4.nc"

# 1. 快速檢視檔案結構
print("Step 1: Inspecting NetCDF structure...")
inspect_netcdf(fname)

# 2. 提取特定深度與變數
print("\nStep 2: Extracting data slice...")
target_depth = 2800  
lons, lats, data_slice, actual_depth = get_nc_slice(fname, 'vsh', target_depth)

# 3. 計算震波速度偏差 (Velocity Anomaly %)
print("\nStep 3: Calculating surface-weighted anomaly...")
dv_anomaly = get_velocity_anomaly(lons, lats, data_slice)

# 4. 繪製原始 2D 地圖 (Mapview)
print("\nStep 4: Plotting mapview...")
plot_map_basemap(lons, lats, dv_anomaly, 
                 title=f"Original Vsh Anomaly at {actual_depth} km")
plt.savefig('test_map.jpg', dpi=300, bbox_inches='tight')

# 5. 重採樣至均勻分佈網格 (Fibonacci Sphere)
print("\nStep 5: Resampling to even grid (nu=16)...")
even_lon, even_lat, even_dv = resample_tomography(lons, lats, dv_anomaly, nu=16)

# 6. 視覺化統計結果 (PDF)
print("\nStep 6: Comparing Statistics (PDF)...")
plt.figure(figsize=(8, 5))
plt.hist(dv_anomaly.flatten(), bins=np.linspace(-5, 5, 51), density=True, histtype='step', label='Raw Grid')
plt.hist(even_dv, bins=np.linspace(-5, 5, 51), density=True, histtype='step', label='Even Grid')
plt.legend()
plt.savefig('test_hist.jpg')

# 7. 檢查均勻網格分佈
print("\nStep 7: Checking even grid distribution...")
plt.figure()
plt.scatter(even_lon, even_lat, s=1)
plt.savefig('test_grid.jpg')

# --- 新增：球諧函數分析與合成 ---

# 8. 球諧函數展開 (Analysis)
print("\nStep 8: Spherical Harmonics Analysis (nmax=20)...")
sh, zlm, Vl = run_sh_analysis(dv_anomaly, nmax=20)

# 9. 從係數重新合成地圖 (Synthesis / Consistency Check)
print("Step 9: Reconstructing field from coefficients (Synthesis)...")
recon_data = reconstruct_from_sh(sh, zlm)

# 繪製重構後的對照圖
plot_map_basemap(lons, lats, recon_data, 
                 title=f"Reconstructed Vsh Anomaly (nmax=20)")
plt.savefig('test_reconstruction.jpg', dpi=300, bbox_inches='tight')

print("\nAll tasks completed! Results saved as jpg files.")