import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import shtns

def main():
    fname = "./data/GLAD-M35.r0.1-n4.nc"
    inspect_netcdf(fname)

import numpy as np

def degree_amplitude(nmax, l_array, zlm):
    """
    計算各階 (Degree l) 的振幅能譜 (Power Spectrum)。
    """
    Vl = np.zeros(nmax + 1)
    for l in range(nmax + 1):
        # 找出所有符合當前階數 l 的索引 (包含不同的 m)
        idx = (l_array == l)
        # Power Spectrum: sum of |zlm|^2 for each l
        Vl[l] = np.sqrt(np.sum(np.abs(zlm[idx])**2) / (2*l + 1))
    return Vl

def run_sh_analysis(data_2d, nmax=20):
    """
    對 2D 網格數據進行球諧函數展開。
    
    Args:
        data_2d: 2D 陣列 [nlat, nlon]，需為 regular grid (含兩極)
        nmax: 最高展開階數
    """
    nlat, nlon = data_2d.shape
    data_ready = np.array(data_2d, dtype=np.float64, copy=True)

    # 1. 初始化 shtns 對象
    # mmax 通常設為與 nmax 相同
    sh = shtns.sht(nmax, nmax)
    
    # 2. 設定網格 (指定為包含兩極的常規網格)
    sh.set_grid(nlat, nlon, flags=shtns.sht_reg_poles)
    
    # 3. 執行正向變換 (Analysis)
    # zlm 是複數陣列，存儲球諧係數
    zlm = sh.analys(data_ready)
    
    # 4. 計算能譜 (Degree Amplitude)
    Vl = degree_amplitude(nmax, sh.l, zlm)
    
    return sh, zlm, Vl

def reconstruct_from_sh(sh, zlm):
    """
    將球諧係數 zlm 重新合成回空間域 (Synthesis)。
    
    Args:
        sh: 已經初始化過的 shtns 對象
        zlm: 球諧展開係數
    Returns:
        recon_data: 重構後的 2D 空間陣列 [nlat, nlon]
    """
    # 執行反向變換 (Synthesis)
    # y = sh.synth(zlm) 會根據 sh 當初設定的 grid (nlat, nlon) 產出資料
    recon_data = sh.synth(zlm)
    
    return recon_data


def generate_fibonacci_grid(nu=16):
    """
    產生均勻分佈的全球採樣點 (Fibonacci Sphere)。
    """
    n = nu**2 * 10 
    phi = np.pi * (3. - np.sqrt(5.))  # 黃金角度
    
    indices = np.arange(n)
    y = 1 - (indices / float(n - 1)) * 2  # y 從 1 到 -1
    radius = np.sqrt(1 - y * y) 
    
    theta = phi * indices 
    
    x = np.cos(theta) * radius
    y_coords = y # 修正變數名稱避免混淆
    z = np.sin(theta) * radius
    
    # 轉換為經緯度
    lon = np.rad2deg(np.arctan2(z, x))
    lat = 90 - np.rad2deg(np.arccos(y_coords)) # 使用 arccos(y) 較直接
    
    return lon, lat

def resample_tomography(old_lons, old_lats, data_2d, nu=16, method='linear'):
    """
    將傳統經緯度網格數據重採樣到均勻分佈的網格上。
    
    Args:
        old_lons, old_lats: 原始 1D 經緯度陣列
        data_2d: 原始 2D 數據 (shape: [n_lat, n_lon])
        nu: 控制網格密度的參數
        method: 插值方法 'linear', 'nearest', 'cubic'
    """
    # 1. 產生均勻網格點
    new_lons, new_lats = generate_fibonacci_grid(nu=nu)
    
    # 2. 準備原始數據的座標點
    # griddata 需要 (N, 2) 的座標輸入
    lon_mg, lat_mg = np.meshgrid(old_lons, old_lats)
    points = np.vstack((lon_mg.flatten(), lat_mg.flatten())).T
    values = data_2d.flatten()
    
    # 3. 執行插值
    new_points = np.vstack((new_lons, new_lats)).T
    resampled_data = griddata(points, values, new_points, method=method)
    
    # 處理可能的 NaN (邊界問題)
    if np.isnan(resampled_data).any():
        resampled_data = griddata(points, values, new_points, method='nearest')
        print(f"[{'WARN':^10}] 部分點使用最近鄰插值填補 NaN")

    print(f"[{'DONE':^10}] 已重採樣至 {len(new_lons)} 個均勻分佈點 (nu={nu})")
    return new_lons, new_lats, resampled_data


def calculate_surface_average(lons, lats, data):
    """
    計算全球表面加權平均值。
    """
    # 1. 將緯度轉為餘緯 (Colatitude) theta，並轉為弧度
    # 假設 lats 範圍是 -90 到 90
    theta = np.deg2rad(90 - lats) 
    
    # 2. 自動計算網格間距 (弧度)
    # 這裡假設是均勻網格，取第一個差值
    dtheta = np.abs(np.deg2rad(lats[1] - lats[0]))
    dphi = np.abs(np.deg2rad(lons[1] - lons[0]))
    
    # 3. 建立 2D 的 sin(theta) 權重矩陣，形狀需與 data 一致 [n_lat, n_lon]
    # 使用 np.meshgrid 或直接利用 broadcasting
    weights = np.sin(theta).reshape(-1, 1) # 轉為 [n_lat, 1] 方便與 [n_lat, n_lon] 相乘
    
    # 4. 計算加權總和與總面積
    # 面積元素 dA = R^2 * sin(theta) * dtheta * dphi (R^2 可約掉)
    total_weighted_sum = np.sum(data * weights) * dtheta * dphi
    total_area = np.sum(np.ones_like(data) * weights) * dtheta * dphi   # compare to 4*pi 

    surf_avg = total_weighted_sum / total_area
    return surf_avg

def get_velocity_anomaly(lons, lats, data):
    """
    計算百分比偏差: (x - mean) / mean * 100%
    """
    avg = calculate_surface_average(lons, lats, data)
    
    # 套用公式: (x - avg) / avg * 100
    anomaly = ((data - avg) / avg) * 100
    
    print(f"[{'CALC':^10}] 球面平均值: {avg:.4f} | 偏差範圍: {np.min(anomaly):.2f}% ~ {np.max(anomaly):.2f}%")
    return anomaly

def plot_map_basemap(lons, lats, data, title="Mapview", unit="%", cmap="RdBu_r"):
    """
    使用 Basemap 繪製全球 2D 地圖。
    """
    fig = plt.figure(figsize=(12, 8))
    
    # 1. 初始化地圖投影 (cyl 代表 Cylindrical Equidistant，類似 PlateCarree)
    # llcrnrlat/lon: 左下角緯度/經度；urcrnrlat/lon: 右上角
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, 
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    
    # 2. 繪製地理特徵
    m.drawcoastlines(linewidth=0.8)
    m.drawcountries(linewidth=0.5, linestyle=':')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0,0,0,1], fontsize=10)

    # 3. 準備網格數據
    # Basemap 的 pcolormesh 通常需要 2D 的經緯度矩陣
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    x, y = m(lon_2d, lat_2d) # 將經緯度轉換為地圖投影座標

    # 4. 繪製數據
    # 注意: Basemap 的 pcolormesh 處理方式與 Cartopy 略有不同
    im = m.pcolormesh(x, y, data, cmap=cmap, shading='auto')

    # 5. 顏色條與標題
    cbar = m.colorbar(im, location='bottom', pad="10%")
    
    plt.title(title, fontsize=14)
    return fig, m

# 測試呼叫：
# plot_map_basemap(lons, lats, data_slice, title="dVs at 2800km")
# plt.show()


def get_nc_slice(file_path: str, var_name: str, target_depth: float):
    """
    從 netCDF 讀取 3D 變數，並提取最接近指定深度的 2D 切片。
    
    Args:
        file_path: netCDF 檔案路徑
        var_name: 欲提取的變數名稱 (e.g., 'vsh', 'vsv')
        target_depth: 目標深度 (km)
        
    Returns:
        tuple: (lons, lats, data_slice, actual_depth)
    """
    try:
        with netCDF4.Dataset(file_path, 'r') as nc:
            # 1. 檢查變數是否存在
            if var_name not in nc.variables:
                available = [v for v in nc.variables if nc.variables[v].ndim == 3]
                raise ValueError(f"變數 '{var_name}' 不存在。可用 3D 變數: {available}")

            # 2. 讀取座標 (假設標準名稱為 longitude, latitude, depth)
            lons = nc.variables['longitude'][:]
            lats = nc.variables['latitude'][:]
            depths = nc.variables['depth'][:]

            # 3. 尋找最接近 target_depth 的索引
            # 使用 NumPy 的 argmin 找到差值絕對值最小的位置
            idx = np.argmin(np.abs(depths - target_depth))
            actual_depth = depths[idx]

            # 4. 提取 2D 切片 (假設維度順序為 [depth, lat, lon])
            # [:] 會將數據讀入記憶體成為 NumPy array
            data_slice = nc.variables[var_name][idx, :, :]

            print(f"[{'INFO':^10}] 目標深度: {target_depth} km | 實際提取: {actual_depth} km (Index: {idx})")
            
            return lons, lats, data_slice, actual_depth

    except Exception as e:
        print(f"[{'ERROR':^10}] 提取失敗: {e}")
        return None

def inspect_netcdf(file_path: str, return_metadata: bool = False):
    """
    開啟並讀取 netCDF 檔案資訊。
    
    Args:
        file_path: 檔案路徑
        return_metadata: 是否以字典形式回傳變數名稱與維度
    """
    try:
        with netCDF4.Dataset(file_path, 'r') as nc_file:
            print(f"[{'SUCCESS':^10}] 成功開啟: {file_path}")
            
            # 1. 全域屬性 (Global Attributes)
            print("\n>>> 全域屬性 (Global Attributes)")
            attrs = nc_file.ncattrs()
            if not attrs:
                print("   (無全域屬性)")
            for attr in attrs:
                print(f"   - {attr}: {getattr(nc_file, attr)}")

            # 2. 維度 (Dimensions)
            print("\n>>> 維度 (Dimensions)")
            for dim_name, dim in nc_file.dimensions.items():
                print(f"   - {dim_name}: {len(dim)}")

            # 3. 變數 (Variables)
            print("\n>>> 變數 (Variables)")
            var_info = {}
            for var_name, var in nc_file.variables.items():
                shape = var.shape
                dtype = var.dtype
                print(f"   - {var_name:<15} | Shape: {str(shape):<15} | Dtype: {dtype}")
                var_info[var_name] = {"shape": shape, "dtype": dtype}

            if return_metadata:
                return var_info

    except FileNotFoundError:
        print(f"[{'ERROR':^10}] 找不到檔案: {file_path}")
    except Exception as e:
        print(f"[{'ERROR':^10}] 發生錯誤: {e}")

# 使用範例
# inspect_netcdf("your_data.nc")

if __name__ == "__main__":
    main() 