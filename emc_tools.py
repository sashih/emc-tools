import netCDF4
import numpy as np

def main():
    fname = "./data/GLAD-M35.r0.1-n4.nc"
    inspect_netcdf(fname)



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