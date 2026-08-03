import os
import pandas as pd

def check_missing_dates(csv_path: str, start_date: str = "2023-01-02", end_date: str = "2025-12-31"):
    try:
        # Đọc file và xóa khoảng trắng ở tên cột
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"Không tìm thấy file CSV tại đường dẫn: {csv_path}")
        return
        
    if 'date' not in df.columns:
        print(f"Không tìm thấy cột 'date'. Các cột hiện tại: {df.columns.tolist()}")
        return
        
    df['date_clean'] = pd.to_datetime(df['date'].astype(str).str.strip(), errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date_clean'])
    existing_dates = set(df['date_clean'])
    
    expected_range = pd.date_range(start=start_date, end=end_date, freq='D')
    expected_dates = set(expected_range.strftime('%Y-%m-%d'))

    missing_dates = sorted(list(expected_dates - existing_dates))
    
    print(f"Kiểm tra lịch từ {start_date} -> {end_date}")
    print(f"Tổng số ngày cần có: {len(expected_dates)} | Khớp thực tế: {len(expected_dates.intersection(existing_dates))}")
    print("-" * 50)
    
    if len(missing_dates) == 0:
        print("Dữ liệu đầy đủ 100%")
    else:
        print(f"⚠️ Phát hiện THIẾU {len(missing_dates)} ngày sau đây:")
        for i, d in enumerate(missing_dates):
            print(f"  {i+1}. {d}")
    print("-" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_CSV = os.path.join(BASE_DIR, '../ema_daily_demand.csv')

    check_missing_dates(
        csv_path=FILE_CSV, 
        start_date="2023-01-02", 
        end_date="2025-12-31"
    )