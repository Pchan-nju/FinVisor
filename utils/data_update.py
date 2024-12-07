import tushare as ts
import os  # 用于创建文件夹

def data_update(folder_path):
    # 设置你的 tushare token（可以去 tushare 官网注册并获取）
    ts.set_token('637be502ff3ef07449e718eba3432df68ee919e5fa1fd23febd2e316')

    # 初始化pro接口
    pro = ts.pro_api()

    # 股票代码列表
    stock_codes = [
        '000031.SZ', '000069.SZ', '000402.SZ', '000608.SZ', '001979.SZ',
        '002305.SZ', '600007.SH', '600239.SH', '600663.SH', '601155.SH'
    ]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 遍历股票代码并获取数据
    for stock_code in stock_codes:
        # 获取A股市场的股票数据
        data = pro.daily(ts_code=stock_code, start_date='20210101', end_date='20211231')

        # 打印每个股票的前几行数据，确认是否正确
        print(f"Fetching data for {stock_code}...")
        print(data.head())

        # 保存数据到CSV文件
        file_path = os.path.join(folder_path, f"{stock_code}_stock_data.csv")
        data.to_csv(file_path, index=False)
        print(f"Data for {stock_code} saved to {file_path}")

    print("All data has been successfully saved.")


if __name__ == "__main__":
    data_update()