#步骤1，导入数据（已确认）
import yfinance as yf
import pandas as pd
import numpy as np

# 假设我们有中国大陆股票的代码列表
stocks = [
    '600000.SS', '600004.SS', '600009.SS', '600010.SS', '600011.SS', '600015.SS', '600016.SS', '600018.SS', '600019.SS', '600021.SS', 
    '600022.SS', '600027.SS', '600028.SS', '600029.SS', '600030.SS', '600031.SS', '600036.SS', '600037.SS', '600038.SS', '600048.SS', 
    '600050.SS', '600053.SS', '600054.SS', '600056.SS', '600058.SS', '600059.SS', '600060.SS', '600061.SS', '600062.SS', '600063.SS', 
    '600066.SS', '600067.SS', '600068.SS', '600070.SS', '600072.SS', '600073.SS', '600075.SS', '600077.SS', '600078.SS', '600079.SS',
    '600085.SS', '600086.SS', '600088.SS', '600089.SS', '600100.SS', '600104.SS', '600105.SS', '600106.SS', '600107.SS', '600108.SS'
]

# 获取所有股票的日线数据
def fetch_all_stock_data(stocks, start_date, end_date):
    all_data = pd.DataFrame()
    
    for stock in stocks:
        try:
            stock_data = yf.Ticker(stock).history(start=start_date, end=end_date)
            stock_data['Ticker'] = stock
            
            # 调用插值补缺函数
            stock_data = fill_missing_data(stock_data)
            
            all_data = pd.concat([all_data, stock_data], ignore_index=True)
            print(f"获取 {stock} 的数据...")
        except Exception as e:
            print(f"无法获取 {stock} 的数据，原因: {e}")
    
    return all_data

# 填充缺失值（NaN）的方法：使用线性插值
def fill_missing_data(df):
    # 选择数值类型的列进行插值
    num_df = df.select_dtypes(include=[np.number])  # 选择所有数值型列
    num_df = num_df.interpolate(method='linear', limit_direction='forward', axis=0)  # 进行插值
    
    # 将插值后的数值列赋回原始DataFrame
    df[num_df.columns] = num_df
    
    # 如果插值后仍然有NaN（例如前后的数据都缺失），使用前向填充
    df[num_df.columns] = df[num_df.columns].ffill()  # 使用前向填充
    # 如果仍然有NaN（比如全部缺失的列），使用后向填充
    df[num_df.columns] = df[num_df.columns].bfill()  # 使用后向填充
    
    return df

# 获取数据
start_date = '2019-01-01'
end_date = '2024-12-27'
all_stocks_data = fetch_all_stock_data(stocks, start_date, end_date)

# 保存数据到CSV
all_stocks_data.to_csv('2024_stocks_data.csv', index=False)
print("所有股票数据已保存到 2024_stocks_data.csv")


#步骤2（已确认，随机森林方法）
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# 计算RSI指标
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 计算布林带
def compute_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()  # 计算20日简单移动平均
    rolling_std = data.rolling(window=window).std()  # 计算标准差
    upper_band = sma + (rolling_std * num_std)  # 上轨
    lower_band = sma - (rolling_std * num_std)  # 下轨
    return upper_band, lower_band

# 随机森林模型预测股票走势
def random_forest_predict(stock_data):
    df = stock_data.copy()  # 避免直接修改原数据
    
    # 检查缺失值，打印每列的NaN数量
    print(f"股票 {df['Ticker'].iloc[0]} 数据缺失值情况：")
    print(df.isna().sum())  # 打印每列的NaN数量
    
    # 生成目标列，判断明天是否上涨
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1表示上涨，0表示下跌
    
    # 特征工程，使用过去的几天收盘价作为特征
    df['SMA_5'] = df['Close'].rolling(window=5).mean()  # 5日均线
    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20日均线
    df['RSI'] = compute_rsi(df['Close'], 14)  # 相对强弱指标
    
    # 计算布林带
    df['Upper_Band'], df['Lower_Band'] = compute_bollinger_bands(df['Close'])
    
    # 新特征：使用 Open, High, Low, Close, Volume
    df['Open_Close'] = df['Open'] - df['Close']  # 开盘与收盘价差
    df['High_Low'] = df['High'] - df['Low']  # 最高与最低价差
    df['Close_Open'] = df['Close'] - df['Open']  # 收盘价与开盘价差
    df['Volume_Close'] = df['Volume'] / df['Close']  # 成交量与收盘价的比率
    
    # 使用 infer_objects 转换列类型，避免警告
    df = df.infer_objects()

    # 使用插值填充缺失值
    df = df.interpolate(method='linear', limit_direction='forward')

    # 删除不需要的列
    df = df.drop(columns=['Dividends', 'Stock Splits', 'Adj Close'])
    
    # 删除包含NaN值的行（如果插值后还有NaN值）
    df = df.dropna()
    
    # 如果数据为空，跳过该股票
    if df.empty:
        print(f"股票 {df['Ticker'].iloc[0]} 删除缺失值后数据为空，跳过该股票")
        return None, None
    
    # 确保数据不为空
    if len(df) < 2:  # 数据不足无法划分训练集
        print("数据量不足，无法训练模型")
        return None, None
    
    # 准备数据集，包含所有生成的特征
    X = df[['Open_Close', 'High_Low', 'Close_Open', 'Volume_Close', 'SMA_5', 'SMA_20', 'RSI', 'Upper_Band', 'Lower_Band']]
    y = df['Target']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测并评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"股票 {df['Ticker'].iloc[0]} 预测的准确率: {accuracy * 100:.2f}%")
    
    return model, df

# 读取数据文件
file_path = "E:/stock/2024_stocks_data.csv"
all_stocks_data = pd.read_csv(file_path)

# 检查数据结构，确保有 'Ticker', 'Close' 等必要字段
print(all_stocks_data.head())

# 保存每支股票的模型
stock_models = {}

# 使用随机森林预测每支股票走势
for stock in all_stocks_data['Ticker'].unique():  # 遍历每只股票
    stock_data = all_stocks_data[all_stocks_data['Ticker'] == stock].copy()
    
    # 检查数据是否足够进行训练
    if stock_data.shape[0] < 20:  # 保证至少有足够多的样本
        print(f"股票 {stock} 数据量不足，跳过该股票")
        continue
    
    # 训练并获取模型和特征数据
    model, features_df = random_forest_predict(stock_data)
    
    if model is None:
        continue
    
    # 将当前股票的特征和预测结果保存到字典中
    if features_df is not None:
        stock_models[stock] = model  # 将模型存入字典，键是股票代码
        all_stock_features.append(features_df[['Ticker', 'Close', 'SMA_5', 'SMA_20', 'RSI', 'Upper_Band', 'Lower_Band', 'Open_Close', 'High_Low', 'Close_Open', 'Volume_Close', 'Target']])

# 合并所有股票的结果
features_df = pd.concat(all_stock_features)

# 存储为 CSV 文件
features_df.to_csv('E:/stock/2024_stocks_features_and_predictions.csv', index=False)

print("特征数据已保存到 E:/stock/2024_stocks_features_and_predictions.csv")


# 保存模型字典到文件
with open('E:/stock/stock_models.pkl', 'wb') as f:
    pickle.dump(stock_models, f)

print("所有股票的模型已保存到 E:/stock/stock_models.pkl")


#步骤3，RSI策略+均线交叉策略（已确认）
import pandas as pd
import numpy as np

# 计算RSI（相对强弱指数）
def calculate_rsi(stock_data, window=14):
    delta = stock_data['Close'].diff()  # 计算每日收盘价变化
    gain = delta.where(delta > 0, 0)    # 只取正的收益
    loss = -delta.where(delta < 0, 0)   # 只取负的亏损

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))         # 计算RSI
    return rsi

# 简单的均线交叉策略，结合RSI
def moving_average_crossover_with_rsi(stock_data, short_window=5, long_window=20, rsi_window=14):
    # 计算短期和长期均线
    stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window).mean()
    stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window).mean()

    # 计算RSI
    stock_data['RSI'] = calculate_rsi(stock_data, window=rsi_window)

    # 去除缺失值
    stock_data = stock_data.dropna(subset=['Short_MA', 'Long_MA', 'RSI'])

    # 买入信号：短期均线上穿长期均线，且RSI低于30
    stock_data['Buy_signal'] = (stock_data['Short_MA'] > stock_data['Long_MA']) & (stock_data['RSI'] < 30)

    # 卖出信号：短期均线下穿长期均线，且RSI高于70
    stock_data['Sell_signal'] = (stock_data['Short_MA'] < stock_data['Long_MA']) & (stock_data['RSI'] > 70)

    return stock_data

# 从文件中读取股票数据
file_path = "E:/stock/2024_stocks_data.csv"
all_stocks_data = pd.read_csv(file_path)

# 确保数据包含 'Ticker', 'Close' 等必要字段
print(all_stocks_data.head())

# 创建一个空的 DataFrame 来保存所有股票的买卖信号
all_signals = pd.DataFrame()

# 对每只股票应用均线交叉策略和RSI策略
for ticker in all_stocks_data['Ticker'].unique():
    stock_data = all_stocks_data[all_stocks_data['Ticker'] == ticker].copy(deep=True)
    stock_data = moving_average_crossover_with_rsi(stock_data)
    
    # 保存每只股票的结果到新的 DataFrame 中
    all_signals = pd.concat([all_signals, stock_data])

# 输出最后几行结果，查看每只股票的买卖信号
print(all_signals[['Ticker', 'Close', 'Short_MA', 'Long_MA', 'RSI', 'Buy_signal', 'Sell_signal']].tail())

# 保存到新的 CSV 文件
output_file_path = "E:/stock/2024_stocks_signals.csv"
all_signals.to_csv(output_file_path, index=False)

print(f"所有股票的买卖信号已保存到: {output_file_path}")


import gymnasium as gym  # 使用 gymnasium 替代 gym
import numpy as np
import pandas as pd
import pickle
from stable_baselines3 import PPO  # 使用 PPO 算法替代 A2C
from gymnasium import spaces

# 加载保存的股票模型
with open('E:/stock/stock_models.pkl', 'rb') as f:
    stock_models = pickle.load(f)

class StockTradingEnv(gym.Env):
    def __init__(self, stock_signal_data, stock_predict_data, models, initial_balance=100000):
        super(StockTradingEnv, self).__init__()
        self.stock_signal_data = stock_signal_data.copy()
        self.stock_predict_data = stock_predict_data.copy()
        self.models = models
        self.initial_balance = initial_balance
        self.stocks = self.stock_signal_data['Ticker'].unique()
        
        # 动作空间：每只股票的资金比例
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(len(self.stocks),), 
            dtype=np.float32
        )
        
        # 状态空间：每只股票4个特征
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.stocks) * 4,), 
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {stock: 0 for stock in self.stocks}
        self.total_value = self.balance
        return self._get_state(), {}  # 返回初始状态和空字典

    def _get_state(self):
        state = []
        for stock in self.stocks:
            try:
                stock_signal = self.stock_signal_data[
                    self.stock_signal_data['Ticker'] == stock
                ].iloc[-1]
                
                stock_predict = self.stock_predict_data[
                    self.stock_predict_data['Ticker'] == stock
                ].iloc[-1]
                
                price = stock_signal['Close']
                buy_signal = float(stock_signal['Buy_signal'])
                sell_signal = float(stock_signal['Sell_signal'])
                
                features = stock_predict[[
                    'Open_Close', 'High_Low', 'Close_Open', 'Volume_Close', 
                    'SMA_5', 'SMA_20', 'RSI', 'Upper_Band', 'Lower_Band'
                ]].values.reshape(1, -1)
                
                prediction = float(self.models[stock].predict(features)[0])
                state.extend([price, buy_signal, sell_signal, prediction])
                
            except Exception as e:
                state.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-10)
        
        prev_total_value = self.total_value
        
        # 执行交易
        for i, stock in enumerate(self.stocks):
            try:
                stock_data = self.stock_signal_data[
                    self.stock_signal_data['Ticker'] == stock
                ].iloc[-1]
                
                stock_price = stock_data['Close']
                target_amount = self.total_value * action[i]
                current_amount = self.positions[stock] * stock_price
                
                if target_amount > current_amount:
                    shares_to_buy = int((target_amount - current_amount) // stock_price)
                    cost = shares_to_buy * stock_price
                    if cost <= self.balance:
                        self.positions[stock] += shares_to_buy
                        self.balance -= cost
                
                elif target_amount < current_amount:
                    shares_to_sell = int((current_amount - target_amount) // stock_price)
                    self.positions[stock] -= shares_to_sell
                    self.balance += shares_to_sell * stock_price
                    
            except Exception as e:
                continue
        
        # 计算新的总资产
        portfolio_value = sum(
            self.positions[stock] * 
            self.stock_signal_data[self.stock_signal_data['Ticker'] == stock].iloc[-1]['Close']
            for stock in self.stocks
        )
        self.total_value = self.balance + portfolio_value
        
        # 计算奖励
        reward = (self.total_value - prev_total_value) / prev_total_value
        
        # 更新状态
        self.current_step += 1
        done = self.current_step >= 10  # 减少步数到10
        truncated = False
        
        return self._get_state(), reward, done, truncated, {}

def train_model(signal_file_path, predict_file_path):
    print("正在加载数据...")
    all_stock_signal_data = pd.read_csv(signal_file_path)
    all_stock_predict_data = pd.read_csv(predict_file_path)
    
    print("创建环境...")
    env = StockTradingEnv(all_stock_signal_data, all_stock_predict_data, stock_models)
    
    print("初始化模型...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    print("开始训练...")
    model.learn(total_timesteps=200)  # 减少训练步数到200
    
    return model, env

def get_buy_strategy(model, env, balance):
    state, _ = env.reset()
    action, _states = model.predict(state, deterministic=True)
    action = action / (action.sum() + 1e-10)
    
    buy_strategy = {}
    for i, stock in enumerate(env.stocks):
        try:
            stock_data = env.stock_signal_data[
                env.stock_signal_data['Ticker'] == stock
            ].iloc[-1]
            
            stock_price = stock_data['Close']
            amount_to_invest = balance * action[i]
            num_shares = int(amount_to_invest // stock_price)
            
            if num_shares > 0:
                buy_strategy[stock] = {
                    "amount_invested": amount_to_invest,
                    "num_shares": num_shares,
                    "price": stock_price
                }
        except Exception as e:
            continue
    
    return buy_strategy

if __name__ == "__main__":
    try:
        print("开始训练模型...")
        signal_file_path = "E:/stock/2024_stocks_signals.csv"
        predict_file_path = "E:/stock/2024_stocks_features_and_predictions.csv"
        
        model, env = train_model(signal_file_path, predict_file_path)
        print("模型训练完成！")
        
        initial_balance = float(input("\n请输入初始资金："))
        buy_strategy = get_buy_strategy(model, env, initial_balance)
        
        print("\n投资建议：")
        print("-" * 50)
        total_investment = 0
        
        for stock, strategy in buy_strategy.items():
            if strategy["amount_invested"] >= 100:  # 只显示投资金额大于100的建议
                print(f"股票代码: {stock}")
                print(f"建议投资金额: {strategy['amount_invested']:,.2f}")
                print(f"建议购买股数: {strategy['num_shares']}")
                print(f"当前股价: {strategy['price']:.2f}")
                print("-" * 50)
                total_investment += strategy['amount_invested']
        
        print(f"\n总投资金额: {total_investment:,.2f}")
        print(f"剩余现金: {initial_balance - total_investment:,.2f}")
        print(f"建议投资比例: {(total_investment/initial_balance)*100:.2f}%")
        
    except Exception as e:
        import traceback
        print(f"程序执行出错: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())