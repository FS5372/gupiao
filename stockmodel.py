import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import gc
import os
import traceback
from utils import logger
import time
import concurrent.futures
from functools import partial
from datetime import datetime
from stock_utils import get_stock_history
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import joblib
import torch
import torch.nn as nn

def prepare_features_for_ticker(ticker, stock_data):
    """为单个股票准备特征（用于多线程）"""
    logger.info(f"\n--- 处理股票 {ticker} ---")
    ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
    features_list = []
    
    try:
        # 计算技术指标
        returns = ticker_data['Close'].pct_change()
        delta = ticker_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        ma5 = ticker_data['Close'].rolling(window=5).mean()
        ma20 = ticker_data['Close'].rolling(window=20).mean()
        ma_signal = (ma5 > ma20).astype(int) * 2 - 1
        next_day_return = returns.shift(-1)
        target = (next_day_return > 0).astype(int)
        
        # 组合特征
        for i in range(20, len(ticker_data)):
            features_list.append({
                'Ticker': ticker,
                'Date': ticker_data.index[i],
                'Returns': returns.iloc[i],
                'RSI': rsi.iloc[i],
                'MA_Signal': ma_signal.iloc[i],
                'Target': target.iloc[i],
                'Price': ticker_data['Close'].iloc[i],
                'Volume': ticker_data['Volume'].iloc[i],
                'PE': ticker_data['PE'].iloc[i],
                'PB': ticker_data['PB'].iloc[i]
            })
            
        return features_list
    except Exception as e:
        logger.error(f"处理股票 {ticker} 时出错: {str(e)}")
        return []

class StockTradingEnv(gym.Env):
    """股票交易强化学习环境"""
    
    def __init__(self, stock_data, rf_models):
        super(StockTradingEnv, self).__init__()
        
        # 数据预处理
        self.stock_data = {}
        for stock, data in stock_data.items():
            # 确保所有数值列都是有效的
            numeric_data = data.select_dtypes(include=[np.number])
            data[numeric_data.columns] = numeric_data.ffill().fillna(0)  # 使用 ffill() 替代 fillna(method='ffill')
            self.stock_data[stock] = data
            
        self.rf_models = rf_models
        self.stocks = list(self.stock_data.keys())
        self.n_stocks = len(self.stocks)
        
        # 标准化器
        self.state_normalizer = StandardScaler()
        
        # 定义动作空间（每只股票的权重）
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(self.n_stocks,),
            dtype=np.float32
        )
        
        # 定义观察空间
        self.n_features = 6
        self.observation_space = spaces.Box(
            low=-10,  # 使用有限范围而不是无穷
            high=10,
            shape=(self.n_stocks * self.n_features,),
            dtype=np.float32
        )
        
        # 初始化标准化器
        self._initialize_normalizer()
    
    def _initialize_normalizer(self):
        """初始化状态标准化器"""
        try:
            all_states = []
            for stock, data in self.stock_data.items():
                for i in range(min(20, len(data))):  # 使用前20天数据来初始化
                    state = self._get_single_stock_state(data, i)
                    if state is not None:
                        all_states.append(state)
            
            if all_states:
                all_states = np.vstack(all_states)
                self.state_normalizer.fit(all_states)
        except Exception as e:
            logger.warning(f"初始化标准化器失败: {str(e)}")
    
    def _get_single_stock_state(self, data, step):
        """获取单个股票的状态"""
        try:
            if step >= len(data):
                return np.zeros(self.n_features)
            
            current_data = data.iloc[step]
            
            # 安全地获取特征
            price = float(current_data.get('close', 0))
            volume = float(current_data.get('volume', 0))
            rsi = float(current_data.get('RSI', 50))
            macd = float(current_data.get('MACD', 0))
            
            # 计算布林带位置
            bb_position = self._safe_bb_position(current_data)
            
            # 获取预测概率
            rf_prob = self._safe_rf_prediction(data, step)
            
            state = np.array([price, volume, rsi, macd, bb_position, rf_prob])
            
            # 替换无效值
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state
            
        except Exception as e:
            logger.warning(f"获取单个股票状态失败: {str(e)}")
            return np.zeros(self.n_features)
    
    def _safe_bb_position(self, data):
        """安全地计算布林带位置"""
        try:
            price = float(data.get('close', 0))
            bb_lower = float(data.get('BB_lower', price * 0.95))
            bb_upper = float(data.get('BB_upper', price * 1.05))
            bb_width = bb_upper - bb_lower
            
            if bb_width > 0:
                return (price - bb_lower) / bb_width
            return 0.5
        except:
            return 0.5
    
    def _safe_rf_prediction(self, data, step):
        """安全地获取随机森林预测"""
        try:
            features = self._prepare_prediction_features(data, step)
            if features is not None and len(features) > 0:
                return float(self.rf_models.predict_proba(features)[0][1])
            return 0.5
        except:
            return 0.5
    
    def reset(self, *, seed=None, options=None):
        """重置环境"""
        try:
            # 设置随机种子
            if seed is not None:
                super().reset(seed=seed)
            
            # 重置环境状态
            self.current_step = 0
            self.total_steps = 20  # 设置回测期长度
            self.balance = 100000  # 初始资金
            self.total_value = self.balance
            self.positions = {stock: 0 for stock in self.stocks}  # 持仓数量
            
            # 获取初始观察
            observation = self._get_state()
            
            # 返回初始观察和信息字典
            return observation, {}
            
        except Exception as e:
            logger.error(f"重置环境失败: {str(e)}")
            return np.zeros(self.n_stocks * self.n_features, dtype=np.float32), {}
    
    def _get_state(self):
        """获取当前状态"""
        try:
            state = []
            for stock in self.stocks:
                stock_state = self._get_single_stock_state(
                    self.stock_data[stock], 
                    self.current_step
                )
                state.extend(stock_state)
            
            state = np.array(state, dtype=np.float32)
            
            # 标准化状态
            normalized_state = self.state_normalizer.transform(
                state.reshape(1, -1)
            ).flatten()
            
            # 确保没有无效值
            normalized_state = np.nan_to_num(
                normalized_state, 
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            )
            
            # 裁剪到有效范围
            normalized_state = np.clip(
                normalized_state,
                self.observation_space.low[0],
                self.observation_space.high[0]
            )
            
            return normalized_state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"获取状态失败: {str(e)}")
            return np.zeros(self.n_stocks * self.n_features, dtype=np.float32)
    
    def step(self, action):
        """执行一步交易"""
        try:
            # 确保动作合法（归一化）
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-10)
            
            # 记录当前总资产
            prev_total_value = self.total_value
            
            # 执行交易
            for i, stock in enumerate(self.stocks):
                try:
                    data = self.stock_data[stock]
                    if len(data) <= self.current_step:
                        continue
                    
                    current_price = float(data.iloc[self.current_step]['close'])
                    target_value = self.total_value * action[i]
                    current_value = self.positions[stock] * current_price
                    
                    # 计算需要买卖的股数
                    if target_value > current_value:  # 需要买入
                        shares_to_buy = int((target_value - current_value) / current_price)
                        cost = shares_to_buy * current_price
                        if cost <= self.balance:
                            self.positions[stock] += shares_to_buy
                            self.balance -= cost
                    else:  # 需要卖出
                        shares_to_sell = int((current_value - target_value) / current_price)
                        self.positions[stock] -= shares_to_sell
                        self.balance += shares_to_sell * current_price
                        
                except Exception as e:
                    logger.warning(f"执行交易时出错 {stock}: {str(e)}")
                    continue
            
            # 更新当前步骤
            self.current_step += 1
            
            # 计算新的总资产
            self.total_value = self.balance
            for stock, shares in self.positions.items():
                try:
                    if len(self.stock_data[stock]) > self.current_step:
                        price = float(self.stock_data[stock].iloc[self.current_step]['close'])
                        self.total_value += shares * price
                except Exception as e:
                    logger.warning(f"计算总资产时出错 {stock}: {str(e)}")
            
            # 计算奖励（收益率）
            reward = (self.total_value - prev_total_value) / prev_total_value
            
            # 判断是否结束
            terminated = self.current_step >= min(self.total_steps, 
                min(len(data) for data in self.stock_data.values()) - 1)
            truncated = False
            
            # 获取新的观察
            observation = self._get_state()
            
            # 准备信息字典
            info = {
                'total_value': self.total_value,
                'balance': self.balance,
                'positions': self.positions.copy(),
                'reward': reward,
                'step': self.current_step
            }
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"执行步骤失败: {str(e)}")
            return (
                np.zeros(self.n_stocks * self.n_features, dtype=np.float32),
                0.0,
                True,
                False,
                {'error': str(e)}
            )

def cleanup_models():
    """清理模型和临时文件"""
    try:
        if os.path.exists("best_model.zip"):
            os.remove("best_model.zip")
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"清理模型失败: {str(e)}")

class StockModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.initialize_model()

    def initialize_model(self):
        """初始化模型和标准化器"""
        try:
            logger.info("开始初始化模型...")
            # 初始化标准化器
            self.scaler = StandardScaler()
            
            # 训练随机森林模型
            self.rf_model = self._train_random_forest()
            logger.info("随机森林模型训练完成")
            
            # 创建神经网络模型
            self.model = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
            # 使用随机森林的预测结果进行强化学习
            self._reinforcement_learning()
            logger.info("强化学习训练完成")
            
        except Exception as e:
            logger.error(f"初始化模型失败: {str(e)}")

    def _train_random_forest(self):
        """训练随机森林模型"""
        try:
            logger.info("开始训练随机森林模型...")
            
            # 获取训练数据
            train_data = self._get_training_data()
            if train_data is None:
                raise Exception("无法获取训练数据")
            
            X_train = train_data['features']
            y_train = train_data['labels']
            
            # 标准化特征
            X_train = self.scaler.fit_transform(X_train)
            
            # 创建并训练随机森林
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            rf_model.fit(X_train, y_train)
            logger.info(f"随机森林模型训练完成，准确率: {rf_model.score(X_train, y_train):.4f}")
            
            return rf_model
            
        except Exception as e:
            logger.error(f"随机森林训练失败: {str(e)}")
            return None

    def _get_training_data(self):
        """获取训练数据"""
        try:
            logger.info("开始获取训练数据...")
            
            # 获取股票列表（这里可以从配置文件或数据库中获取）
            stock_list = ['000001', '000002', '000063', '000066', '000333', '000651', '000858']  # 示例股票
            
            features = []
            labels = []
            
            for stock in stock_list:
                # 获取历史数据
                df = get_stock_history(stock, days=365)  # 获取一年的数据
                if df is not None:
                    # 计算特征
                    df['returns'] = df['close'].pct_change()
                    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
                    df['ma5'] = df['close'].rolling(5).mean()
                    df['ma20'] = df['close'].rolling(20).mean()
                    df['ma60'] = df['close'].rolling(60).mean()
                    
                    # 生成标签（未来5天的收益率）
                    df['future_returns'] = df['returns'].shift(-5)
                    df['label'] = (df['future_returns'] > 0).astype(int)
                    
                    # 删除含有NaN的行
                    df = df.dropna()
                    
                    # 提取特征和标签
                    for i in range(len(df)-5):
                        features.append([
                            df['close'].iloc[i],
                            df['ma5'].iloc[i],
                            df['ma20'].iloc[i],
                            df['ma60'].iloc[i],
                            df['volatility'].iloc[i],
                            df['returns'].iloc[i]
                        ])
                        labels.append(df['label'].iloc[i])
            
            if not features:
                raise Exception("没有获取到有效的训练数据")
            
            return {
                'features': np.array(features),
                'labels': np.array(labels)
            }
            
        except Exception as e:
            logger.error(f"获取训练数据失败: {str(e)}")
            return None

    def _reinforcement_learning(self, epochs=2000):
        """使用强化学习优化模型"""
        try:
            logger.info("开始强化学习训练...")
            
            # 获取训练数据
            train_data = self._get_training_data()
            if train_data is None or self.rf_model is None:
                raise Exception("无法进行强化学习训练：缺少必要数据或模型")
            
            X = train_data['features']
            X = self.scaler.transform(X)
            X = torch.FloatTensor(X)
            
            # 使用随机森林的预测作为初始策略
            rf_predictions = self.rf_model.predict_proba(self.scaler.transform(train_data['features']))
            y = torch.FloatTensor(rf_predictions[:, 1])
            
            # 定义损失函数和优化器，提高学习率
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)  # 提高学习率
            
            # 添加学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=100,
                verbose=True
            )
            
            # 训练循环
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # 前向传播
                outputs = self.model(X)
                loss = criterion(outputs.squeeze(), y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新学习率
                scheduler.step(loss)
                
                # 早停检查
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 200:  # 如果200个epoch没有改善就停止
                    logger.info(f"早停触发，在epoch {epoch+1}/{epochs}")
                    break
                
                if (epoch + 1) % 100 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            logger.info(f"强化学习训练完成，最终损失: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"强化学习训练失败: {str(e)}\n{traceback.format_exc()}")

    def get_stock_state(self, stock_code):
        """获取股票当前状态"""
        try:
            logger.info(f"开始获取股票 {stock_code} 的状态")
            
            # 获取股票数据
            df = get_stock_history(stock_code)
            if df is None or df.empty:
                logger.error(f"未能获取到股票 {stock_code} 的历史数据")
                return None
                
            logger.info(f"成功获取股票 {stock_code} 的历史数据，数据长度: {len(df)}")
            
            # 获取最新数据
            latest = df.iloc[-1]
            
            # 计算技术指标
            try:
                volatility = df['close'].pct_change().std() * 100  # 波动率
                returns = df['close'].pct_change()
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()  # 年化夏普比率
                
                # 计算最大回撤
                cummax = df['close'].cummax()
                drawdown = (df['close'] - cummax) / cummax
                max_drawdown = abs(drawdown.min() * 100)
                
                state = {
                    'close': float(latest['close']),
                    'ma5': float(df['ma5'].iloc[-1]),
                    'ma10': float(df['close'].rolling(10).mean().iloc[-1]),
                    'ma20': float(df['ma20'].iloc[-1]),
                    'ma60': float(df['ma60'].iloc[-1]),
                    'volume': float(latest['volume']),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown)
                }
                
                logger.info(f"股票 {stock_code} 状态计算完成: {state}")
                return state
                
            except Exception as e:
                logger.error(f"计算股票 {stock_code} 技术指标时出错: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 状态失败: {str(e)}\n{traceback.format_exc()}")
            return None

    def analyze(self, stocks, investment_amount):
        """分析股票并生成投资建议"""
        try:
            # 获取每只股票的状态和历史数据
            stock_states = {}
            historical_data = {}
            
            for stock_code in stocks:
                # 获取历史数据
                df = get_stock_history(stock_code)
                if df is not None:
                    historical_data[stock_code] = {
                        'dates': df.index.strftime('%Y-%m-%d').tolist(),
                        'klineData': df[['open', 'high', 'low', 'close']].values.tolist(),
                        'ma5': df['ma5'].tolist(),
                        'ma20': df['ma20'].tolist(),
                        'ma60': df['ma60'].tolist()
                    }
                    state = self.get_stock_state(stock_code)
                    if state is not None:
                        stock_states[stock_code] = state

            if not stock_states:
                return {
                    "success": False,
                    "message": "没有获取到有效的股票数据",
                    "data": None
                }

            # 预测每只股票的权重
            weights = self.predict_weights(list(stock_states.values()))
            
            # 确保权重和为1且非负
            weights = np.maximum(weights, 0)  # 确保非负
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)  # 归一化
            
            # 生成每只股票的投资建议
            results = {}
            for stock_code, state in stock_states.items():
                suggestion = self.generate_stock_suggestion(stock_code, investment_amount, state)
                if suggestion:
                    suggestion['historical_data'] = historical_data.get(stock_code)  # 添加历史数据
                    results[stock_code] = suggestion
            
            # 添加组合分析
            portfolio_analysis = {
                "total_investment": sum(r['basic_info']['suggested_amount'] for r in results.values()),
                "avg_risk": np.mean([r['risk_analysis']['volatility'] for r in results.values()]),
                "stock_count": len(results),
                "diversification": len(results) / len(stocks) * 100
            }
            
            # 处理历史数据中的 NaN
            if 'historical_data' in results:
                for stock_code, stock_data in results['data'].items():
                    if 'historical_data' in stock_data:
                        stock_data['historical_data']['ma5'] = self._process_ma_data(stock_data['historical_data']['ma5'])
                        stock_data['historical_data']['ma20'] = self._process_ma_data(stock_data['historical_data']['ma20'])
                        stock_data['historical_data']['ma60'] = self._process_ma_data(stock_data['historical_data']['ma60'])
            
            return {
                "success": True,
                "message": "分析完成",
                "data": results,
                "portfolio": portfolio_analysis
            }
            
        except Exception as e:
            logger.error(f"分析失败: {str(e)}\n{traceback.format_exc()}")
            raise

    def generate_stock_suggestion(self, stock_code, amount, state):
        """生成单只股票的投资建议"""
        try:
            logger.info(f"开始生成股票 {stock_code} 的投资建议")
            
            current_price = state['close']
            trend_type = self._get_trend_type(state)
            
            # 计算技术指标
            ma5, ma20, ma60 = state['ma5'], state['ma20'], state['ma60']
            price_to_ma20 = ((current_price - ma20) / ma20) * 100 if ma20 > 0 else 0
            
            # 根据趋势和技术分析调整投资金额
            adjusted_amount = amount
            if trend_type == "强势下跌":
                adjusted_amount *= 0.3  # 强势下跌时只使用30%资金
            elif trend_type == "下跌":
                adjusted_amount *= 0.5  # 下跌时使用50%资金
            elif trend_type == "震荡":
                adjusted_amount *= 0.7  # 震荡时使用70%资金
            
            num_shares = int(adjusted_amount / current_price) if adjusted_amount > 0 else 0
            
            # 设置信心指数
            if trend_type == "强势上涨":
                confidence = 85.0
                action = "买入"
            elif trend_type == "上涨":
                confidence = 75.0
                action = "买入"
            elif trend_type == "强势下跌":
                confidence = 25.0
                action = "谨慎" if adjusted_amount > 0 else "观望"
            elif trend_type == "下跌":
                confidence = 35.0
                action = "谨慎" if adjusted_amount > 0 else "观望"
            else:  # 震荡
                confidence = 50.0
                action = "谨慎" if adjusted_amount > 0 else "观望"

            # 计算止损止盈价位
            volatility = state.get('volatility', 30)
            stop_loss = current_price * (1 - volatility * 0.05)
            take_profit = current_price * (1 + volatility * 0.15)
            
            # 生成建议
            suggestion = {
                "basic_info": {
                    "code": stock_code,
                    "current_price": current_price,
                    "suggested_amount": adjusted_amount,
                    "suggested_shares": num_shares
                },
                "technical_analysis": {
                    "trend": trend_type,
                    "ma5": ma5,
                    "ma10": state.get('ma10', 0),
                    "ma20": ma20,
                    "ma60": ma60,
                    "price_to_ma20": price_to_ma20
                },
                "risk_analysis": {
                    "risk_level": "高风险" if volatility > 40 else "中风险" if volatility > 20 else "低风险",
                    "volatility": volatility,
                    "max_drawdown": state.get('max_drawdown', 0),
                    "sharpe_ratio": state.get('sharpe_ratio', 0)
                },
                "trading_suggestions": {
                    "action": action,
                    "confidence": confidence,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
            }
            
            logger.info(f"股票 {stock_code} 的投资建议生成完成: {suggestion}")
            return suggestion
            
        except Exception as e:
            logger.error(f"生成股票 {stock_code} 建议时出错: {str(e)}\n{traceback.format_exc()}")
            return None

    def predict_weights(self, states):
        """预测投资权重"""
        try:
            logger.info(f"开始预测权重，输入状态数量: {len(states)}")
            
            # 准备特征
            features = np.array([self._extract_features(state) for state in states])
            features = self.scaler.transform(features)
            features = torch.FloatTensor(features)
            
            # 使用模型预测
            with torch.no_grad():
                scores = self.model(features).numpy()
                # 确保scores是一维数组
                scores = scores.reshape(-1)
            
            # 调整得分
            adjusted_scores = []
            for i, state in enumerate(states):
                trend_type = self._get_trend_type(state)
                score = scores[i]
                adjusted_score = score
                
                # 根据趋势类型调整得分
                if trend_type == "强势上涨":
                    adjusted_score *= 1.2
                elif trend_type == "上涨":
                    adjusted_score *= 1.1
                elif trend_type == "下跌":
                    adjusted_score *= 0.8
                elif trend_type == "强势下跌":
                    adjusted_score *= 0.6
                
                adjusted_scores.append(adjusted_score)
                logger.info(f"股票得分 - 原始: {score:.4f}, 趋势: {trend_type}, 调整后: {adjusted_score:.4f}")
            
            weights = np.array(adjusted_scores)
            
            # 确保权重非负且和为1
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(states)) / len(states)
            
            logger.info(f"最终权重分配: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"预测权重时出错: {str(e)}\n{traceback.format_exc()}")
            return np.ones(len(states)) / len(states)

    def _calculate_technical_score(self, state):
        """计算技术分析得分"""
        try:
            price = state['close']
            ma5 = state['ma5']
            ma20 = state['ma20']
            ma60 = state['ma60']
            volatility = state.get('volatility', 30)
            sharpe = state.get('sharpe_ratio', 0)
            
            # 趋势得分
            trend_score = 0
            if price > ma5 > ma20 > ma60:
                trend_score = 1.0
            elif price > ma5 and ma5 > ma20:
                trend_score = 0.7
            elif price < ma5 < ma20 < ma60:
                trend_score = 0.0
            elif price < ma5 and ma5 < ma20:
                trend_score = 0.3
            else:
                trend_score = 0.5
            
            # 风险调整
            risk_score = max(0, 1 - (volatility / 100))
            
            # 夏普比率调整
            sharpe_score = (sharpe + 2) / 4  # 将夏普比率映射到0-1区间
            
            # 综合得分
            final_score = (trend_score * 0.5 + risk_score * 0.3 + sharpe_score * 0.2)
            
            logger.info(f"技术分析得分明细 - 趋势: {trend_score:.2f}, 风险: {risk_score:.2f}, 夏普: {sharpe_score:.2f}, 最终: {final_score:.2f}")
            return final_score
            
        except Exception as e:
            logger.error(f"计算技术分析得分时出错: {str(e)}\n{traceback.format_exc()}")
            return 0.5

    def _extract_features(self, state):
        """从状态中提取特征"""
        return np.array([
            state['close'],
            state['ma5'],
            state['ma20'],
            state['ma60'],
            state.get('volatility', 0),
            state.get('sharpe_ratio', 0)
        ])

    def _get_trend_type(self, state):
        """判断趋势类型"""
        try:
            price = state['close']
            ma5 = state['ma5']
            ma20 = state['ma20']
            ma60 = state['ma60']
            
            if price > ma5 > ma20 > ma60:
                return "强势上涨"
            elif price > ma5 and ma5 > ma20:
                return "上涨"
            elif price < ma5 < ma20 < ma60:
                return "强势下跌"
            elif price < ma5 and ma5 < ma20:
                return "下跌"
            else:
                return "震荡"
        except Exception as e:
            logger.error(f"判断趋势类型时出错: {str(e)}")
            return "震荡"

    def cleanup(self):
        """清理模型和相关资源"""
        try:
            logger.info("开始清理模型资源...")
            if self.model is not None:
                del self.model
                self.model = None
            if self.scaler is not None:
                del self.scaler
                self.scaler = None
            # 强制进行垃圾回收
            gc.collect()
            logger.info("模型资源清理完成")
        except Exception as e:
            logger.error(f"清理模型资源时出错: {str(e)}")

    def _process_ma_data(self, data):
        """处理移动平均线数据，将 NaN 转换为 None"""
        if isinstance(data, (list, np.ndarray)):
            return [None if (isinstance(x, float) and np.isnan(x)) else x for x in data]
        return data