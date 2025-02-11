import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def optimized_triple_ensemble_trading(
        df: pd.DataFrame,
        lookback_window: int = 30,
        predict_days: int = 1,
        test_size: float = 0.1,
        transaction_cost: float = 0.001,
        initial_capital: float = 1000000,
        random_seed: int = 42
):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # =================== Enhanced Feature Engineering ===================
    df = df.sort_values('date').reset_index(drop=True)
    df = _enhanced_feature_engineering(df)

    # Create labels: Determine if the price increases in the next day
    df['label'] = (df['close'].shift(-predict_days) > df['close']).astype(int)
    df = df.dropna()

    # Data splitting (ensuring time series continuity)
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    test_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    feature_cols = ['close', 'volume', 'rsi', 'macd', 'boll_%b', 'volatility',
                   'atr', 'cci', 'vwap', 'obv', 'adx', 'stoch']

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    # =================== Create time series samples ===================
    def create_sequences(data, labels):
        X, y = [], []
        for i in range(len(data) - lookback_window - predict_days + 1):
            X.append(data[i:i + lookback_window])
            y.append(labels[i + lookback_window])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, train_df['label'].values)
    X_val, y_val = create_sequences(val_scaled, val_df['label'].values)
    X_test, y_test = create_sequences(test_scaled, test_df['label'].values)

    # =================== Model Construction ===================
    # LSTM Model
    def build_enhanced_lstm():
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(lookback_window, len(feature_cols))),
            Dropout(0.4),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dense(32, activation='relu', kernel_regularizer='l2'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    # GRU Model
    def build_enhanced_gru():
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(lookback_window, len(feature_cols))),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dense(32, activation='relu', kernel_regularizer='l2'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    # Random Forest Model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced_subsample',
        random_state=random_seed,
        n_jobs=-1)

    # =================== Model Training ===================
    # Train deep learning models
    def train_enhanced_model(model, X, y):
        """Returns a trained model instance"""
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f'best_{model.name}.keras',  # Save best model
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )

        history = model.fit(
            X, y,
            epochs=150,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=15,
                              restore_best_weights=True),
                ReduceLROnPlateau(factor=0.3, patience=5),
                checkpoint
            ],
            verbose=0
        )
        return model

    # Train LSTM model
    print("Training LSTM...")
    lstm_model = build_enhanced_lstm()
    lstm_model = train_enhanced_model(lstm_model, X_train, y_train)

    # Train GRU model
    print("Training GRU...")
    gru_model = build_enhanced_gru()
    gru_model = train_enhanced_model(gru_model, X_train, y_train)

    # Train Random Forest (reshape input to 2D)
    print("Training Random Forest...")
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_val_rf = X_val.reshape(X_val.shape[0], -1)

    rf_model.fit(X_train_rf, y_train)

    try:
        _ = lstm_model.predict(X_train[:1])
        _ = gru_model.predict(X_train[:1])
        _ = rf_model.predict_proba(X_train_rf[:1])
    except Exception as e:
        raise RuntimeError(f"Model not initialized correctly: {str(e)}")

    global _global_models
    _global_models = {
        'lstm': lstm_model,
        'gru': gru_model,
        'rf': rf_model,
        'scaler': scaler
    }


    def ensemble_predict(X, return_components=False):
        lstm_preds = np.array([lstm_model.predict(X, verbose=0).flatten() for _ in range(5)])
        lstm_mean = np.mean(lstm_preds, axis=0)  # (samples,)

        gru_preds = gru_model.predict(X, verbose=0).flatten()  # (samples,)

        rf_probs = rf_model.predict_proba(X.reshape(X.shape[0], -1))[:, 1]  # (samples,)

        weights = np.array([0.35, 0.35, 0.3])
        final_prob = np.average([lstm_mean, gru_preds, rf_probs],
                                axis=0, weights=weights)

        if return_components:
            return {
                'final_prob': final_prob,
                'lstm': lstm_mean,
                'gru': gru_preds,
                'rf': rf_probs,
                'weights': weights
            }
        return final_prob

    def backtest_strategy(data_df, scaled_data, params):
        portfolio = {
            'date': [], 'close': [], 'cash': [],
            'shares': [], 'total': [], 'prediction': [],
            'actual': [], 'prob': [], 'daily_return': []
        }

        cash = initial_capital
        shares = 0
        prev_total = 0

        for i in range(lookback_window, len(scaled_data) - predict_days):
            current_data = scaled_data[i - lookback_window:i]
            current_price = data_df.iloc[i]['close']
            actual_label = data_df.iloc[i]['label']

            prob = ensemble_predict(current_data[np.newaxis, ...])[0]
            prediction = 1 if prob > params['buy_threshold'] else 0

            if prob > params['buy_threshold'] and cash > 1000:
                position_size = min(0.8 * cash, cash - 1000)  # 动态仓位管理
                shares_bought = max(1, int(position_size / current_price))
                cost = shares_bought * current_price * (1 + transaction_cost)
                if cash >= cost:
                    shares += shares_bought
                    cash -= cost

            elif prob < params['sell_threshold'] and shares > 0:
                sell_ratio = 0.9 if (prob < params['sell_threshold'] * 0.9) else 0.5  # 动态卖出比例
                sell_shares = max(1, int(shares * sell_ratio))
                cash += sell_shares * current_price * (1 - transaction_cost)
                shares -= sell_shares

            current_total = current_price * shares + cash
            if i > lookback_window:
                daily_return = (current_total - prev_total) / prev_total
                portfolio['daily_return'].append(daily_return)
            else:
                portfolio['daily_return'].append(0)

            prev_total = current_price * shares + cash

            portfolio['date'].append(data_df.iloc[i]['date'])
            portfolio['close'].append(current_price)
            portfolio['cash'].append(cash)
            portfolio['shares'].append(shares)
            portfolio['total'].append(current_total)
            portfolio['prediction'].append(prediction)
            portfolio['actual'].append(actual_label)
            portfolio['prob'].append(prob)

        return pd.DataFrame(portfolio)

    def calculate_metrics(df):
        if len(df) == 0:
            return {'Sharpe Ratio': -np.inf}

        total_return = df['total'].iloc[-1] / initial_capital - 1

        accuracy = accuracy_score(df['actual'], df['prediction'])

        returns = df['total'].pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()

        return {
            'Total Return': total_return,
            'Prediction Accuracy': accuracy,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': (df['total'] / df['total'].cummax() - 1).min()
        }

    def optimize_thresholds(buy_threshold, sell_threshold):
        params = {
            'buy_threshold': np.clip(buy_threshold, 0.45, 0.5),
            'sell_threshold': np.clip(sell_threshold, 0.35, 0.4)
        }

        portfolio = backtest_strategy(val_df, val_scaled, params)
        metrics = calculate_metrics(portfolio)
        return metrics['Sharpe Ratio']

    optimizer = BayesianOptimization(
        f=optimize_thresholds,
        pbounds={
            'buy_threshold': (0.45, 0.55),
            'sell_threshold': (0.3, 0.4)
        },
        random_state=random_seed
    )
    optimizer.maximize(init_points=2, n_iter=5)
    best_params = optimizer.max['params']
    best_params['buy_threshold'] = round(best_params['buy_threshold'], 3)
    best_params['sell_threshold'] = round(best_params['sell_threshold'], 3)

    final_backtest = backtest_strategy(test_df, test_scaled, best_params)

    def predict_tomorrow(latest_df):
        processed_data = _enhanced_feature_engineering(latest_df).iloc[-lookback_window:]
        scaled_data = scaler.transform(processed_data[feature_cols])

        lstm_pred = lstm_model.predict(scaled_data[np.newaxis, ...], verbose=0)[0][0]
        gru_pred = gru_model.predict(scaled_data[np.newaxis, ...], verbose=0)[0][0]
        rf_pred = rf_model.predict_proba(scaled_data.reshape(1, -1))[0][1]

        final_prob = 0.4*lstm_pred + 0.4*gru_pred + 0.2*rf_pred
        decision = "Buy" if final_prob > best_params['buy_threshold'] else \
            ("Sell" if final_prob < best_params['sell_threshold'] else "Hold")

        return {
            'Probability_Up': float(final_prob),
            'Recommendation': decision,
            'Buy_Threshold': best_params['buy_threshold'],
            'Sell_Threshold': best_params['sell_threshold']
        }

    tomorrow_prediction = predict_tomorrow(df)

    return {
        'backtest_data': final_backtest,
        'performance': calculate_metrics(final_backtest),
        'optimal_params': best_params,
        'tomorrow_prediction': tomorrow_prediction,
        'models': {
            'LSTM': lstm_model,
            'GRU': gru_model,
            'RandomForest': rf_model
        }
    }


def _enhanced_feature_engineering(df):

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(5).std()

    # 扩展技术指标
    df['rsi'] = _calculate_rsi(df['close'], 14)
    df['macd'], df['macd_signal'] = _calculate_macd(df['close'])
    df['boll_%b'] = _calculate_bollinger(df['close'], 20)
    df['atr'] = _calculate_atr(df, 14)
    df['cci'] = _calculate_cci(df['close'], 20)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv'] = _calculate_obv(df)
    df['adx'] = _calculate_adx(df, 14)
    df['stoch'] = _calculate_stochastic(df, 14)

    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['price_trend'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean()

    df['volatility_1d'] = df['close'].pct_change().abs()
    df['volatility_3d'] = df['close'].pct_change(3).abs()

    return df.dropna()


def _calculate_adx(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.DataFrame(index=df.index)
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    tr_roll = tr['tr'].rolling(window).sum()
    plus_di = 100 * (plus_dm.rolling(window).sum() / tr_roll)
    minus_di = 100 * (minus_dm.rolling(window).sum() / tr_roll)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window).mean()


def _calculate_stochastic(df, window=14):
    low_min = df['low'].rolling(window).min()
    high_max = df['high'].rolling(window).max()
    return 100 * (df['close'] - low_min) / (high_max - low_min)


def _calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line


def _calculate_bollinger(series, window):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - (sma - 2 * std)) / (4 * std)

def _calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def _calculate_cci(series, window=20):
    sma = series.rolling(window).mean()
    mad = (series - sma).abs().rolling(window).mean()
    return (series - sma) / (0.015 * mad)

def _calculate_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv
