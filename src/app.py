from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class CryptoPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.predictions = {}
        self.load_and_prepare_data()
        self.train_models()
    
    def load_and_prepare_data(self):
        """Cargar y preparar los datos"""
        try:
            # Intentar cargar desde diferentes ubicaciones
            possible_paths = [
                '../Data/new_data.csv',
                'Data/new_data.csv',
                './Data/new_data.csv',
                '../../../Data/new_data.csv'
            ]
            
            data_loaded = False
            for path in possible_paths:
                try:
                    self.data = pd.read_csv(path)
                    print(f"Datos cargados desde: {path}")
                    data_loaded = True
                    break
                except FileNotFoundError:
                    continue
            
            if not data_loaded:
                raise FileNotFoundError("No se encontró el archivo de datos")
            
            # Procesar timestamp
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], utc=True)
            self.data.set_index("timestamp", inplace=True)
            
            # Crear indicadores técnicos
            self.create_technical_indicators()
            
            # Crear características adicionales
            self.create_features()
            
            # Limpiar datos
            self.data.dropna(inplace=True)
            
            print(f"Datos procesados: {len(self.data)} registros")
            print(f"Rango de fechas: {self.data.index.min()} a {self.data.index.max()}")
            
        except Exception as e:
            print(f"Error cargando datos reales: {e}")
            print("Creando datos de ejemplo...")
            # Crear datos de ejemplo si no se pueden cargar
            self.create_sample_data()
    
    def create_technical_indicators(self):
        """Crear indicadores técnicos"""
        # RSI
        rsi = RSIIndicator(close=self.data['close'], window=14)
        self.data['rsi_14'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=self.data['close'], window_slow=26, window_fast=12, window_sign=9)
        self.data['macd'] = macd.macd()
        self.data['macd_signal'] = macd.macd_signal()
        self.data['macd_diff'] = macd.macd_diff()
        
        # Estocástico
        stoch = StochasticOscillator(high=self.data['high'], low=self.data['low'], 
                                   close=self.data['close'], window=14, smooth_window=3)
        self.data['stoch_k'] = stoch.stoch()
        self.data['stoch_d'] = stoch.stoch_signal()
    
    def create_features(self):
        """Crear características adicionales"""
        # Características temporales
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data["quarter"] = self.data.index.quarter
        
        # Características de precio y volumen
        self.data["price_range_pct"] = (self.data["high"] - self.data["low"]) / self.data["open"]
        self.data["volume_log_change"] = self.data["volume_log"].diff()
        self.data["volume_log_ma_7d"] = self.data["volume_log"].rolling(window=7).mean()
        self.data["marketCap_log_change"] = self.data["marketCap_log"].diff()
        self.data["liq_ratio"] = self.data["volume_log"] / self.data["marketCap_log"]
    
    def create_sample_data(self):
        """Crear datos de ejemplo si no se pueden cargar los reales"""
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Generar precios simulados
        n_days = len(dates)
        price_base = 100
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [price_base]
        
        for i in range(1, n_days):
            prices.append(prices[-1] * (1 + returns[i]))
        
        self.data = pd.DataFrame({
            'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'high': [p * np.random.uniform(1.01, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 0.99) for p in prices],
            'close': prices,
            'volume_log': np.random.uniform(10, 15, n_days),
            'marketCap_log': np.random.uniform(20, 25, n_days)
        }, index=dates)
        
        self.create_technical_indicators()
        self.create_features()
        self.data.dropna(inplace=True)
    
    def create_sliding_windows(self, features, target="close", window_size=7):
        """Crear ventanas deslizantes para el modelo"""
        X, y = [], []
        for i in range(window_size, len(self.data)):
            X.append(self.data[features].iloc[i - window_size:i].values)
            y.append(self.data[target].iloc[i])
        return np.array(X), np.array(y)
    
    def train_models(self):
        """Entrenar los modelos de predicción"""
        features = ['close', 'volume_log', 'marketCap_log', 'rsi_14',
                   'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 'year',
                   'month', 'day', 'day_of_week', 'quarter', 'price_range_pct',
                   'volume_log_change', 'volume_log_ma_7d', 'marketCap_log_change',
                   'liq_ratio']
        
        # Crear ventanas deslizantes
        X, y = self.create_sliding_windows(features, window_size=7)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=0.2, shuffle=False
        )
        
        # Entrenar modelos
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Random Forest
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Guardar modelos y predicciones
        self.models = {
            'linear_regression': lr_model,
            'random_forest': rf_model
        }
        
        self.predictions = {
            'y_test': y_test,
            'linear_regression': lr_pred,
            'random_forest': rf_pred,
            'test_dates': self.data.index[-len(y_test):]
        }
        
        # Calcular métricas
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calcular métricas de los modelos"""
        y_test = self.predictions['y_test']
        
        self.metrics = {}
        for model_name in ['linear_regression', 'random_forest']:
            pred = self.predictions[model_name]
            self.metrics[model_name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'mae': mean_absolute_error(y_test, pred),
                'r2': r2_score(y_test, pred)
            }
    
    def predict_future(self, model_name='random_forest', days=30):
        """Predecir precios futuros"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        features = ['close', 'volume_log', 'marketCap_log', 'rsi_14',
                   'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 'year',
                   'month', 'day', 'day_of_week', 'quarter', 'price_range_pct',
                   'volume_log_change', 'volume_log_ma_7d', 'marketCap_log_change',
                   'liq_ratio']
        
        # Última ventana de datos
        last_window = self.data[features].values[-7:]  # últimos 7 días
        predictions = []
        
        for _ in range(days):
            X_input = last_window.reshape(1, -1)
            pred = model.predict(X_input)[0]
            predictions.append(pred)
            
            # Actualizar ventana (simplificado)
            new_row = last_window[-1].copy()
            new_row[0] = pred  # Actualizar precio de cierre
            last_window = np.vstack([last_window[1:], new_row])
        
        # Crear fechas futuras
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        return {
            'dates': future_dates,
            'predictions': predictions
        }

# Instancia global del predictor
print("Inicializando CryptoPredictor...")
try:
    crypto_predictor = CryptoPredictor()
    print("CryptoPredictor inicializado exitosamente")
except Exception as e:
    print(f"Error inicializando CryptoPredictor: {e}")
    crypto_predictor = None

@app.route('/health')
def health_check():
    """Verificar estado de la aplicación"""
    try:
        status = {
            'status': 'healthy',
            'predictor_initialized': crypto_predictor is not None,
            'data_loaded': crypto_predictor is not None and crypto_predictor.data is not None,
            'models_trained': crypto_predictor is not None and len(crypto_predictor.models) > 0
        }
        
        if crypto_predictor is not None:
            status['data_points'] = len(crypto_predictor.data)
            status['models_available'] = list(crypto_predictor.models.keys())
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/historical_data')
def get_historical_data():
    """API para obtener datos históricos"""
    try:
        if crypto_predictor is None:
            return jsonify({'error': 'Predictor no inicializado'}), 500
            
        data = crypto_predictor.data.copy()
        
        # Preparar datos para el gráfico
        result = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'prices': data['close'].tolist(),
            'volume': data['volume_log'].tolist(),
            'rsi': data['rsi_14'].fillna(50).tolist(),
            'macd': data['macd'].fillna(0).tolist()
        }
        
        print(f"Enviando {len(result['dates'])} registros históricos")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error en get_historical_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_predictions')
def get_model_predictions():
    """API para obtener predicciones del modelo"""
    try:
        preds = crypto_predictor.predictions
        
        result = {
            'dates': preds['test_dates'].strftime('%Y-%m-%d').tolist(),
            'actual': preds['y_test'].tolist(),
            'linear_regression': preds['linear_regression'].tolist(),
            'random_forest': preds['random_forest'].tolist(),
            'metrics': crypto_predictor.metrics
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/future_predictions/<model_name>/<int:days>')
def get_future_predictions(model_name, days):
    """API para obtener predicciones futuras"""
    try:
        future_data = crypto_predictor.predict_future(model_name, days)
        
        if future_data is None:
            return jsonify({'error': 'Modelo no encontrado'}), 404
        
        result = {
            'dates': future_data['dates'].strftime('%Y-%m-%d').tolist(),
            'predictions': future_data['predictions']
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart_data')
def get_chart_data():
    """API para obtener todos los datos para el gráfico principal"""
    try:
        if crypto_predictor is None:
            return jsonify({'error': 'Predictor no inicializado'}), 500
            
        print("Obteniendo datos para gráfico...")
        
        # Datos históricos
        historical = crypto_predictor.data.copy()
        print(f"Datos históricos: {len(historical)} registros")
        
        # Predicciones del modelo
        preds = crypto_predictor.predictions
        print(f"Predicciones disponibles: {len(preds['y_test'])} registros")
        
        # Predicciones futuras
        future_rf = crypto_predictor.predict_future('random_forest', 30)
        print(f"Predicciones futuras: {len(future_rf['predictions'])} días")
        
        result = {
            'historical': {
                'dates': historical.index.strftime('%Y-%m-%d').tolist(),
                'close': historical['close'].tolist()
            },
            'predictions': {
                'dates': preds['test_dates'].strftime('%Y-%m-%d').tolist(),
                'actual': preds['y_test'].tolist(),
                'random_forest': preds['random_forest'].tolist(),
                'linear_regression': preds['linear_regression'].tolist()
            },
            'future': {
                'dates': future_rf['dates'].strftime('%Y-%m-%d').tolist(),
                'predictions': future_rf['predictions']
            },
            'metrics': crypto_predictor.metrics
        }
        
        print("Datos preparados exitosamente")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error en get_chart_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_info')
def get_system_info():
    """API para obtener información del sistema"""
    try:
        return jsonify({
            'data_points': len(crypto_predictor.data),
            'date_range': {
                'start': crypto_predictor.data.index.min().strftime('%Y-%m-%d'),
                'end': crypto_predictor.data.index.max().strftime('%Y-%m-%d')
            },
            'models_available': list(crypto_predictor.models.keys()),
            'features_count': len(['close', 'volume_log', 'marketCap_log', 'rsi_14',
                                 'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 'year',
                                 'month', 'day', 'day_of_week', 'quarter', 'price_range_pct',
                                 'volume_log_change', 'volume_log_ma_7d', 'marketCap_log_change',
                                 'liq_ratio']),
            'current_price': float(crypto_predictor.data['close'].iloc[-1]),
            'status': 'active'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
