from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class CryptoPredictor:
    def __init__(self):
        self.data = None
        # Paso 1: Modelos de regresión
        self.regression_models = {}
        self.regression_predictions = {}
        # Paso 2: Modelos de clasificación binaria
        self.classification_models = {}
        self.classification_predictions = {}
        # Paso 3: Modelos de probabilidad
        self.probability_models = {}
        self.probability_predictions = {}
        
        self.load_and_prepare_data()
        self.train_all_models()
    
    def create_sliding_window(self, data, target, days_past, day_future, features, binarize=False):
        """Crear ventanas deslizantes para diferentes tipos de predicción"""
        df = data.reset_index(drop=True)
        n = len(df)
        rows = []

        for end in range(days_past, n - day_future + 1):
            start = end - days_past
            future_idx = end + day_future - 1
            future_val = df.loc[future_idx, target]
            last_val = df.loc[end - 1, target]
            row = {}
            
            for i in range(days_past):
                day_idx = start + i
                suffix = f"t-{days_past - i}"
                for feat in features:
                    row[f"{feat}_{suffix}"] = df.loc[day_idx, feat]
            
            if binarize:
                row['target'] = int(future_val > last_val)
            else:
                row['target'] = future_val
            rows.append(row)
        
        return pd.DataFrame(rows)
    
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
    
    def train_all_models(self):
        """Entrenar modelos para los 3 pasos"""
        features = ['close', 'volume_log', 'marketCap_log', 'rsi_14',
                   'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 
                   'price_range_pct', 'volume_log_change', 'volume_log_ma_7d', 
                   'marketCap_log_change', 'liq_ratio']
        
        # Paso 1: Regresión directa del precio
        self.train_regression_models(features)
        
        # Paso 2: Clasificación binaria
        self.train_classification_models(features)
        
        # Paso 3: Probabilidad de incremento
        self.train_probability_models(features)
    
    def train_regression_models(self, features):
        """Paso 1: Entrenar modelos de regresión"""
        # Crear ventanas deslizantes para regresión
        X, y = self.create_sliding_windows_numpy(features, window_size=7)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=0.2, shuffle=False
        )
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Random Forest Regression
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Guardar modelos y predicciones
        self.regression_models = {
            'linear_regression': lr_model,
            'random_forest': rf_model
        }
        
        self.regression_predictions = {
            'y_test': y_test,
            'linear_regression': lr_pred,
            'random_forest': rf_pred,
            'test_dates': self.data.index[-len(y_test):],
            'X_test': X_test,
            'X_train': X_train,
            'y_train': y_train
        }
        
        # Calcular métricas
        self.regression_metrics = {}
        for model_name in ['linear_regression', 'random_forest']:
            pred = self.regression_predictions[model_name]
            self.regression_metrics[model_name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'mae': mean_absolute_error(y_test, pred),
                'r2': r2_score(y_test, pred)
            }
    
    def train_classification_models(self, features):
        """Paso 2: Entrenar modelos de clasificación binaria"""
        # Crear dataset para clasificación (1 día futuro)
        df_1day = self.create_sliding_window(self.data, 'close', 7, 1, features, binarize=True)
        
        X = df_1day.drop('target', axis=1).values
        y = df_1day['target'].astype(int).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Logistic Regression
        lr_clf = LogisticRegression(max_iter=1000, random_state=42)
        lr_clf.fit(X_train, y_train)
        lr_pred = lr_clf.predict(X_test)
        lr_proba = lr_clf.predict_proba(X_test)[:, 1]
        
        # Random Forest Classification
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_proba = rf_clf.predict_proba(X_test)[:, 1]
        
        # Guardar modelos y predicciones
        self.classification_models = {
            'logistic_regression': lr_clf,
            'random_forest': rf_clf
        }
        
        self.classification_predictions = {
            'y_test': y_test,
            'logistic_regression': lr_pred,
            'random_forest': rf_pred,
            'logistic_regression_proba': lr_proba,
            'random_forest_proba': rf_proba,
            'test_dates': self.data.index[-len(y_test):]
        }
        
        # Calcular métricas
        self.classification_metrics = {}
        for model_name in ['logistic_regression', 'random_forest']:
            pred = self.classification_predictions[model_name]
            proba = self.classification_predictions[f'{model_name}_proba']
            self.classification_metrics[model_name] = {
                'accuracy': accuracy_score(y_test, pred),
                'f1_score': f1_score(y_test, pred),
                'roc_auc': roc_auc_score(y_test, proba)
            }
    
    def train_probability_models(self, features):
        """Paso 3: Entrenar modelos de probabilidad de incremento"""
        # Usar datos de diferentes horizontes temporales para mejor análisis
        df_1day = self.create_sliding_window(self.data, 'close', 7, 1, features, binarize=True)
        df_7day = self.create_sliding_window(self.data, 'close', 15, 7, features, binarize=True)
        df_15day = self.create_sliding_window(self.data, 'close', 15, 15, features, binarize=True)
        
        # Usar el dataset de 7 días como principal para el Paso 3
        self.probability_1day_results = self.logistic_regression_analysis(df_1day)
        self.probability_7day_results = self.random_forest_analysis(df_7day)
        self.probability_15day_results = self.random_forest_analysis(df_15day)
        
        # Guardar modelos principales para la API
        self.probability_models = {
            'random_forest': self.probability_7day_results['model'],
            'logistic_regression': self.probability_1day_results['model']
        }
        
        # Usar resultados de 7 días como principales
        main_results = self.probability_7day_results
        
        self.probability_predictions = {
            'y_test': main_results['y_test'],
            'random_forest': main_results['y_pred'],
            'logistic_regression': self.probability_1day_results['y_pred'][:len(main_results['y_test'])],
            'random_forest_proba': main_results['proba_test'],
            'logistic_regression_proba': self.probability_1day_results['proba_test'][:len(main_results['y_test'])],
            'test_dates': self.data.index[-len(main_results['y_test']):]
        }
        
        # Calcular métricas
        self.probability_metrics = {
            'random_forest': main_results['metrics'],
            'logistic_regression': self.probability_1day_results['metrics']
        }
    
    def logistic_regression_analysis(self, df):
        """Implementación de la función logisticRegresion del notebook"""
        # Prepara X e y
        X = df.drop('target', axis=1).values
        y = df['target'].astype(int).values

        # Divide en train/validation (80% / 20%), manteniendo proporción de clases
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        # Crea y entrena el modelo
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Predicciones de probabilidad de clase 1
        probs_train = model.predict_proba(X_train)[:, 1]
        probs_val = model.predict_proba(X_val)[:, 1]
        
        # Predicciones binarias
        y_pred_val = model.predict(X_val)

        # Calcular métricas
        metrics = {
            'train_roc_auc': roc_auc_score(y_train, probs_train),
            'val_roc_auc': roc_auc_score(y_val, probs_val),
            'accuracy': accuracy_score(y_val, y_pred_val),
            'f1_score': f1_score(y_val, y_pred_val),
            'roc_auc': roc_auc_score(y_val, probs_val)
        }

        print(f"Logistic Regression - Train ROC AUC: {metrics['train_roc_auc']:.4f}")
        print(f"Logistic Regression - Val ROC AUC: {metrics['val_roc_auc']:.4f}")
        print(f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_val,
            'y_train': y_train,
            'y_test': y_val,
            'y_pred': y_pred_val,
            'proba_train': probs_train,
            'proba_test': probs_val,
            'metrics': metrics
        }
    
    def random_forest_analysis(self, df):
        """Implementación de la función RandomForest del notebook"""
        # Prepara X e y
        X = df.drop('target', axis=1).values
        y = df['target'].astype(int).values

        # Divide en train/validation (80% / 20%), manteniendo proporción de clases
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        # Crea y entrena el Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'  # opcional, útil si hay desbalance
        )
        rf.fit(X_train, y_train)

        # Obtén probabilidades de la clase "1" (subida)
        probs_train = rf.predict_proba(X_train)[:, 1]
        probs_val = rf.predict_proba(X_val)[:, 1]
        
        # Predicciones binarias
        y_pred_val = (probs_val >= 0.5).astype(int)

        # Calcular métricas
        metrics = {
            'train_roc_auc': roc_auc_score(y_train, probs_train),
            'val_roc_auc': roc_auc_score(y_val, probs_val),
            'accuracy': accuracy_score(y_val, y_pred_val),
            'f1_score': f1_score(y_val, y_pred_val),
            'roc_auc': roc_auc_score(y_val, probs_val)
        }

        print(f"Random Forest - Train ROC AUC: {metrics['train_roc_auc']:.4f}")
        print(f"Random Forest - Val ROC AUC: {metrics['val_roc_auc']:.4f}")
        print(f"Random Forest - Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'model': rf,
            'X_train': X_train,
            'X_test': X_val,
            'y_train': y_train,
            'y_test': y_val,
            'y_pred': y_pred_val,
            'proba_train': probs_train,
            'proba_test': probs_val,
            'metrics': metrics
        }
    
    def create_sliding_windows_numpy(self, features, target="close", window_size=7):
        """Crear ventanas deslizantes en formato numpy (para compatibilidad)"""
        X, y = [], []
        for i in range(window_size, len(self.data)):
            X.append(self.data[features].iloc[i - window_size:i].values)
            y.append(self.data[target].iloc[i])
        return np.array(X), np.array(y)
    
    def predict_future(self, model_name='random_forest', days=30):
        """Predecir precios futuros usando modelos de regresión"""
        if model_name not in self.regression_models:
            return None
        
        model = self.regression_models[model_name]
        features = ['close', 'volume_log', 'marketCap_log', 'rsi_14',
                   'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 
                   'price_range_pct', 'volume_log_change', 'volume_log_ma_7d', 
                   'marketCap_log_change', 'liq_ratio']
        
        # Crear ventanas deslizantes
        X, y = self.create_sliding_windows_numpy(features, window_size=7)
        
        # Última ventana de datos
        last_window = X[-1]  # últimos 7 días
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
    
    def predict_probability_future(self, model_name='random_forest', days=7):
        """Predecir probabilidades futuras de incremento usando modelos de clasificación"""
        if model_name not in self.probability_models:
            return None
        
        model = self.probability_models[model_name]
        features = ['close', 'volume_log', 'marketCap_log', 'rsi_14',
                   'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 
                   'price_range_pct', 'volume_log_change', 'volume_log_ma_7d', 
                   'marketCap_log_change', 'liq_ratio']
        
        # Usar ventanas de diferentes tamaños según el modelo
        window_size = 15 if model_name == 'random_forest' else 7
        
        # Crear dataset para predicción (similar al entrenamiento)
        df_for_prediction = self.create_sliding_window(
            self.data, 'close', window_size, days, features, binarize=True
        )
        
        if len(df_for_prediction) == 0:
            return None
        
        # Usar la última ventana disponible
        last_window = df_for_prediction.drop('target', axis=1).values[-1:] 
        
        # Predecir probabilidad de incremento
        probability = float(model.predict_proba(last_window)[0, 1])
        prediction = bool(model.predict(last_window)[0])
        
        # Crear fechas futuras
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=1,  # Solo una predicción
            freq='D'
        )
        
        return {
            'dates': future_dates,
            'probability': probability,
            'prediction': prediction,
            'prediction_days': days,
            'model_used': model_name
        }
    
    def predict_classification_future(self, model_name='random_forest', days=7):
        """Predecir clasificación futura (subida/bajada) usando modelos de clasificación"""
        if model_name not in self.classification_models:
            return None
        
        model = self.classification_models[model_name]
        features = ['close', 'volume_log', 'marketCap_log', 'rsi_14',
                   'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 
                   'price_range_pct', 'volume_log_change', 'volume_log_ma_7d', 
                   'marketCap_log_change', 'liq_ratio']
        
        # Usar ventanas de 7 días para clasificación (igual que en entrenamiento)
        window_size = 7
        
        # Crear dataset para predicción (similar al entrenamiento)
        df_for_prediction = self.create_sliding_window(
            self.data, 'close', window_size, days, features, binarize=True
        )
        
        if len(df_for_prediction) == 0:
            return None
        
        # Usar la última ventana disponible
        last_window = df_for_prediction.drop('target', axis=1).values[-1:] 
        
        # Predecir clasificación y probabilidad
        prediction = bool(model.predict(last_window)[0])
        probability = float(model.predict_proba(last_window)[0, 1])
        
        # Crear fechas futuras
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=1,  # Solo una predicción
            freq='D'
        )
        
        return {
            'dates': future_dates,
            'prediction': prediction,
            'probability': probability,
            'prediction_days': days,
            'model_used': model_name
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
            'models_trained': crypto_predictor is not None and (
                len(crypto_predictor.regression_models) > 0 and 
                len(crypto_predictor.classification_models) > 0 and 
                len(crypto_predictor.probability_models) > 0
            )
        }
        
        if crypto_predictor is not None:
            status['data_points'] = len(crypto_predictor.data)
            status['regression_models'] = list(crypto_predictor.regression_models.keys())
            status['classification_models'] = list(crypto_predictor.classification_models.keys())
            status['probability_models'] = list(crypto_predictor.probability_models.keys())
        
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

@app.route('/api/step1_data/<model_name>')
@app.route('/api/step1_data/<model_name>/<int:days>')
def get_step1_data(model_name, days=None):
    """API para obtener datos del Paso 1: Regresión"""
    try:
        if model_name not in crypto_predictor.regression_models:
            return jsonify({'error': 'Modelo no encontrado'}), 404
            
        preds = crypto_predictor.regression_predictions
        
        result = {
            'dates': preds['test_dates'].strftime('%Y-%m-%d').tolist(),
            'actual': [float(x) for x in preds['y_test']],
            'predictions': [float(x) for x in preds[model_name]],
            'metrics': crypto_predictor.regression_metrics[model_name],
            'model_type': 'regression',
            'step': 1
        }
        
        # Si se especifican días, agregar predicciones futuras
        if days:
            future_data = crypto_predictor.predict_future(model_name, days)
            if future_data:
                result['future_dates'] = future_data['dates'].strftime('%Y-%m-%d').tolist()
                result['future_predictions'] = [float(x) for x in future_data['predictions']]
                result['prediction_days'] = days
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/step2_data/<model_name>')
@app.route('/api/step2_data/<model_name>/<int:days>')
def get_step2_data(model_name, days=None):
    """API para obtener datos del Paso 2: Clasificación Binaria"""
    try:
        if model_name not in crypto_predictor.classification_models:
            return jsonify({'error': 'Modelo no encontrado'}), 404
            
        preds = crypto_predictor.classification_predictions
        
        result = {
            'dates': preds['test_dates'].strftime('%Y-%m-%d').tolist(),
            'actual': [int(x) for x in preds['y_test']],
            'predictions': [int(x) for x in preds[model_name]],
            'probabilities': [float(x) for x in preds[f'{model_name}_proba']],
            'metrics': crypto_predictor.classification_metrics[model_name],
            'model_type': 'classification',
            'step': 2
        }
        
        # Si se especifican días, obtener predicción futura
        if days:
            future_class = crypto_predictor.predict_classification_future(model_name, days)
            if future_class:
                result['future_prediction'] = {
                    'prediction': future_class['prediction'],
                    'probability': future_class['probability'],
                    'prediction_days': future_class['prediction_days'],
                    'model_used': future_class['model_used']
                }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/step3_data/<model_name>')
@app.route('/api/step3_data/<model_name>/<int:days>')
def get_step3_data(model_name, days=None):
    """API para obtener datos del Paso 3: Probabilidad de Incremento"""
    try:
        if model_name not in crypto_predictor.probability_models:
            return jsonify({'error': 'Modelo no encontrado'}), 404
            
        preds = crypto_predictor.probability_predictions
        
        result = {
            'dates': preds['test_dates'].strftime('%Y-%m-%d').tolist(),
            'actual': [int(x) for x in preds['y_test']],
            'predictions': [int(x) for x in preds[model_name]],
            'probabilities': [float(x) for x in preds[f'{model_name}_proba']],
            'metrics': crypto_predictor.probability_metrics[model_name],
            'model_type': 'probability',
            'step': 3
        }
        
        def RandomForest(df):
            X = df.drop('target', axis=1).values
            y = df['target'].astype(int).values

            # 3) Divide en train/validation (80% / 20%), manteniendo proporción de clases
            X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
                X, y, df, 
                test_size=0.2, 
                random_state=42, 
                stratify=y
            )

            # 5) Crea y entrena el Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'  # opcional, útil si hay desbalance
            )
            rf.fit(X_train, y_train)

            # 6) Obtén probabilidades de la clase “1” (subida)
            probs_train = rf.predict_proba(X_train)[:, 1]
            probs_val   = rf.predict_proba(X_val)[:, 1]

            # 7) Añade esas probabilidades a los DataFrames
            df_train['prob_subida'] = probs_train
            df_val  ['prob_subida'] = probs_val

            # 8) Evalúa rendimiento
            print(f"Train ROC AUC: {roc_auc_score(y_train, probs_train):.4f}")
            print(f" Val  ROC AUC: {roc_auc_score(y_val,   probs_val):.4f}\n")

            y_pred_val = (probs_val >= 0.5).astype(int)
            # Esta función ya está implementada en la clase CryptoPredictor
            # La función duplicada debe ser eliminada

        # Si se especifican días, obtener predicción futura
        if days:
            future_prob = crypto_predictor.predict_probability_future(model_name, days)
            if future_prob:
                result['future_prediction'] = {
                    'probability': future_prob['probability'],
                    'prediction': future_prob['prediction'],
                    'prediction_days': future_prob['prediction_days'],
                    'model_used': future_prob['model_used']
                }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/available_models')
def get_available_models():
    """API para obtener modelos disponibles por paso"""
    try:
        return jsonify({
            'step1': list(crypto_predictor.regression_models.keys()),
            'step2': list(crypto_predictor.classification_models.keys()),
            'step3': list(crypto_predictor.probability_models.keys())
        })
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
    """API para obtener todos los datos para los gráficos principales"""
    try:
        if crypto_predictor is None:
            return jsonify({'error': 'Predictor no inicializado'}), 500
            
        print("Obteniendo datos para gráficos...")
        
        # Datos históricos
        historical = crypto_predictor.data.copy()
        print(f"Datos históricos: {len(historical)} registros")
        
        result = {
            'historical': {
                'dates': historical.index.strftime('%Y-%m-%d').tolist(),
                'close': historical['close'].tolist(),
                'volume': historical['volume_log'].tolist(),
                'rsi': historical['rsi_14'].fillna(50).tolist(),
                'macd': historical['macd'].fillna(0).tolist()
            },
            'step1': {
                'models': list(crypto_predictor.regression_models.keys()),
                'metrics': crypto_predictor.regression_metrics
            },
            'step2': {
                'models': list(crypto_predictor.classification_models.keys()),
                'metrics': crypto_predictor.classification_metrics
            },
            'step3': {
                'models': list(crypto_predictor.probability_models.keys()),
                'metrics': crypto_predictor.probability_metrics
            }
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
