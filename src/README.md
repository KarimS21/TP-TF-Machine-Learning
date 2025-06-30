# Dashboard de Predicción de Criptomonedas

Una aplicación web Flask que muestra gráficos interactivos de predicción de precios de criptomonedas usando modelos de Machine Learning.

## Características

- 📈 **Gráficos Interactivos**: Visualización de precios históricos, predicciones y proyecciones futuras
- 🤖 **Modelos ML**: Random Forest y Regresión Lineal para predicción de precios
- 📊 **Indicadores Técnicos**: RSI, MACD, Estocástico y más
- 🎛️ **Dashboard Responsivo**: Interfaz moderna con Bootstrap
- 📱 **Métricas en Tiempo Real**: RMSE, MAE, R² y último precio

## Instalación

1. **Clonar el repositorio**:
   ```bash
   cd src
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verificar que existe el archivo de datos**:
   - Asegúrate de que existe `../Data/new_data.csv`
   - Si no existe, la aplicación creará datos de ejemplo

## Uso

1. **Ejecutar la aplicación**:
   ```bash
   python flask.py
   ```

2. **Abrir en el navegador**:
   - Ir a `http://localhost:5000`

3. **Usar el dashboard**:
   - Seleccionar modelo (Random Forest o Regresión Lineal)
   - Elegir días de predicción (7, 15, 30, 60)
   - Hacer clic en "Actualizar Gráfico"

## Estructura del Proyecto

```
src/
├── flask.py              # Aplicación Flask principal
├── templates/
│   └── index.html        # Interfaz web
├── requirements.txt      # Dependencias
└── README.md            # Este archivo
```

## APIs Disponibles

- `GET /api/historical_data` - Datos históricos
- `GET /api/model_predictions` - Predicciones de modelos
- `GET /api/future_predictions/<model>/<days>` - Predicciones futuras
- `GET /api/chart_data` - Todos los datos para gráficos

## Modelos Implementados

1. **Random Forest**: Modelo ensemble robusto
2. **Regresión Lineal**: Modelo baseline simple

## Indicadores Técnicos

- **RSI (14)**: Índice de Fuerza Relativa
- **MACD**: Convergencia/Divergencia de Medias Móviles
- **Estocástico**: Oscilador de momentum
- **Características adicionales**: Volumen, capitalización, etc.

## Personalización

Para usar tus propios datos:
1. Coloca tu archivo CSV en `../Data/new_data.csv`
2. Asegúrate de que tenga las columnas: `open`, `high`, `low`, `close`, `volume_log`, `marketCap_log`, `timestamp`

## Troubleshooting

- **Error de datos**: Verifica que el archivo CSV existe y tiene el formato correcto
- **Puerto ocupado**: Cambia el puerto en `app.run(port=5001)`
- **Dependencias**: Ejecuta `pip install -r requirements.txt`

## Próximas Mejoras

- [ ] Más modelos de ML (XGBoost, LSTM)
- [ ] Alertas de trading
- [ ] Análisis de sentimiento
- [ ] Múltiples criptomonedas
- [ ] Backtesting de estrategias