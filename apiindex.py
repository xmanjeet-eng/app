# api/index.py
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import json

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import yfinance as yf

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')

class SimpleNiftyPredictor:
    def __init__(self):
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 300  # 5 minutes
    
    def fetch_data(self):
        """Fetch Nifty 50 data"""
        try:
            # Try multiple tickers
            tickers = ['^NSEI', 'NSEI.NS', '^NSEI.BO']
            for ticker in tickers:
                try:
                    data = yf.download(ticker, period='1mo', progress=False, timeout=10)
                    if not data.empty and len(data) > 5:
                        print(f"Successfully fetched data using {ticker}")
                        return data
                except Exception as e:
                    print(f"Failed with {ticker}: {e}")
                    continue
            
            # Fallback to sample data
            print("Using sample data")
            dates = pd.date_range(end=datetime.now(), periods=30, freq='B')
            prices = 22000 + np.random.randn(30).cumsum() * 100
            return pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, 30)
            }, index=dates)
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return minimal sample data
            dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
            prices = 22000 + np.random.randn(30).cumsum() * 100
            return pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices
            }, index=dates)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            if len(prices) < period:
                return 50
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def calculate_sma(self, prices, window=20):
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < window:
                return float(prices.iloc[-1])
            return float(prices.rolling(window=window).mean().iloc[-1])
        except:
            return float(prices.iloc[-1])
    
    def predict(self, data):
        """Make prediction based on technical indicators"""
        try:
            if len(data) < 10:
                return 'NEUTRAL', 50.0
            
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            
            # Calculate indicators
            price_change = ((current_price - prev_price) / prev_price * 100)
            rsi = self.calculate_rsi(data['Close'])
            sma_20 = self.calculate_sma(data['Close'], 20)
            sma_10 = self.calculate_sma(data['Close'], 10)
            
            # Rule-based prediction
            score = 0
            
            # Price momentum
            if price_change > 0.5:
                score += 2
            elif price_change < -0.5:
                score -= 2
            
            # RSI conditions
            if rsi < 30:
                score += 2  # Oversold - bullish signal
            elif rsi > 70:
                score -= 2  # Overbought - bearish signal
            
            # Moving average crossover
            if sma_10 > sma_20:
                score += 1  # Short-term above long-term - bullish
            else:
                score -= 1  # Short-term below long-term - bearish
            
            # Determine prediction
            if score >= 3:
                prediction = 'BULLISH'
                confidence = min(80 + score, 90)
            elif score <= -3:
                prediction = 'BEARISH'
                confidence = min(80 - score, 90)
            else:
                prediction = 'NEUTRAL'
                confidence = 55 + abs(score) * 5
            
            return prediction, round(min(max(confidence, 50), 95), 1)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'NEUTRAL', 50.0
    
    def get_analysis(self, prediction, confidence):
        """Get analysis text"""
        analyses = {
            'BULLISH': [
                "📈 Positive momentum detected with improving market sentiment.",
                "🚀 Technical indicators show strength and upward potential.",
                "✅ Market conditions favor bullish movement in near term."
            ],
            'BEARISH': [
                "⚠️ Caution advised as technical indicators show weakness.",
                "📉 Market sentiment appears negative with downward pressure.",
                "🔄 Consider risk management strategies in current conditions."
            ],
            'NEUTRAL': [
                "⚖️ Market in consolidation phase with balanced forces.",
                "🔄 Mixed signals suggest waiting for clearer direction.",
                "⏳ Patience recommended as market seeks new direction."
            ]
        }
        
        texts = analyses.get(prediction, analyses['NEUTRAL'])
        idx = 0 if confidence > 75 else 1 if confidence > 60 else 2
        return texts[idx % len(texts)]

# Initialize predictor
predictor = SimpleNiftyPredictor()

@app.route('/', methods=['GET'])
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    """API endpoint for prediction"""
    try:
        # Fetch data
        data = predictor.fetch_data()
        
        # Calculate prediction
        prediction, confidence = predictor.predict(data)
        
        # Get analysis
        analysis = predictor.get_analysis(prediction, confidence)
        
        # Get current price
        current_price = float(data['Close'].iloc[-1])
        
        # Calculate today's change
        if len(data) > 1:
            prev_price = float(data['Close'].iloc[-2])
            change = ((current_price - prev_price) / prev_price * 100)
            today_change = f"{'+' if change >= 0 else ''}{change:.2f}%"
            change_color = 'green' if change >= 0 else 'red'
        else:
            today_change = "0.00%"
            change_color = 'gray'
        
        # Calculate volatility
        if len(data) > 5:
            returns = data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility %
            expected_move = f"±{volatility * 0.1:.2f}%"  # Daily expected move
        else:
            expected_move = "±1.5%"
        
        # Get technical indicators
        rsi = predictor.calculate_rsi(data['Close'])
        sma_20 = predictor.calculate_sma(data['Close'], 20)
        price_vs_sma = ((current_price - sma_20) / sma_20 * 100)
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'analysis': analysis,
            'data': {
                'current_price': f"₹{current_price:,.2f}",
                'today_change': today_change,
                'change_color': change_color,
                'expected_move': expected_move,
                'rsi': round(rsi, 1),
                'price_vs_sma': f"{'+' if price_vs_sma >= 0 else ''}{price_vs_sma:.2f}%",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            'success': False,
            'prediction': 'NEUTRAL',
            'confidence': 50.0,
            'analysis': 'System updating. Please refresh.',
            'data': {
                'current_price': '₹22,000.00',
                'today_change': '0.00%',
                'change_color': 'gray',
                'expected_move': '±0.0%',
                'rsi': 50.0,
                'price_vs_sma': '0.00%',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Nifty 50 AI Predictor'
    })

# Vercel requires this
if __name__ == '__main__':
    app.run(debug=False)
else:
    # For Vercel serverless
    from serverless_wsgi import handle_request
    handler = handle_request(app)