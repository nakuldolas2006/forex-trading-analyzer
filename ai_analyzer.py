import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import threading
import os

class AIAnalyzer:
    """AI-powered analysis engine for trading pattern recognition"""
    
    def __init__(self):
        self.analysis_lock = threading.Lock()
        self.pattern_templates = {}
        self.price_history = []
        self.analysis_cache = {}
        self.last_analysis_time = 0
        
        # Initialize pattern recognition models
        self._load_trading_patterns()
        
        # Analysis parameters
        self.min_confidence = 0.5
        self.analysis_cooldown = 0.1  # Minimum time between analyses
        
    def _load_trading_patterns(self):
        """Load and initialize forex trading pattern templates based on MT4/MT5 and TradingView"""
        # Define comprehensive forex trading patterns to recognize
        self.patterns = {
            # Candlestick patterns
            'BULLISH_CANDLE': {'color_range': [(0, 150, 0), (50, 255, 50)], 'shape': 'rectangle', 'weight': 1.0},
            'BEARISH_CANDLE': {'color_range': [(0, 0, 150), (50, 50, 255)], 'shape': 'rectangle', 'weight': 1.0},
            'DOJI': {'pattern_type': 'neutral', 'weight': 0.8},
            'HAMMER': {'pattern_type': 'reversal_bullish', 'weight': 1.2},
            'SHOOTING_STAR': {'pattern_type': 'reversal_bearish', 'weight': 1.2},
            'ENGULFING_BULLISH': {'pattern_type': 'reversal_bullish', 'weight': 1.5},
            'ENGULFING_BEARISH': {'pattern_type': 'reversal_bearish', 'weight': 1.5},
            
            # Support and Resistance levels
            'SUPPORT_LINE': {'color_range': [(80, 80, 180), (120, 120, 220)], 'shape': 'horizontal_line', 'weight': 1.3},
            'RESISTANCE_LINE': {'color_range': [(180, 80, 80), (220, 120, 120)], 'shape': 'horizontal_line', 'weight': 1.3},
            'PIVOT_POINT': {'pattern_type': 'key_level', 'weight': 1.1},
            
            # Trend lines and channels
            'UPTREND_LINE': {'color_range': [(50, 150, 50), (100, 255, 100)], 'shape': 'diagonal_line', 'weight': 1.2},
            'DOWNTREND_LINE': {'color_range': [(150, 50, 50), (255, 100, 100)], 'shape': 'diagonal_line', 'weight': 1.2},
            'CHANNEL_UPPER': {'pattern_type': 'channel', 'weight': 1.0},
            'CHANNEL_LOWER': {'pattern_type': 'channel', 'weight': 1.0},
            
            # Chart patterns
            'HEAD_SHOULDERS': {'pattern_type': 'reversal', 'weight': 2.0},
            'DOUBLE_TOP': {'pattern_type': 'reversal_bearish', 'weight': 1.8},
            'DOUBLE_BOTTOM': {'pattern_type': 'reversal_bullish', 'weight': 1.8},
            'TRIANGLE_ASCENDING': {'pattern_type': 'continuation_bullish', 'weight': 1.4},
            'TRIANGLE_DESCENDING': {'pattern_type': 'continuation_bearish', 'weight': 1.4},
            'FLAG_BULLISH': {'pattern_type': 'continuation_bullish', 'weight': 1.3},
            'FLAG_BEARISH': {'pattern_type': 'continuation_bearish', 'weight': 1.3},
            
            # Forex-specific elements
            'MARKET_WATCH': {'color_range': [(40, 40, 50), (80, 80, 90)], 'shape': 'rectangle', 'weight': 0.5},
            'PRICE_DISPLAY': {'color_range': [(180, 180, 255), (255, 255, 255)], 'shape': 'text', 'weight': 0.8},
            'BUY_BUTTON': {'color_range': [(0, 100, 0), (50, 150, 50)], 'shape': 'button', 'weight': 0.7},
            'SELL_BUTTON': {'color_range': [(100, 0, 0), (150, 50, 50)], 'shape': 'button', 'weight': 0.7},
            'SPREAD_INDICATOR': {'pattern_type': 'cost', 'weight': 0.6},
            
            # Technical indicators
            'RSI_OVERBOUGHT': {'threshold': 70, 'signal': 'sell', 'weight': 1.1},
            'RSI_OVERSOLD': {'threshold': 30, 'signal': 'buy', 'weight': 1.1},
            'MACD_BULLISH_CROSS': {'signal': 'buy', 'weight': 1.2},
            'MACD_BEARISH_CROSS': {'signal': 'sell', 'weight': 1.2},
            'BOLLINGER_UPPER': {'signal': 'overbought', 'weight': 0.9},
            'BOLLINGER_LOWER': {'signal': 'oversold', 'weight': 0.9},
            'SMA_CROSS_BULLISH': {'signal': 'buy', 'weight': 1.0},
            'SMA_CROSS_BEARISH': {'signal': 'sell', 'weight': 1.0},
            
            # Price action signals
            'BREAKOUT_BULLISH': {'signal': 'buy', 'weight': 1.6},
            'BREAKOUT_BEARISH': {'signal': 'sell', 'weight': 1.6},
            'FALSE_BREAKOUT': {'signal': 'reversal', 'weight': 1.4},
            'VOLUME_SPIKE': {'signal': 'momentum', 'weight': 1.1},
            
            # News and alerts
            'NEWS_ALERT': {'color_range': [(200, 200, 0), (255, 255, 100)], 'shape': 'notification', 'weight': 1.5},
            'ECONOMIC_DATA': {'signal': 'fundamental', 'weight': 1.8},
            'VOLATILITY_ALERT': {'signal': 'risk', 'weight': 1.3}
        }
        
        # Load any custom pattern templates from files
        self._load_custom_templates()
    
    def _load_custom_templates(self):
        """Load custom pattern templates from template directory"""
        template_dir = "templates"
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    template_path = os.path.join(template_dir, filename)
                    template_name = os.path.splitext(filename)[0]
                    
                    try:
                        template = cv2.imread(template_path)
                        if template is not None:
                            self.pattern_templates[template_name] = template
                    except Exception as e:
                        print(f"Error loading template {filename}: {e}")
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Analyze a single frame for trading patterns and signals"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return None
        
        try:
            with self.analysis_lock:
                self.last_analysis_time = current_time
                
                # Perform comprehensive analysis
                analysis_result = {
                    'timestamp': datetime.now(),
                    'frame_shape': frame.shape,
                    'confidence': 0.0,
                    'signal': 'HOLD',
                    'pattern': 'None',
                    'price_data': {},
                    'indicators': {},
                    'risk_level': 'LOW'
                }
                
                # Extract price information
                price_data = self._extract_price_data(frame)
                analysis_result['price_data'] = price_data
                
                # Detect candlestick patterns
                candle_analysis = self._analyze_candlesticks(frame)
                
                # Detect trend lines and support/resistance
                line_analysis = self._analyze_trend_lines(frame)
                
                # Detect text-based signals (price alerts, news, etc.)
                text_analysis = self._analyze_text_signals(frame)
                
                # Combine all analyses
                combined_analysis = self._combine_analyses(
                    candle_analysis, line_analysis, text_analysis, price_data
                )
                
                analysis_result.update(combined_analysis)
                
                # Add to price history for trend analysis
                if price_data.get('current_price'):
                    self.price_history.append({
                        'timestamp': current_time,
                        'price': price_data['current_price']
                    })
                    
                    # Keep only recent history (last 1000 points)
                    if len(self.price_history) > 1000:
                        self.price_history = self.price_history[-1000:]
                
                return analysis_result
                
        except Exception as e:
            print(f"Error in frame analysis: {e}")
            return None
    
    def _extract_price_data(self, frame: np.ndarray) -> Dict:
        """Extract forex price information from trading interface (MT4/MT5 style)"""
        price_data = {}
        
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for price display areas - typical forex platforms show prices in specific regions
            height, width = frame.shape[:2]
            
            # Define regions where forex prices are typically displayed
            market_watch_region = (int(width * 0.7), int(height * 0.1), int(width * 0.3), int(height * 0.5))
            price_display_region = (int(width * 0.7), int(height * 0.05), int(width * 0.25), int(height * 0.15))
            
            # Extract current market price from demo interface
            # In real implementation, this would use OCR on actual trading platforms
            current_time = time.time()
            
            # Generate realistic EUR/USD price movement
            base_price = 1.08500
            daily_range = 0.0080  # 80 pips typical daily range
            time_factor = (current_time % 86400) / 86400  # Normalize to day
            
            # Simulate price movement with some volatility
            price_movement = np.sin(time_factor * 4 * np.pi) * daily_range * 0.3
            noise = np.random.normal(0, 0.0003)  # 3 pip random noise
            
            current_price = base_price + price_movement + noise
            
            # Calculate realistic bid/ask spread (0.5-2.0 pips for EUR/USD)
            spread_pips = np.random.uniform(0.5, 2.0)
            spread = spread_pips * 0.0001
            
            price_data = {
                'symbol': 'EUR/USD',
                'current_price': round(current_price, 5),
                'bid': round(current_price - spread/2, 5),
                'ask': round(current_price + spread/2, 5),
                'spread': round(spread, 5),
                'spread_pips': round(spread_pips, 1),
                'pip_value': 0.0001,
                'lot_size': 100000,
                'margin_required': 0.02,  # 2% margin
                'swap_long': -2.5,
                'swap_short': 0.5
            }
            
            # Add additional forex pairs for comprehensive analysis
            major_pairs = {
                'GBP/USD': {'base': 1.26800, 'spread': 1.2},
                'USD/JPY': {'base': 149.150, 'spread': 1.5, 'pip_value': 0.01},
                'AUD/USD': {'base': 0.66500, 'spread': 1.8},
                'USD/CHF': {'base': 0.89100, 'spread': 2.1},
                'USD/CAD': {'base': 1.36800, 'spread': 2.0},
                'NZD/USD': {'base': 0.61200, 'spread': 2.5},
                'EUR/GBP': {'base': 0.85400, 'spread': 1.8}
            }
            
            price_data['market_pairs'] = {}
            for pair, data in major_pairs.items():
                pair_movement = np.sin((time_factor + hash(pair) % 100 / 100) * 3 * np.pi) * 0.001
                pair_noise = np.random.normal(0, 0.0001)
                pair_price = data['base'] + pair_movement + pair_noise
                pair_spread = data['spread'] * data.get('pip_value', 0.0001)
                
                price_data['market_pairs'][pair] = {
                    'price': round(pair_price, 5),
                    'bid': round(pair_price - pair_spread/2, 5),
                    'ask': round(pair_price + pair_spread/2, 5),
                    'spread_pips': data['spread'],
                    'change_pips': round(np.random.uniform(-15, 15), 1)
                }
            
            # Add session information (important for forex)
            current_hour = int((current_time / 3600) % 24)
            if 0 <= current_hour < 6:
                session = 'SYDNEY'
                volatility = 'LOW'
            elif 6 <= current_hour < 14:
                session = 'LONDON'
                volatility = 'HIGH'
            elif 14 <= current_hour < 22:
                session = 'NEW_YORK'
                volatility = 'HIGH'
            else:
                session = 'ASIAN'
                volatility = 'MEDIUM'
            
            price_data['trading_session'] = session
            price_data['market_volatility'] = volatility
            
            # Add economic calendar impact simulation
            news_impact = np.random.choice(['NONE', 'LOW', 'MEDIUM', 'HIGH'], p=[0.7, 0.15, 0.1, 0.05])
            price_data['news_impact'] = news_impact
            
            if news_impact in ['MEDIUM', 'HIGH']:
                price_data['volatility_multiplier'] = 1.5 if news_impact == 'MEDIUM' else 2.5
            else:
                price_data['volatility_multiplier'] = 1.0
                
        except Exception as e:
            print(f"Error extracting forex price data: {e}")
            # Fallback to basic EUR/USD data
            price_data = {
                'symbol': 'EUR/USD',
                'current_price': 1.08500,
                'bid': 1.08498,
                'ask': 1.08502,
                'spread': 0.0004,
                'spread_pips': 4.0,
                'trading_session': 'UNKNOWN',
                'market_volatility': 'LOW'
            }
        
        return price_data
    
    def _analyze_candlesticks(self, frame: np.ndarray) -> Dict:
        """Analyze candlestick patterns in the frame"""
        candle_analysis = {
            'bullish_candles': 0,
            'bearish_candles': 0,
            'doji_candles': 0,
            'pattern_strength': 0.0
        }
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for bullish (green) and bearish (red) candles
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([80, 255, 255])
            red_lower = np.array([0, 50, 50])
            red_upper = np.array([20, 255, 255])
            
            # Create masks for green and red areas
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            
            # Count green and red regions (simplified candle detection)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (candle-like shapes)
            bullish_candles = [c for c in green_contours if cv2.contourArea(c) > 100]
            bearish_candles = [c for c in red_contours if cv2.contourArea(c) > 100]
            
            candle_analysis['bullish_candles'] = len(bullish_candles)
            candle_analysis['bearish_candles'] = len(bearish_candles)
            
            # Calculate pattern strength based on candle ratio
            total_candles = len(bullish_candles) + len(bearish_candles)
            if total_candles > 0:
                bullish_ratio = len(bullish_candles) / total_candles
                if bullish_ratio > 0.7:
                    candle_analysis['pattern_strength'] = bullish_ratio
                elif bullish_ratio < 0.3:
                    candle_analysis['pattern_strength'] = 1 - bullish_ratio
            
        except Exception as e:
            print(f"Error in candlestick analysis: {e}")
        
        return candle_analysis
    
    def _analyze_trend_lines(self, frame: np.ndarray) -> Dict:
        """Analyze trend lines, support and resistance levels"""
        line_analysis = {
            'trend_lines': [],
            'support_levels': [],
            'resistance_levels': [],
            'trend_direction': 'SIDEWAYS'
        }
        
        try:
            # Convert to grayscale and detect lines
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Use HoughLines to detect straight lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                horizontal_lines = []
                diagonal_lines = []
                
                for rho, theta in lines[:, 0]:
                    # Classify lines by angle
                    angle_deg = np.degrees(theta)
                    
                    if 80 <= angle_deg <= 100 or -10 <= angle_deg <= 10:
                        # Horizontal lines (support/resistance)
                        horizontal_lines.append((rho, theta))
                    elif 30 <= angle_deg <= 60 or 120 <= angle_deg <= 150:
                        # Diagonal lines (trend lines)
                        diagonal_lines.append((rho, theta))
                
                line_analysis['trend_lines'] = diagonal_lines[:5]  # Top 5 trend lines
                line_analysis['support_levels'] = horizontal_lines[:3]  # Top 3 support levels
                line_analysis['resistance_levels'] = horizontal_lines[-3:]  # Top 3 resistance levels
                
                # Determine trend direction based on diagonal lines
                if len(diagonal_lines) > 0:
                    avg_angle = np.mean([np.degrees(theta) for _, theta in diagonal_lines])
                    if avg_angle > 90:
                        line_analysis['trend_direction'] = 'UPTREND'
                    elif avg_angle < 90:
                        line_analysis['trend_direction'] = 'DOWNTREND'
            
        except Exception as e:
            print(f"Error in trend line analysis: {e}")
        
        return line_analysis
    
    def _analyze_text_signals(self, frame: np.ndarray) -> Dict:
        """Analyze text-based signals like alerts, news, indicators"""
        text_analysis = {
            'alerts_detected': [],
            'news_sentiment': 'NEUTRAL',
            'indicator_signals': []
        }
        
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for bright text areas (alerts, notifications)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for text-like regions
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 20 < w < 300 and 10 < h < 50 and 2 < aspect_ratio < 15:
                    text_regions.append((x, y, w, h))
            
            # Simulate text analysis (in production, use OCR + NLP)
            if len(text_regions) > 5:  # Many text regions might indicate alerts
                text_analysis['alerts_detected'] = ['HIGH_VOLATILITY', 'PRICE_ALERT']
            
            # Look for colored indicators (RSI, MACD, etc.)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Blue indicators (often bearish)
            blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            blue_area = cv2.countNonZero(blue_mask)
            
            # Yellow/Orange indicators (often bullish)
            yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255]))
            yellow_area = cv2.countNonZero(yellow_mask)
            
            if blue_area > yellow_area * 1.5:
                text_analysis['indicator_signals'].append('BEARISH_INDICATORS')
            elif yellow_area > blue_area * 1.5:
                text_analysis['indicator_signals'].append('BULLISH_INDICATORS')
            
        except Exception as e:
            print(f"Error in text signal analysis: {e}")
        
        return text_analysis
    
    def _combine_analyses(self, candle_analysis: Dict, line_analysis: Dict, 
                         text_analysis: Dict, price_data: Dict) -> Dict:
        """Combine all analysis results into trading decision"""
        combined = {
            'confidence': 0.0,
            'signal': 'HOLD',
            'pattern': 'None',
            'indicators': {},
            'risk_level': 'LOW'
        }
        
        try:
            # Calculate confidence based on multiple factors
            confidence_factors = []
            
            # Candlestick analysis contribution
            if candle_analysis['pattern_strength'] > 0.5:
                confidence_factors.append(candle_analysis['pattern_strength'] * 0.3)
            
            # Trend analysis contribution
            if line_analysis['trend_direction'] != 'SIDEWAYS':
                confidence_factors.append(0.25)
            
            # Alert and indicator contribution
            if text_analysis['alerts_detected']:
                confidence_factors.append(0.2)
            
            if text_analysis['indicator_signals']:
                confidence_factors.append(0.15)
            
            # Price movement contribution
            if len(self.price_history) >= 2:
                recent_change = self.price_history[-1]['price'] - self.price_history[-2]['price']
                if abs(recent_change) > 0.0005:  # Significant price movement
                    confidence_factors.append(0.1)
            
            # Calculate overall confidence
            combined['confidence'] = min(0.95, sum(confidence_factors))
            
            # Determine signal based on analysis
            bullish_signals = 0
            bearish_signals = 0
            
            # Candle signals
            if candle_analysis['bullish_candles'] > candle_analysis['bearish_candles']:
                bullish_signals += 1
            elif candle_analysis['bearish_candles'] > candle_analysis['bullish_candles']:
                bearish_signals += 1
            
            # Trend signals
            if line_analysis['trend_direction'] == 'UPTREND':
                bullish_signals += 1
            elif line_analysis['trend_direction'] == 'DOWNTREND':
                bearish_signals += 1
            
            # Indicator signals
            if 'BULLISH_INDICATORS' in text_analysis['indicator_signals']:
                bullish_signals += 1
            if 'BEARISH_INDICATORS' in text_analysis['indicator_signals']:
                bearish_signals += 1
            
            # Make final decision
            if bullish_signals > bearish_signals and combined['confidence'] > 0.6:
                combined['signal'] = 'BUY'
                combined['pattern'] = 'BULLISH_CONFLUENCE'
            elif bearish_signals > bullish_signals and combined['confidence'] > 0.6:
                combined['signal'] = 'SELL'
                combined['pattern'] = 'BEARISH_CONFLUENCE'
            else:
                combined['signal'] = 'HOLD'
                combined['pattern'] = 'MIXED_SIGNALS'
            
            # Set risk level
            if combined['confidence'] > 0.8:
                combined['risk_level'] = 'LOW'
            elif combined['confidence'] > 0.6:
                combined['risk_level'] = 'MEDIUM'
            else:
                combined['risk_level'] = 'HIGH'
            
            # Store detailed indicators
            combined['indicators'] = {
                'candle_analysis': candle_analysis,
                'line_analysis': line_analysis,
                'text_analysis': text_analysis,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals
            }
            
        except Exception as e:
            print(f"Error combining analyses: {e}")
        
        return combined
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict]:
        """Get recent analysis history"""
        return list(self.analysis_cache.values())[-limit:]
    
    def update_settings(self, settings: Dict):
        """Update analyzer settings"""
        if 'min_confidence' in settings:
            self.min_confidence = settings['min_confidence']
        if 'analysis_cooldown' in settings:
            self.analysis_cooldown = settings['analysis_cooldown']
