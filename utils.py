import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        "recordings",
        "templates",
        "logs",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%H:%M:%S")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def calculate_pip_value(price: float, currency_pair: str = "EURUSD") -> float:
    """Calculate pip value for currency pair"""
    # Simplified pip calculation
    if currency_pair in ["USDJPY", "EURJPY", "GBPJPY"]:
        return 0.01  # For JPY pairs, pip is 0.01
    else:
        return 0.0001  # For major pairs, pip is 0.0001

def price_to_pips(price_diff: float, currency_pair: str = "EURUSD") -> float:
    """Convert price difference to pips"""
    pip_value = calculate_pip_value(0, currency_pair)
    return price_diff / pip_value

def pips_to_price(pips: float, currency_pair: str = "EURUSD") -> float:
    """Convert pips to price difference"""
    pip_value = calculate_pip_value(0, currency_pair)
    return pips * pip_value

def calculate_position_size(account_balance: float, risk_percent: float, 
                          stop_loss_pips: float, pip_value: float = 10) -> float:
    """Calculate position size based on risk management"""
    risk_amount = account_balance * (risk_percent / 100)
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)

def validate_trading_params(params: Dict) -> Tuple[bool, str]:
    """Validate trading parameters"""
    required_fields = ['symbol', 'side', 'amount', 'entry_price']
    
    for field in required_fields:
        if field not in params:
            return False, f"Missing required field: {field}"
    
    # Validate side
    if params['side'].upper() not in ['BUY', 'SELL']:
        return False, "Side must be BUY or SELL"
    
    # Validate amount
    if params['amount'] <= 0:
        return False, "Amount must be positive"
    
    # Validate entry price
    if params['entry_price'] <= 0:
        return False, "Entry price must be positive"
    
    # Validate stop loss and take profit if provided
    if 'stop_loss' in params and params['stop_loss'] <= 0:
        return False, "Stop loss must be positive"
    
    if 'take_profit' in params and params['take_profit'] <= 0:
        return False, "Take profit must be positive"
    
    return True, "Valid"

def calculate_profit_loss(entry_price: float, current_price: float, 
                         position_size: float, side: str) -> float:
    """Calculate profit/loss for a position"""
    price_diff = current_price - entry_price
    
    if side.upper() == 'SELL':
        price_diff = -price_diff
    
    # Convert to dollars (simplified for major pairs)
    pnl = price_diff * position_size * 100000  # Standard lot size
    return round(pnl, 2)

def detect_trading_interface(frame: np.ndarray) -> Dict:
    """Detect if frame contains a trading interface"""
    result = {
        'is_trading_interface': False,
        'confidence': 0.0,
        'detected_elements': [],
        'platform': 'Unknown'
    }
    
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for common trading interface elements
        elements_found = 0
        
        # Check for price-like patterns (numbers with 4-5 decimal places)
        # Look for rectangular regions that might contain prices
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        price_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Price display characteristics
            if 50 < w < 200 and 15 < h < 50 and 2 < aspect_ratio < 8:
                price_regions += 1
        
        if price_regions >= 3:
            elements_found += 1
            result['detected_elements'].append('price_displays')
        
        # Check for chart-like elements (candlesticks, lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None and len(lines) > 20:
            elements_found += 1
            result['detected_elements'].append('chart_lines')
        
        # Check for button-like elements
        # Look for rectangular shapes that might be buttons
        button_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Button characteristics
            if 60 < w < 150 and 20 < h < 40 and 1.5 < aspect_ratio < 5:
                button_count += 1
        
        if button_count >= 2:
            elements_found += 1
            result['detected_elements'].append('buttons')
        
        # Check for color patterns typical of trading platforms
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green areas (bullish)
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        green_area = cv2.countNonZero(green_mask)
        
        # Red areas (bearish)
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([20, 255, 255]))
        red_area = cv2.countNonZero(red_mask)
        
        total_pixels = frame.shape[0] * frame.shape[1]
        color_ratio = (green_area + red_area) / total_pixels
        
        if color_ratio > 0.05:  # At least 5% colored areas
            elements_found += 1
            result['detected_elements'].append('trading_colors')
        
        # Calculate confidence based on elements found
        max_elements = 4
        result['confidence'] = min(elements_found / max_elements, 1.0)
        result['is_trading_interface'] = result['confidence'] > 0.5
        
        # Try to identify platform (simplified)
        if elements_found >= 3:
            if price_regions > 5 and len(lines) > 50:
                result['platform'] = 'MetaTrader'
            elif button_count > 4:
                result['platform'] = 'WebTrader'
            else:
                result['platform'] = 'Generic Trading Platform'
        
    except Exception as e:
        print(f"Error in trading interface detection: {e}")
    
    return result

def extract_text_regions(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Extract regions that likely contain text"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on text characteristics
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Text region criteria
            if (10 < w < 300 and 8 < h < 50 and 
                1 < aspect_ratio < 20 and 
                100 < area < 5000):
                text_regions.append((x, y, w, h))
        
        return text_regions
        
    except Exception as e:
        print(f"Error extracting text regions: {e}")
        return []

def create_analysis_report(analysis_results: List[Dict], trade_results: List[Dict]) -> Dict:
    """Create comprehensive analysis report"""
    if not analysis_results:
        return {}
    
    try:
        # Calculate statistics
        total_analyses = len(analysis_results)
        
        # Confidence statistics
        confidences = [r.get('confidence', 0) for r in analysis_results]
        avg_confidence = np.mean(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        
        # Signal distribution
        signals = [r.get('signal', 'HOLD') for r in analysis_results]
        signal_counts = {
            'BUY': signals.count('BUY'),
            'SELL': signals.count('SELL'),
            'HOLD': signals.count('HOLD')
        }
        
        # Pattern analysis
        patterns = [r.get('pattern', 'None') for r in analysis_results]
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Trading performance
        trade_stats = {}
        if trade_results:
            total_trades = len(trade_results)
            profitable_trades = sum(1 for t in trade_results if t.get('profit_loss', 0) > 0)
            total_pnl = sum(t.get('profit_loss', 0) for t in trade_results)
            
            trade_stats = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
            }
        
        # Time analysis
        if analysis_results:
            start_time = analysis_results[0].get('timestamp')
            end_time = analysis_results[-1].get('timestamp')
            duration = (end_time - start_time).total_seconds() if start_time and end_time else 0
        else:
            duration = 0
        
        report = {
            'analysis_stats': {
                'total_analyses': total_analyses,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'duration_seconds': duration,
                'analyses_per_minute': (total_analyses / (duration / 60)) if duration > 0 else 0
            },
            'signal_distribution': signal_counts,
            'pattern_distribution': pattern_counts,
            'trading_performance': trade_stats,
            'generated_at': datetime.now()
        }
        
        return report
        
    except Exception as e:
        print(f"Error creating analysis report: {e}")
        return {}

def log_error(error_message: str, context: str = ""):
    """Log error message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ERROR in {context}: {error_message}"
    
    # Print to console
    print(log_entry)
    
    # Save to log file
    try:
        with open("logs/error.log", "a") as f:
            f.write(log_entry + "\n")
    except Exception:
        pass  # Fail silently if can't write to log

def log_trade(trade_data: Dict):
    """Log trade execution details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open("logs/trades.log", "a") as f:
            f.write(f"[{timestamp}] TRADE: {trade_data}\n")
    except Exception:
        pass  # Fail silently if can't write to log

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system operations"""
    import re
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    return filename
