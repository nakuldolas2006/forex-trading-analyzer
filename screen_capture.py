import mss
import cv2
import numpy as np
from typing import Optional, Tuple, List
import threading
import time
import os

class ScreenCapture:
    """Handles Windows screen capture functionality"""
    
    def __init__(self):
        self.sct = None
        self.monitor_index = 0
        self.custom_region = None
        self.capture_lock = threading.Lock()
        self.is_headless = self._check_headless_environment()
        
        # Initialize screen capture if possible
        try:
            if not self.is_headless:
                self.sct = mss.mss()
                # Only import pyautogui if we have display access
                import pyautogui
                pyautogui.FAILSAFE = False
                self.pyautogui = pyautogui
            else:
                print("Running in headless environment - screen capture will use demo mode")
                self.sct = None
                self.pyautogui = None
        except Exception as e:
            print(f"Screen capture initialization failed: {e}")
            self.is_headless = True
            self.sct = None
            self.pyautogui = None
    
    def _check_headless_environment(self) -> bool:
        """Check if running in headless environment"""
        # Check if DISPLAY is set and accessible
        if 'DISPLAY' not in os.environ:
            return True
        
        # Check if we're in a container or CI environment
        if any(env in os.environ for env in ['CI', 'REPLIT_ENVIRONMENT', 'GITHUB_ACTIONS']):
            return True
        
        return False
        
    def get_monitors(self) -> List[dict]:
        """Get list of available monitors"""
        if self.is_headless or not self.sct:
            # Return demo monitor info for headless environment
            return [{'left': 0, 'top': 0, 'width': 1920, 'height': 1080}]
        return self.sct.monitors[1:]  # Skip the combined monitor at index 0
    
    def set_monitor(self, monitor_index: int):
        """Set the monitor to capture from"""
        monitors = self.get_monitors()
        if 0 <= monitor_index < len(monitors):
            self.monitor_index = monitor_index
        else:
            raise ValueError(f"Monitor index {monitor_index} out of range")
    
    def set_region(self, x: int, y: int, width: int, height: int):
        """Set custom capture region"""
        self.custom_region = {
            'left': x,
            'top': y,
            'width': width,
            'height': height
        }
    
    def clear_region(self):
        """Clear custom capture region"""
        self.custom_region = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the screen"""
        try:
            with self.capture_lock:
                if self.is_headless or not self.sct:
                    # Generate demo frame for headless environment
                    return self._generate_demo_frame()
                
                if self.custom_region:
                    # Capture custom region
                    monitor = self.custom_region
                else:
                    # Capture full monitor
                    monitors = self.get_monitors()
                    if self.monitor_index < len(monitors):
                        monitor = monitors[self.monitor_index]
                    else:
                        monitor = monitors[0]  # Fallback to first monitor
                
                # Capture screenshot
                screenshot = self.sct.grab(monitor)
                
                # Convert to numpy array
                frame = np.array(screenshot)
                
                # Convert BGRA to BGR (remove alpha channel)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                return frame
                
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return self._generate_demo_frame()
    
    def _generate_demo_frame(self) -> np.ndarray:
        """Generate a realistic forex trading interface simulation based on MT4/MT5 and TradingView"""
        import random
        import math
        
        height, width = 720, 1280  # Standard HD resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dark professional trading platform background
        frame[:] = (25, 25, 35)  # Dark charcoal
        
        # Main chart area (like TradingView/MT4)
        chart_x, chart_y = 60, 60
        chart_w, chart_h = 800, 480
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (35, 35, 45), -1)
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (60, 60, 70), 1)
        
        # Generate realistic forex candlesticks with proper OHLC data
        random.seed(int(time.time()) % 100)
        base_price = 1.08500  # EUR/USD base
        candle_width = 12
        num_candles = chart_w // (candle_width + 3)
        
        prices = []
        current_price = base_price
        
        for i in range(num_candles):
            # Generate realistic price movement
            price_change = random.uniform(-0.0020, 0.0020)
            current_price += price_change
            
            open_price = current_price
            high_price = open_price + random.uniform(0, 0.0015)
            low_price = open_price - random.uniform(0, 0.0015)
            close_price = open_price + random.uniform(-0.0010, 0.0010)
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'bullish': close_price > open_price
            })
        
        # Draw candlesticks
        price_range = 0.008  # 80 pips range
        price_min = min([p['low'] for p in prices])
        price_scale = chart_h / price_range
        
        for i, price_data in enumerate(prices):
            x = chart_x + i * (candle_width + 3)
            
            # Calculate y positions
            open_y = chart_y + chart_h - int((price_data['open'] - price_min) * price_scale)
            close_y = chart_y + chart_h - int((price_data['close'] - price_min) * price_scale)
            high_y = chart_y + chart_h - int((price_data['high'] - price_min) * price_scale)
            low_y = chart_y + chart_h - int((price_data['low'] - price_min) * price_scale)
            
            # Draw wick (high-low line)
            cv2.line(frame, (x + candle_width//2, high_y), (x + candle_width//2, low_y), (150, 150, 150), 1)
            
            # Draw candle body
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)
            color = (0, 180, 0) if price_data['bullish'] else (0, 0, 180)  # Green/Red
            
            if body_bottom - body_top < 2:  # Doji candle
                cv2.line(frame, (x, body_top), (x + candle_width, body_top), color, 2)
            else:
                cv2.rectangle(frame, (x, body_top), (x + candle_width, body_bottom), color, -1)
        
        # Add grid lines (like professional platforms)
        grid_color = (50, 50, 60)
        for i in range(5):
            y = chart_y + i * (chart_h // 4)
            cv2.line(frame, (chart_x, y), (chart_x + chart_w, y), grid_color, 1)
        
        # Market Watch Panel (right side like MT4/MT5)
        watch_x, watch_y = 880, 60
        watch_w, watch_h = 380, 300
        cv2.rectangle(frame, (watch_x, watch_y), (watch_x + watch_w, watch_y + watch_h), (40, 40, 50), -1)
        cv2.rectangle(frame, (watch_x, watch_y), (watch_x + watch_w, watch_y + watch_h), (80, 80, 90), 2)
        
        # Market Watch Header
        cv2.putText(frame, "Market Watch", (watch_x + 10, watch_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Currency pairs with live-like prices
        forex_pairs = [
            ("EUR/USD", 1.08456, 1.08459, "+0.0012"),
            ("GBP/USD", 1.26834, 1.26837, "-0.0008"),
            ("USD/JPY", 149.123, 149.128, "+0.234"),
            ("AUD/USD", 0.66542, 0.66545, "+0.0003"),
            ("USD/CHF", 0.89123, 0.89126, "-0.0015"),
            ("USD/CAD", 1.36789, 1.36792, "+0.0021"),
            ("NZD/USD", 0.61234, 0.61237, "-0.0007"),
            ("EUR/GBP", 0.85456, 0.85459, "+0.0004")
        ]
        
        for i, (pair, bid, ask, change) in enumerate(forex_pairs):
            y_pos = watch_y + 50 + i * 30
            # Pair name
            cv2.putText(frame, pair, (watch_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            # Bid price
            cv2.putText(frame, f"{bid:.5f}", (watch_x + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
            # Ask price  
            cv2.putText(frame, f"{ask:.5f}", (watch_x + 180, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 180), 1)
            # Change
            change_color = (0, 200, 0) if change.startswith('+') else (0, 0, 200)
            cv2.putText(frame, change, (watch_x + 260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, change_color, 1)
        
        # Trading Panel (bottom right)
        trade_x, trade_y = 880, 380
        trade_w, trade_h = 380, 280
        cv2.rectangle(frame, (trade_x, trade_y), (trade_x + trade_w, trade_y + trade_h), (45, 45, 55), -1)
        cv2.rectangle(frame, (trade_x, trade_y), (trade_x + trade_w, trade_y + trade_h), (80, 80, 90), 2)
        
        # Trading panel elements
        cv2.putText(frame, "Order Entry", (trade_x + 10, trade_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "Symbol: EUR/USD", (trade_x + 10, trade_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Volume: 0.10", (trade_x + 10, trade_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Stop Loss: 1.0820", (trade_x + 10, trade_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Take Profit: 1.0880", (trade_x + 10, trade_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Buy/Sell buttons (like MT4/MT5)
        buy_button = (trade_x + 20, trade_y + 160, 80, 35)
        sell_button = (trade_x + 120, trade_y + 160, 80, 35)
        cv2.rectangle(frame, (buy_button[0], buy_button[1]), (buy_button[0] + buy_button[2], buy_button[1] + buy_button[3]), (0, 120, 0), -1)
        cv2.rectangle(frame, (sell_button[0], sell_button[1]), (sell_button[0] + sell_button[2], sell_button[1] + sell_button[3]), (120, 0, 0), -1)
        cv2.putText(frame, "BUY", (buy_button[0] + 20, buy_button[1] + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SELL", (sell_button[0] + 15, sell_button[1] + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Account info panel (top right)
        acc_x, acc_y = 880, 10
        cv2.putText(frame, "Balance: $10,000.00", (acc_x, acc_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Equity: $10,087.50", (acc_x, acc_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.putText(frame, "Margin: $432.10", (acc_x + 150, acc_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Free: $9,655.40", (acc_x + 150, acc_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Chart timeframe buttons (like TradingView)
        timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
        for i, tf in enumerate(timeframes):
            tf_x = chart_x + i * 45
            tf_y = chart_y - 35
            active = i == 4  # H1 active
            color = (100, 100, 150) if active else (60, 60, 70)
            cv2.rectangle(frame, (tf_x, tf_y), (tf_x + 40, tf_y + 25), color, -1)
            cv2.rectangle(frame, (tf_x, tf_y), (tf_x + 40, tf_y + 25), (120, 120, 130), 1)
            cv2.putText(frame, tf, (tf_x + 8, tf_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add indicators panel (left side)
        ind_x, ind_y = 10, 60
        cv2.putText(frame, "Indicators", (ind_x, ind_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        indicators = ["RSI: 56.2", "MACD: +0.0012", "SMA20: 1.0841", "BB: 1.0855"]
        for i, ind in enumerate(indicators):
            cv2.putText(frame, ind, (ind_x, ind_y + 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # Terminal tabs (bottom)
        tabs = ["Trade", "Account History", "Journal", "Alerts"]
        tab_y = height - 120
        for i, tab in enumerate(tabs):
            tab_x = 60 + i * 120
            active = i == 0
            color = (60, 60, 80) if active else (40, 40, 50)
            cv2.rectangle(frame, (tab_x, tab_y), (tab_x + 110, tab_y + 25), color, -1)
            cv2.putText(frame, tab, (tab_x + 10, tab_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add some realistic trend lines and levels
        support_y = chart_y + int(chart_h * 0.7)
        resistance_y = chart_y + int(chart_h * 0.3)
        cv2.line(frame, (chart_x, support_y), (chart_x + chart_w, support_y), (100, 100, 200), 2)
        cv2.line(frame, (chart_x, resistance_y), (chart_x + chart_w, resistance_y), (200, 100, 100), 2)
        cv2.putText(frame, "Support 1.0820", (chart_x + chart_w - 120, support_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 200), 1)
        cv2.putText(frame, "Resistance 1.0860", (chart_x + chart_w - 140, resistance_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 100), 1)
        
        # Demo mode watermark
        cv2.putText(frame, "DEMO TRADING MODE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add current time
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def capture_region_interactive(self) -> Tuple[int, int, int, int]:
        """Interactive region selection (for future GUI implementation)"""
        # For now, return a default region
        # This could be enhanced with a GUI selection tool
        if self.pyautogui:
            screen_width, screen_height = self.pyautogui.size()
        else:
            screen_width, screen_height = 1920, 1080  # Default resolution
        return (
            screen_width // 4,      # x
            screen_height // 4,     # y
            screen_width // 2,      # width
            screen_height // 2      # height
        )
    
    def get_screen_info(self) -> dict:
        """Get information about the current screen setup"""
        monitors = self.get_monitors()
        current_monitor = monitors[self.monitor_index] if self.monitor_index < len(monitors) else monitors[0]
        
        if self.pyautogui:
            screen_size = self.pyautogui.size()
        else:
            screen_size = (1920, 1080)  # Default size for headless
        
        return {
            'total_monitors': len(monitors),
            'current_monitor': self.monitor_index,
            'monitor_info': current_monitor,
            'custom_region': self.custom_region,
            'screen_size': screen_size,
            'headless_mode': self.is_headless
        }
    
    def test_capture(self) -> bool:
        """Test if screen capture is working"""
        frame = self.capture_frame()
        return frame is not None and frame.size > 0
    
    def capture_with_cursor(self) -> Optional[np.ndarray]:
        """Capture frame with cursor position marked"""
        frame = self.capture_frame()
        if frame is not None:
            try:
                if self.pyautogui:
                    # Get cursor position
                    pos = self.pyautogui.position()
                    cursor_x, cursor_y = int(pos.x), int(pos.y)
                    
                    # Adjust cursor position relative to capture region
                    if self.custom_region:
                        cursor_x -= self.custom_region['left']
                        cursor_y -= self.custom_region['top']
                    else:
                        monitors = self.get_monitors()
                        monitor = monitors[self.monitor_index] if self.monitor_index < len(monitors) else monitors[0]
                        cursor_x -= monitor['left']
                        cursor_y -= monitor['top']
                    
                    # Draw cursor indicator
                    if 0 <= cursor_x < frame.shape[1] and 0 <= cursor_y < frame.shape[0]:
                        cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), 2)
                        cv2.circle(frame, (cursor_x, cursor_y), 3, (0, 255, 0), -1)
                else:
                    # In headless mode, add a simulated cursor
                    cursor_x, cursor_y = frame.shape[1] // 2, frame.shape[0] // 2
                    cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 255), 2)
                    cv2.circle(frame, (cursor_x, cursor_y), 3, (0, 255, 255), -1)
                
            except Exception as e:
                print(f"Error adding cursor to frame: {e}")
        
        return frame
    
    def capture_continuous(self, callback, fps: int = 10):
        """Continuous capture with callback for each frame"""
        interval = 1.0 / fps
        
        while True:
            start_time = time.time()
            
            frame = self.capture_frame()
            if frame is not None:
                callback(frame)
            
            # Maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get pixel color at specific coordinates"""
        try:
            if self.pyautogui:
                # Use pyautogui for quick pixel access
                pixel = self.pyautogui.pixel(x, y)
                return pixel  # Returns (R, G, B)
            else:
                # Return demo color for headless mode
                return (100, 100, 100)
        except Exception as e:
            print(f"Error getting pixel color: {e}")
            return (0, 0, 0)
    
    def find_template(self, template_path: str, threshold: float = 0.8) -> List[Tuple[int, int]]:
        """Find template matches in current screen"""
        try:
            frame = self.capture_frame()
            if frame is None:
                return []
            
            # Load template
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                return []
            
            # Convert to grayscale for matching
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            locations = np.where(result >= threshold)
            matches = list(zip(*locations[::-1]))  # Convert to (x, y) format
            
            return matches
            
        except Exception as e:
            print(f"Error in template matching: {e}")
            return []
