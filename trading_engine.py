import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import json

class TradingEngine:
    """Automated trading execution engine"""
    
    def __init__(self, api_key: str = "demo_key"):
        self.api_key = api_key or os.getenv("TRADING_API_KEY", "demo_key")
        self.base_url = os.getenv("TRADING_API_URL", "https://api.demo.trading.com")
        
        # Professional forex trading settings
        self.settings = {
            'enabled': False,
            'lot_size': 0.10,  # Standard lot size (0.01 = micro, 0.10 = mini, 1.0 = standard)
            'max_trades_per_hour': 5,  # Conservative for forex
            'max_daily_trades': 20,
            'confidence_threshold': 0.75,  # Higher threshold for forex
            'stop_loss_pips': 15,  # Tighter stops for forex
            'take_profit_pips': 30,  # 2:1 risk/reward ratio
            'max_spread_pips': 2.5,  # Maximum spread for major pairs
            'currency_pair': 'EURUSD',
            'risk_per_trade_percent': 2.0,  # Risk 2% per trade
            'max_drawdown_percent': 10.0,  # Maximum account drawdown
            'trailing_stop_enabled': True,
            'trailing_stop_pips': 10,
            'break_even_pips': 10,  # Move SL to breakeven after 10 pips profit
            'session_filter': True,  # Only trade during active sessions
            'news_filter': True,  # Avoid trading during high-impact news
            'weekend_close': True,  # Close positions before weekend
            'hedging_enabled': False,  # Allow hedging positions
            'scalping_mode': False,  # Quick scalping trades
            'swing_mode': True,  # Longer-term swing trades
        }
        
        # Trading state
        self.active_trades = {}
        self.trade_history = []
        self.hourly_trade_count = 0
        self.last_trade_time = None
        self.last_hour_reset = datetime.now()
        self.trading_lock = threading.Lock()
        
        # Professional forex risk management
        self.daily_loss_limit = 500.0  # Maximum daily loss in account currency
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.max_concurrent_trades = 3
        self.max_risk_per_pair = 5.0  # Max % risk per currency pair
        self.correlation_limit = 0.7  # Avoid highly correlated pairs
        self.exposure_limits = {
            'USD': 300.0,  # Max USD exposure in lots
            'EUR': 200.0,
            'GBP': 150.0,
            'JPY': 200.0,
            'CHF': 100.0,
            'AUD': 100.0,
            'CAD': 100.0,
            'NZD': 50.0
        }
        
        # Account info
        self.account_balance = self._get_account_balance()
        
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            # Simulate API call to get balance
            # In production, this would make a real API call
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # For demo purposes, return a simulated balance
            return 10000.0  # $10,000 demo account
            
        except Exception as e:
            print(f"Error getting account balance: {e}")
            return 0.0
    
    def update_settings(self, new_settings: Dict):
        """Update trading settings"""
        with self.trading_lock:
            self.settings.update(new_settings)
    
    def execute_trade(self, signal: str, confidence: float, analysis_data: Dict) -> Optional[Dict]:
        """Execute a trade based on AI analysis"""
        if not self._can_trade(signal, confidence):
            return None
        
        try:
            with self.trading_lock:
                # Reset hourly counter if needed
                self._reset_hourly_counter()
                
                # Prepare trade parameters
                trade_params = self._prepare_trade_params(signal, confidence, analysis_data)
                
                if not trade_params:
                    return None
                
                # Execute the trade
                trade_result = self._send_trade_order(trade_params)
                
                if trade_result and trade_result.get('success'):
                    # Update trading state
                    self._update_trading_state(trade_result)
                    
                    return {
                        'status': 'SUCCESS',
                        'trade_id': trade_result.get('trade_id'),
                        'entry_price': trade_result.get('entry_price'),
                        'amount': trade_params['amount'],
                        'signal': signal,
                        'confidence': confidence,
                        'timestamp': datetime.now(),
                        'profit_loss': 0.0  # Initial P&L
                    }
                else:
                    return {
                        'status': 'FAILED',
                        'error': trade_result.get('error', 'Unknown error'),
                        'signal': signal,
                        'confidence': confidence,
                        'timestamp': datetime.now(),
                        'profit_loss': 0.0
                    }
                    
        except Exception as e:
            print(f"Error executing trade: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'profit_loss': 0.0
            }
    
    def _can_trade(self, signal: str, confidence: float) -> bool:
        """Check if trading is allowed based on current conditions"""
        # Check if trading is enabled
        if not self.settings['enabled']:
            return False
        
        # Check confidence threshold
        if confidence < self.settings['confidence_threshold']:
            return False
        
        # Check hourly trade limit
        if self.hourly_trade_count >= self.settings['max_trades_per_hour']:
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return False
        
        # Check maximum concurrent trades
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False
        
        # Check minimum time between trades (prevent spam)
        if self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time
            if time_since_last.total_seconds() < 30:  # 30 seconds minimum
                return False
        
        # Check account balance
        required_margin = self.settings['amount'] * 0.01  # 1% margin requirement
        if self.account_balance < required_margin:
            return False
        
        return True
    
    def _prepare_trade_params(self, signal: str, confidence: float, analysis_data: Dict) -> Optional[Dict]:
        """Prepare trade parameters based on analysis"""
        try:
            # Get current market price
            current_price = analysis_data.get('price_data', {}).get('current_price')
            spread = analysis_data.get('price_data', {}).get('spread', 0.0003)
            
            if not current_price:
                return None
            
            # Check spread
            spread_pips = spread * 10000  # Convert to pips for major pairs
            if spread_pips > self.settings['max_spread']:
                return None
            
            # Calculate position size based on confidence and risk
            base_amount = self.settings['amount']
            confidence_multiplier = min(confidence * 1.5, 1.0)  # Scale with confidence
            position_size = base_amount * confidence_multiplier
            
            # Calculate stop loss and take profit
            pip_value = 0.0001  # For major pairs
            
            if signal == 'BUY':
                entry_price = analysis_data.get('price_data', {}).get('ask', current_price)
                stop_loss = entry_price - (self.settings['stop_loss_pips'] * pip_value)
                take_profit = entry_price + (self.settings['take_profit_pips'] * pip_value)
            else:  # SELL
                entry_price = analysis_data.get('price_data', {}).get('bid', current_price)
                stop_loss = entry_price + (self.settings['stop_loss_pips'] * pip_value)
                take_profit = entry_price - (self.settings['take_profit_pips'] * pip_value)
            
            return {
                'symbol': self.settings['currency_pair'],
                'side': signal.lower(),
                'amount': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'analysis_data': analysis_data
            }
            
        except Exception as e:
            print(f"Error preparing trade parameters: {e}")
            return None
    
    def _send_trade_order(self, trade_params: Dict) -> Optional[Dict]:
        """Send trade order to broker API"""
        try:
            # In a real implementation, this would make an API call to your broker
            # For demo purposes, we'll simulate the trade execution
            
            # Simulate API endpoint
            endpoint = f"{self.base_url}/orders"
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'symbol': trade_params['symbol'],
                'side': trade_params['side'],
                'type': 'market',
                'amount': trade_params['amount'],
                'stop_loss': trade_params['stop_loss'],
                'take_profit': trade_params['take_profit']
            }
            
            # Simulate successful trade execution
            # In production, uncomment the following line:
            # response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            
            # Simulated response for demo
            simulated_response = {
                'success': True,
                'trade_id': f"T{int(time.time())}{len(self.active_trades)}",
                'entry_price': trade_params['entry_price'],
                'amount': trade_params['amount'],
                'timestamp': datetime.now().isoformat()
            }
            
            return simulated_response
            
        except Exception as e:
            print(f"Error sending trade order: {e}")
            return {'success': False, 'error': str(e)}
    
    def _update_trading_state(self, trade_result: Dict):
        """Update internal trading state after successful trade"""
        trade_id = trade_result['trade_id']
        
        # Add to active trades
        self.active_trades[trade_id] = {
            'id': trade_id,
            'entry_price': trade_result['entry_price'],
            'amount': trade_result['amount'],
            'entry_time': datetime.now(),
            'status': 'OPEN'
        }
        
        # Update counters
        self.hourly_trade_count += 1
        self.last_trade_time = datetime.now()
        
        # Add to history
        self.trade_history.append(trade_result)
    
    def _reset_hourly_counter(self):
        """Reset hourly trade counter if hour has passed"""
        now = datetime.now()
        if (now - self.last_hour_reset).total_seconds() >= 3600:  # 1 hour
            self.hourly_trade_count = 0
            self.last_hour_reset = now
    
    def close_trade(self, trade_id: str, reason: str = "Manual") -> Optional[Dict]:
        """Close an active trade"""
        if trade_id not in self.active_trades:
            return None
        
        try:
            with self.trading_lock:
                trade = self.active_trades[trade_id]
                
                # Simulate closing the trade
                # In production, make API call to close position
                
                # Calculate P&L (simplified)
                current_price = trade['entry_price'] + (0.0001 * (1 if reason == "Take Profit" else -1))
                pnl = (current_price - trade['entry_price']) * trade['amount'] * 10000
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
                close_result = {
                    'trade_id': trade_id,
                    'close_time': datetime.now(),
                    'close_reason': reason,
                    'profit_loss': pnl,
                    'status': 'CLOSED'
                }
                
                return close_result
                
        except Exception as e:
            print(f"Error closing trade: {e}")
            return None
    
    def get_trading_status(self) -> Dict:
        """Get current trading status and statistics"""
        return {
            'enabled': self.settings['enabled'],
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'active_trades_count': len(self.active_trades),
            'hourly_trades_count': self.hourly_trade_count,
            'trades_remaining_this_hour': max(0, self.settings['max_trades_per_hour'] - self.hourly_trade_count),
            'daily_loss_limit': self.daily_loss_limit,
            'can_trade': self._can_trade('BUY', 1.0),  # Test with max confidence
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
        }
    
    def get_active_trades(self) -> List[Dict]:
        """Get list of active trades"""
        return list(self.active_trades.values())
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        return self.trade_history[-limit:]
    
    def emergency_stop(self):
        """Emergency stop - close all trades and disable trading"""
        with self.trading_lock:
            # Close all active trades
            for trade_id in list(self.active_trades.keys()):
                self.close_trade(trade_id, "Emergency Stop")
            
            # Disable trading
            self.settings['enabled'] = False
            
            print("Emergency stop activated - all trades closed and trading disabled")
