import os
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, List, Optional
import json

Base = declarative_base()

class TradingSession(Base):
    """Store trading session information"""
    __tablename__ = 'trading_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), unique=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    total_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    session_type = Column(String(20))  # LONDON, NEW_YORK, SYDNEY, ASIAN
    volatility_level = Column(String(10))  # LOW, MEDIUM, HIGH
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketData(Base):
    """Store real-time market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    volume = Column(Float)
    session = Column(String(20))
    volatility = Column(String(10))
    news_impact = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisResults(Base):
    """Store AI analysis results"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(10), nullable=False)
    signal = Column(String(10))  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    pattern = Column(String(50))
    price_data = Column(JSON)
    technical_indicators = Column(JSON)
    candlestick_analysis = Column(JSON)
    trend_analysis = Column(JSON)
    support_resistance = Column(JSON)
    frame_hash = Column(String(64))  # Hash of analyzed frame
    created_at = Column(DateTime, default=datetime.utcnow)

class Trades(Base):
    """Store executed trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    session_id = Column(String(50), nullable=False)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # BUY, SELL
    lot_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    status = Column(String(10), default='OPEN')  # OPEN, CLOSED, CANCELLED
    pnl = Column(Float, default=0.0)
    pnl_pips = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    exit_reason = Column(String(20))  # TP, SL, MANUAL, TIME
    confidence = Column(Float)
    analysis_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class TradingSettings(Base):
    """Store user trading preferences and settings"""
    __tablename__ = 'trading_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default='default_user')
    setting_name = Column(String(50), nullable=False)
    setting_value = Column(Text)
    setting_type = Column(String(20))  # STRING, FLOAT, INTEGER, BOOLEAN, JSON
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)

class RiskManagement(Base):
    """Store risk management data"""
    __tablename__ = 'risk_management'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    account_balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin_used = Column(Float, default=0.0)
    margin_free = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    weekly_pnl = Column(Float, default=0.0)
    monthly_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    risk_percentage = Column(Float, default=0.0)
    active_trades = Column(Integer, default=0)
    currency_exposure = Column(JSON)  # Exposure per currency
    created_at = Column(DateTime, default=datetime.utcnow)

class VideoRecordings(Base):
    """Store video recording metadata"""
    __tablename__ = 'video_recordings'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    session_id = Column(String(50))
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    frame_count = Column(Integer)
    file_size_bytes = Column(Integer)
    file_path = Column(String(500))
    recording_type = Column(String(20))  # FULL, HIGHLIGHT, TRADE_SPECIFIC
    associated_trades = Column(JSON)
    compression_settings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        self.create_tables()
        
        # Initialize default settings
        self.initialize_default_settings()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("Database tables created successfully")
        except Exception as e:
            print(f"Error creating database tables: {e}")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def initialize_default_settings(self):
        """Initialize default trading settings"""
        session = self.get_session()
        try:
            # Check if settings exist
            existing_settings = session.query(TradingSettings).filter_by(user_id='default_user').first()
            if not existing_settings:
                default_settings = [
                    TradingSettings(setting_name='lot_size', setting_value='0.10', setting_type='FLOAT', description='Default lot size'),
                    TradingSettings(setting_name='stop_loss_pips', setting_value='15', setting_type='INTEGER', description='Default stop loss in pips'),
                    TradingSettings(setting_name='take_profit_pips', setting_value='30', setting_type='INTEGER', description='Default take profit in pips'),
                    TradingSettings(setting_name='max_spread_pips', setting_value='2.5', setting_type='FLOAT', description='Maximum allowed spread'),
                    TradingSettings(setting_name='risk_per_trade_percent', setting_value='2.0', setting_type='FLOAT', description='Risk percentage per trade'),
                    TradingSettings(setting_name='currency_pair', setting_value='EURUSD', setting_type='STRING', description='Default currency pair'),
                    TradingSettings(setting_name='confidence_threshold', setting_value='0.75', setting_type='FLOAT', description='Minimum confidence for trades'),
                    TradingSettings(setting_name='session_filter', setting_value='true', setting_type='BOOLEAN', description='Filter trades by session'),
                    TradingSettings(setting_name='news_filter', setting_value='true', setting_type='BOOLEAN', description='Avoid trading during news'),
                ]
                
                for setting in default_settings:
                    session.add(setting)
                session.commit()
                print("Default settings initialized")
        except Exception as e:
            print(f"Error initializing default settings: {e}")
            session.rollback()
        finally:
            session.close()
    
    def save_market_data(self, symbol: str, bid: float, ask: float, spread: float, 
                        session_name: str = None, volatility: str = None, news_impact: str = None):
        """Save market data to database"""
        session = self.get_session()
        try:
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid=bid,
                ask=ask,
                spread=spread,
                session=session_name,
                volatility=volatility,
                news_impact=news_impact
            )
            session.add(market_data)
            session.commit()
            return market_data.id
        except Exception as e:
            print(f"Error saving market data: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def save_analysis_result(self, analysis_data: Dict):
        """Save AI analysis result to database"""
        session = self.get_session()
        try:
            analysis = AnalysisResults(
                session_id=analysis_data.get('session_id', 'default'),
                timestamp=analysis_data.get('timestamp', datetime.utcnow()),
                symbol=analysis_data.get('symbol', 'EURUSD'),
                signal=analysis_data.get('signal', 'HOLD'),
                confidence=analysis_data.get('confidence', 0.0),
                pattern=analysis_data.get('pattern', 'None'),
                price_data=analysis_data.get('price_data', {}),
                technical_indicators=analysis_data.get('indicators', {}),
                candlestick_analysis=analysis_data.get('candlestick_analysis', {}),
                trend_analysis=analysis_data.get('trend_analysis', {}),
                support_resistance=analysis_data.get('support_resistance', {}),
                frame_hash=analysis_data.get('frame_hash')
            )
            session.add(analysis)
            session.commit()
            return analysis.id
        except Exception as e:
            print(f"Error saving analysis result: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        session = self.get_session()
        try:
            trade = Trades(
                trade_id=trade_data.get('trade_id'),
                session_id=trade_data.get('session_id', 'default'),
                symbol=trade_data.get('symbol', 'EURUSD'),
                side=trade_data.get('side', 'BUY'),
                lot_size=trade_data.get('lot_size', 0.1),
                entry_price=trade_data.get('entry_price'),
                stop_loss=trade_data.get('stop_loss'),
                take_profit=trade_data.get('take_profit'),
                entry_time=trade_data.get('entry_time', datetime.utcnow()),
                confidence=trade_data.get('confidence'),
                analysis_id=trade_data.get('analysis_id')
            )
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            print(f"Error saving trade: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def get_trading_settings(self, user_id: str = 'default_user') -> Dict:
        """Get trading settings for user"""
        session = self.get_session()
        try:
            settings = session.query(TradingSettings).filter_by(user_id=user_id).all()
            settings_dict = {}
            for setting in settings:
                value = setting.setting_value
                if setting.setting_type == 'FLOAT':
                    value = float(value)
                elif setting.setting_type == 'INTEGER':
                    value = int(value)
                elif setting.setting_type == 'BOOLEAN':
                    value = value.lower() == 'true'
                elif setting.setting_type == 'JSON':
                    value = json.loads(value)
                settings_dict[setting.setting_name] = value
            return settings_dict
        except Exception as e:
            print(f"Error getting trading settings: {e}")
            return {}
        finally:
            session.close()
    
    def update_trading_setting(self, setting_name: str, setting_value: str, 
                             setting_type: str = 'STRING', user_id: str = 'default_user'):
        """Update or create trading setting"""
        session = self.get_session()
        try:
            setting = session.query(TradingSettings).filter_by(
                user_id=user_id, setting_name=setting_name
            ).first()
            
            if setting:
                setting.setting_value = str(setting_value)
                setting.setting_type = setting_type
                setting.updated_at = datetime.utcnow()
            else:
                setting = TradingSettings(
                    user_id=user_id,
                    setting_name=setting_name,
                    setting_value=str(setting_value),
                    setting_type=setting_type
                )
                session.add(setting)
            
            session.commit()
            return True
        except Exception as e:
            print(f"Error updating trading setting: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_recent_analysis(self, limit: int = 100) -> List[Dict]:
        """Get recent analysis results"""
        session = self.get_session()
        try:
            results = session.query(AnalysisResults).order_by(
                AnalysisResults.timestamp.desc()
            ).limit(limit).all()
            
            analysis_list = []
            for result in results:
                analysis_list.append({
                    'id': result.id,
                    'timestamp': result.timestamp,
                    'symbol': result.symbol,
                    'signal': result.signal,
                    'confidence': result.confidence,
                    'pattern': result.pattern,
                    'price_data': result.price_data,
                    'technical_indicators': result.technical_indicators
                })
            return analysis_list
        except Exception as e:
            print(f"Error getting recent analysis: {e}")
            return []
        finally:
            session.close()
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history"""
        session = self.get_session()
        try:
            trades = session.query(Trades).order_by(
                Trades.entry_time.desc()
            ).limit(limit).all()
            
            trade_list = []
            for trade in trades:
                trade_list.append({
                    'id': trade.id,
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'lot_size': trade.lot_size,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'status': trade.status,
                    'pnl': trade.pnl,
                    'pnl_pips': trade.pnl_pips,
                    'confidence': trade.confidence
                })
            return trade_list
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return []
        finally:
            session.close()
    
    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        session = self.get_session()
        try:
            # Total trades
            total_trades = session.query(Trades).count()
            
            # Profitable trades
            profitable_trades = session.query(Trades).filter(Trades.pnl > 0).count()
            
            # Total PnL
            total_pnl = session.query(Trades).with_entities(
                session.query(Trades.pnl).filter(Trades.pnl.isnot(None)).all()
            )
            total_pnl_value = sum([trade.pnl for trade in session.query(Trades).all() if trade.pnl])
            
            # Win rate
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Average profit per trade
            avg_profit = total_pnl_value / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'total_pnl': total_pnl_value,
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit
            }
        except Exception as e:
            print(f"Error getting performance stats: {e}")
            return {}
        finally:
            session.close()