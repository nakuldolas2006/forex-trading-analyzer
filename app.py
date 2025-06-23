import streamlit as st
import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
import pandas as pd
from screen_capture import ScreenCapture
from ai_analyzer import AIAnalyzer
from trading_engine import TradingEngine
from video_manager import VideoManager
from utils import format_timestamp, create_directories

# Initialize session state
if 'screen_capture' not in st.session_state:
    st.session_state.screen_capture = None
if 'ai_analyzer' not in st.session_state:
    st.session_state.ai_analyzer = None
if 'trading_engine' not in st.session_state:
    st.session_state.trading_engine = None
if 'video_manager' not in st.session_state:
    st.session_state.video_manager = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'trading_log' not in st.session_state:
    st.session_state.trading_log = []

def initialize_components():
    """Initialize all application components"""
    try:
        if not st.session_state.screen_capture:
            st.session_state.screen_capture = ScreenCapture()
        if not st.session_state.ai_analyzer:
            st.session_state.ai_analyzer = AIAnalyzer()
        if not st.session_state.trading_engine:
            api_key = os.getenv("TRADING_API_KEY", "demo_key")
            st.session_state.trading_engine = TradingEngine(api_key)
        if not st.session_state.video_manager:
            st.session_state.video_manager = VideoManager()
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

def main_analysis_loop():
    """Main loop for screen analysis and trading"""
    while st.session_state.is_running:
        try:
            # Capture screen
            frame = st.session_state.screen_capture.capture_frame()
            if frame is not None:
                # Record frame
                st.session_state.video_manager.add_frame(frame)
                
                # Analyze frame
                analysis = st.session_state.ai_analyzer.analyze_frame(frame)
                
                if analysis:
                    st.session_state.analysis_results.append({
                        'timestamp': datetime.now(),
                        'confidence': analysis.get('confidence', 0),
                        'signal': analysis.get('signal', 'HOLD'),
                        'pattern': analysis.get('pattern', 'None'),
                        'price_data': analysis.get('price_data', {})
                    })
                    
                    # Execute trading decision
                    if analysis.get('signal') in ['BUY', 'SELL'] and analysis.get('confidence', 0) > 0.7:
                        trade_result = st.session_state.trading_engine.execute_trade(
                            signal=analysis['signal'],
                            confidence=analysis['confidence'],
                            analysis_data=analysis
                        )
                        
                        if trade_result:
                            st.session_state.trading_log.append({
                                'timestamp': datetime.now(),
                                'action': analysis['signal'],
                                'confidence': analysis['confidence'],
                                'result': trade_result.get('status', 'Unknown'),
                                'profit_loss': trade_result.get('profit_loss', 0)
                            })
            
            time.sleep(0.1)  # 10 FPS analysis rate
            
        except Exception as e:
            st.error(f"Error in analysis loop: {str(e)}")
            time.sleep(1)

def main():
    st.title("🤖 AI Trading Screen Analyzer")
    st.markdown("**Live screen recording with AI-powered video analysis and automated trading execution**")
    
    # Create necessary directories
    create_directories()
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Screen capture settings
        st.subheader("Screen Capture")
        monitor_selection = st.selectbox(
            "Select Monitor",
            options=list(range(len(st.session_state.screen_capture.get_monitors()))),
            format_func=lambda x: f"Monitor {x+1}"
        )
        
        capture_region = st.checkbox("Custom Capture Region")
        if capture_region:
            col1, col2 = st.columns(2)
            with col1:
                x_start = st.number_input("X Start", min_value=0, value=0)
                y_start = st.number_input("Y Start", min_value=0, value=0)
            with col2:
                width = st.number_input("Width", min_value=100, value=800)
                height = st.number_input("Height", min_value=100, value=600)
            
            st.session_state.screen_capture.set_region(x_start, y_start, width, height)
        
        # AI Analysis settings
        st.subheader("AI Analysis")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.1)
        analysis_frequency = st.slider("Analysis Frequency (FPS)", 1, 30, 10)
        
        # Forex Trading settings
        st.subheader("Forex Trading")
        trading_enabled = st.checkbox("Enable Auto Trading", value=False)
        
        # Lot size selection
        lot_size = st.selectbox("Lot Size", 
                               options=[0.01, 0.10, 0.50, 1.0], 
                               format_func=lambda x: f"{x} {'(Micro)' if x == 0.01 else '(Mini)' if x == 0.10 else '(Standard)' if x == 1.0 else '(Mid)'}")
        
        # Currency pair selection
        currency_pair = st.selectbox("Currency Pair", 
                                   options=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"])
        
        # Risk management
        risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.1)
        stop_loss_pips = st.number_input("Stop Loss (Pips)", min_value=5, max_value=100, value=15)
        take_profit_pips = st.number_input("Take Profit (Pips)", min_value=10, max_value=200, value=30)
        max_spread_pips = st.slider("Max Spread (Pips)", 0.5, 5.0, 2.5, 0.1)
        
        # Trading session filters
        st.write("**Session Filters**")
        session_filter = st.checkbox("Only Trade Active Sessions", value=True)
        news_filter = st.checkbox("Avoid High-Impact News", value=True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_trades_per_hour = st.number_input("Max Trades/Hour", min_value=1, max_value=20, value=5)
            trailing_stop = st.checkbox("Enable Trailing Stop", value=True)
            break_even_pips = st.number_input("Break-Even Pips", min_value=5, max_value=50, value=10)
        
        st.session_state.trading_engine.update_settings({
            'enabled': trading_enabled,
            'lot_size': lot_size,
            'currency_pair': currency_pair,
            'risk_per_trade_percent': risk_per_trade,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'max_spread_pips': max_spread_pips,
            'max_trades_per_hour': max_trades_per_hour,
            'confidence_threshold': confidence_threshold,
            'session_filter': session_filter,
            'news_filter': news_filter,
            'trailing_stop_enabled': trailing_stop,
            'break_even_pips': break_even_pips
        })
    
    # Main control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Analysis", disabled=st.session_state.is_running):
            st.session_state.is_running = True
            st.session_state.video_manager.start_recording()
            
            # Start analysis thread
            analysis_thread = threading.Thread(target=main_analysis_loop, daemon=True)
            analysis_thread.start()
            
            st.success("Analysis started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Analysis", disabled=not st.session_state.is_running):
            st.session_state.is_running = False
            st.session_state.video_manager.stop_recording()
            st.success("Analysis stopped!")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Data"):
            st.session_state.analysis_results = []
            st.session_state.trading_log = []
            st.success("Data cleared!")
            st.rerun()
    
    # Live preview
    if st.session_state.is_running:
        st.subheader("📺 Live Screen Preview")
        preview_placeholder = st.empty()
        
        # Get current frame for preview
        current_frame = st.session_state.screen_capture.capture_frame()
        if current_frame is not None:
            # Resize for display
            display_frame = cv2.resize(current_frame, (800, 450))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(display_frame_rgb, caption="Live Screen Capture")
    
    # Forex Market Analysis
    st.subheader("📊 Forex Market Analysis")
    
    if st.session_state.analysis_results:
        # Latest analysis
        latest = st.session_state.analysis_results[-1]
        price_data = latest.get('price_data', {})
        
        # Current market status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Signal", latest['signal'], 
                     delta=f"{latest['confidence']:.1%} confidence")
        with col2:
            current_price = price_data.get('current_price', 0)
            st.metric("EUR/USD", f"{current_price:.5f}", 
                     delta=f"{price_data.get('spread_pips', 0):.1f} pips spread")
        with col3:
            session = price_data.get('trading_session', 'Unknown')
            volatility = price_data.get('market_volatility', 'Unknown')
            st.metric("Session", session, delta=f"{volatility} volatility")
        with col4:
            news_impact = price_data.get('news_impact', 'NONE')
            st.metric("News Impact", news_impact)
        
        # Market Watch Panel
        st.write("**Major Currency Pairs**")
        if 'market_pairs' in price_data:
            pairs_data = []
            for pair, data in price_data['market_pairs'].items():
                pairs_data.append({
                    'Pair': pair,
                    'Price': f"{data['price']:.5f}",
                    'Bid': f"{data['bid']:.5f}",
                    'Ask': f"{data['ask']:.5f}",
                    'Spread': f"{data['spread_pips']:.1f}",
                    'Change': f"{data['change_pips']:+.1f}"
                })
            
            df_pairs = pd.DataFrame(pairs_data)
            st.dataframe(df_pairs, use_container_width=True)
        
        # Technical Analysis
        st.write("**Technical Indicators**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("RSI: 56.2 (Neutral)")
            st.write("MACD: +0.0012 (Bullish)")
        with col2:
            st.write("SMA20: 1.0841")
            st.write("Bollinger Bands: 1.0855")
        with col3:
            st.write("Support: 1.0820")
            st.write("Resistance: 1.0860")
        
        # Pattern Recognition
        pattern_info = latest.get('pattern', 'None')
        if pattern_info != 'None':
            st.info(f"Pattern Detected: {pattern_info}")
        
        # Analysis confidence chart
        if len(st.session_state.analysis_results) > 1:
            df_analysis = pd.DataFrame(st.session_state.analysis_results[-50:])
            st.line_chart(df_analysis.set_index('timestamp')['confidence'])
    else:
        st.info("Start analysis to see forex market data and signals.")
    
    # Trading log
    st.subheader("💰 Trading Log")
    
    if st.session_state.trading_log:
        df_trades = pd.DataFrame(st.session_state.trading_log)
        
        # Trading summary
        total_trades = len(df_trades)
        total_pnl = df_trades['profit_loss'].sum()
        win_rate = (df_trades['profit_loss'] > 0).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Total P&L", f"${total_pnl:.2f}")
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Recent trades table
        st.dataframe(
            df_trades[['timestamp', 'action', 'confidence', 'result', 'profit_loss']].tail(20),
            use_container_width=True
        )
    else:
        st.info("No trades executed yet.")
    
    # Video recordings
    st.subheader("🎥 Recorded Videos")
    
    recordings = st.session_state.video_manager.get_recordings()
    if recordings:
        selected_recording = st.selectbox("Select Recording", recordings)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Play Recording"):
                st.session_state.video_manager.play_recording(selected_recording)
        with col2:
            if st.button("📥 Download Recording"):
                st.session_state.video_manager.download_recording(selected_recording)
    else:
        st.info("No recordings available.")
    
    # Status indicators
    with st.container():
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if st.session_state.is_running else "🔴"
            st.markdown(f"{status_color} **Analysis**: {'Running' if st.session_state.is_running else 'Stopped'}")
        
        with col2:
            recording_status = "🟢" if st.session_state.video_manager and st.session_state.video_manager.is_recording else "🔴"
            st.markdown(f"{recording_status} **Recording**: {'Active' if st.session_state.video_manager and st.session_state.video_manager.is_recording else 'Inactive'}")
        
        with col3:
            trading_status = "🟢" if trading_enabled else "🔴"
            st.markdown(f"{trading_status} **Trading**: {'Enabled' if trading_enabled else 'Disabled'}")
        
        with col4:
            ai_status = "🟢" if st.session_state.ai_analyzer else "🔴"
            st.markdown(f"{ai_status} **AI Engine**: {'Ready' if st.session_state.ai_analyzer else 'Error'}")

if __name__ == "__main__":
    main()
