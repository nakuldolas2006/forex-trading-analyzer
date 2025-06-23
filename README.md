# Forex Trading Analyzer

A professional AI-powered forex trading application with live screen analysis, automated pattern recognition, and trading execution capabilities.

## Features

### Core Functionality
- **Live Screen Capture**: Captures trading platforms (MT4/MT5, TradingView) in real-time
- **AI Pattern Recognition**: Advanced computer vision analysis for forex trading patterns
- **Automated Trading**: Professional trading engine with comprehensive risk management
- **Real-time Analysis**: Live forex market data processing and signal generation
- **Session Recording**: Video capture of all trading sessions for review

### Professional Trading Features
- **Multiple Currency Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, NZD/USD
- **Risk Management**: Stop loss, take profit, trailing stops, position sizing
- **Session Awareness**: London, New York, Sydney, Asian trading sessions
- **News Filtering**: Avoid trading during high-impact economic events
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Pattern Detection**: Candlestick patterns, support/resistance, trend analysis

### Database Integration
- **PostgreSQL Backend**: Stores all trading data, analysis results, and settings
- **Performance Tracking**: Comprehensive trading statistics and analytics
- **Historical Data**: Complete trade history and pattern analysis records

## Installation

### Prerequisites
- Python 3.11 or higher
- PostgreSQL (optional, for database features)

### Quick Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Usage

### Getting Started
1. Launch the application
2. Configure trading settings in the sidebar
3. Start screen analysis to begin pattern recognition
4. Monitor real-time forex analysis and trading signals

### Configuration Options
- **Lot Size**: Micro (0.01), Mini (0.10), Standard (1.0)
- **Risk Management**: Stop loss, take profit, risk percentage per trade
- **Trading Sessions**: Filter trades by active market sessions
- **Pattern Sensitivity**: Adjust AI confidence thresholds

### Demo Mode
The application runs in demo mode by default, simulating realistic forex trading interfaces. This allows testing without actual broker connections.

## Technical Architecture

### Components
- **Screen Capture**: Cross-platform screen recording with headless support
- **AI Analyzer**: OpenCV-based pattern recognition engine
- **Trading Engine**: Professional trade execution with risk controls
- **Database Layer**: SQLAlchemy models for data persistence
- **Video Manager**: Session recording and playback system

### Supported Platforms
- **Trading Platforms**: MT4, MT5, TradingView, WebTrader
- **Operating Systems**: Windows, Linux, macOS
- **Deployment**: Local, cloud, containerized environments

## Security & Risk Management

### Built-in Protections
- Daily loss limits
- Maximum trades per hour/day
- Spread filtering
- Currency exposure limits
- Emergency stop functionality

### Best Practices
- Always test with demo accounts first
- Implement proper API key security
- Monitor risk metrics continuously
- Review trading performance regularly

## Database Schema

The application uses PostgreSQL with comprehensive tables for:
- Trading sessions and performance
- Real-time market data
- AI analysis results
- Trade execution records
- Risk management metrics
- Video recording metadata

## Development

### File Structure
```
├── app.py                 # Main Streamlit application
├── ai_analyzer.py         # AI pattern recognition
├── screen_capture.py      # Screen capture system
├── trading_engine.py      # Trading logic and execution
├── database.py           # Database models and operations
├── video_manager.py      # Video recording management
├── utils.py              # Utility functions
└── .streamlit/           # Streamlit configuration
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## Disclaimer

This software is for educational and demonstration purposes. Trading forex involves substantial risk of loss. Never trade with money you cannot afford to lose. Always test thoroughly with demo accounts before live trading.

## License

This project is provided as-is for educational purposes. Please ensure compliance with local regulations regarding automated trading systems.

## Support

For questions and support, please refer to the documentation or create an issue in the repository.