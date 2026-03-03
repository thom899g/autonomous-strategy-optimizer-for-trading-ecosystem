"""
Market Data Ingestion Module
Responsible for collecting, validating, and preparing market data for strategy analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import ccxt
from firebase_admin import firestore

logger = logging.getLogger(__name__)


class MarketDataIngestor:
    """
    Robust market data collector with validation and caching mechanisms
    """
    
    def __init__(self, exchange_id: str = 'binance', timeframe: str = '1h'):
        """
        Initialize data ingestor with exchange connection
        
        Args:
            exchange_id: Exchange identifier (binance, coinbase, etc.)
            timeframe: Default candle timeframe
        """
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.exchange = None
        self.db = None
        self._initialize_exchange()
        self._initialize_firestore()
        
    def _initialize_exchange(self) -> None:
        """Initialize CCXT exchange connection with error handling"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000
            })
            logger.info(f"Successfully initialized {self.exchange_id} exchange")
        except AttributeError:
            logger.error(f"Exchange {self.exchange_id} not found in CCXT")
            raise ValueError(f"Unsupported exchange: {self.exchange_id}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise
            
    def _initialize_firestore(self) -> None:
        """Initialize Firebase connection for data caching"""
        try:
            import firebase_admin
            from firebase_admin import credentials
            
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                cred = credentials.Certificate('service-account-key.json')
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            logger.info("Firestore client initialized successfully")
        except FileNotFoundError:
            logger.warning("Firebase service account key not found. Using local cache only.")
            self.db = None
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            self.db = None
    
    def fetch_ohlcv(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with robust error handling and validation
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            since: Start datetime (default: 30 days ago)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data and validation metadata
        """
        try:
            # Validate inputs
            if not self._validate_symbol(symbol):
                raise ValueError(f"Invalid symbol format: {symbol}")
            
            # Calculate default start time if not provided
            if since is None:
                since = datetime.now() - timedelta(days=30)
            
            # Convert datetime to milliseconds timestamp
            since_timestamp = int(since.timestamp() * 1000)
            
            # Check cache first
            cached_data = self._get_cached_data(symbol, since_timestamp)
            if cached_data is not None:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
            
            # Fetch from exchange
            logger.info(f"Fetching {symbol} data from {self.exchange_id}")
            raw_data = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                since=since_timestamp,
                limit=limit
            )
            
            # Validate response
            if not raw_data or len(raw_data) == 0:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                raw_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add derived features
            df = self._add_technical_features(df)
            
            # Validate data quality
            validation_result = self._validate_ohlcv_data(df)
            if not validation_result['is_valid']:
                logger.warning(f"Data validation issues: {validation_result['issues']}")
            
            # Cache the data
            self._cache_data(symbol, df, since_timestamp)
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {str(e)}")
            # Return empty DataFrame with error metadata
            return pd.DataFrame(columns=['error', 'timestamp'])
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {str(e)}")
            return pd.DataFrame(columns=['error', 'timestamp'])
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate trading pair symbol format"""
        if not isinstance(symbol, str):
            return False
        if '/' not in symbol:
            return False
        return True
    
    def _get_cached_data(self, symbol: str, since_timestamp: int) -> Optional[pd.DataFrame]:
        """Retrieve cached data from Firestore if available"""
        if self.db is None:
            return None
            
        try:
            # Create cache key
            cache_key = f"{self.exchange_id}_{symbol}_{self.timeframe}_{since_timestamp}"
            
            doc_ref = self.db.collection('market_data_cache').document(cache_key)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                # Check if cache is still valid (less than 5 minutes old)
                cache_time = data.get('cached_at')
                if cache_time:
                    cache_age = datetime.now() - cache_time
                    if cache_age < timedelta(minutes=5):
                        # Convert list back to DataFrame
                        df_data = data.get('data', [])
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                            return df
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    def _cache_data(self, symbol: str, df: pd.DataFrame, since_timestamp: int) -> None:
        """Cache OHLCV data to Firestore"""
        if self.db is None or df.empty:
            return
            
        try:
            # Prepare data for caching
            cache_data = {
                'exchange': self.exchange_id,
                'symbol': symbol,
                'timeframe': self.timeframe,
                'since_timestamp': since_timestamp,
                'c