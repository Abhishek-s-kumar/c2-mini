#!/usr/bin/env python3
"""
Continuous C2 Beacon Monitoring Service
FIXED: JSON serialization for numpy types
"""

import time
import json
import logging
from datetime import datetime
from real_time_analyzer import RealTimeC2Detector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/c2_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class C2Monitor:
    def __init__(self):
        self.detector = RealTimeC2Detector()
        self.alerts = []
        
    def start(self, interval_seconds=60):
        """Start continuous monitoring"""
        if not self.detector.connect():
            logger.error("Failed to connect to database")
            return
        
        logger.info(f"Starting C2 beacon monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                self.run_detection_cycle()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in detection cycle: {e}")
                time.sleep(10)
    
    def run_detection_cycle(self):
        """Run one detection cycle"""
        try:
            # Get recent data
            df = self.detector.get_recent_connections(minutes=5)
            
            if df.empty:
                logger.warning("No data available")
                return
            
            # Analyze each host
            for host in df['id_orig_h'].unique()[:10]:
                result = self.detector.analyze_host(host, df)
                
                if result and result['detected']:
                    # Create alert (FIXED: Convert numpy types to Python types for JSON serialization)
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'host': str(result['host']),
                        'p_score': float(result['p_score']),
                        'details': {
                            'host': str(result['host']),
                            'p_score': float(result['p_score']),
                            'fft_peak': float(result['fft_peak']),
                            'autocorr_max': float(result['autocorr_max']),
                            'entropy_norm': float(result['entropy_norm']),
                            'samples': int(result['samples']),
                            'detected': bool(result['detected'])
                        }
                    }
                    
                    self.alerts.append(alert)
                    
                    # Log alert
                    logger.warning(
                        f"C2 BEACON DETECTED - Host: {result['host']}, "
                        f"P-Score: {result['p_score']:.3f}"
                    )
                    
                    # Save alert to file
                    with open('output/alerts.json', 'a') as f:
                        json.dump(alert, f)
                        f.write('\n')
        
        except Exception as e:
            logger.error(f"Detection error: {e}")

if __name__ == "__main__":
    monitor = C2Monitor()
    monitor.start(interval_seconds=30)