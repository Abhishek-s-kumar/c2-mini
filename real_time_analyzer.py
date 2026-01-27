#!/usr/bin/env python3
"""
Real-time C2 Beacon Detector
Connects to PostgreSQL and analyzes Zeek logs for beaconing patterns
"""

import psycopg2
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import configparser
import sys

class RealTimeC2Detector:
    def __init__(self, config_file='config/database.conf'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.config['postgresql']['host'],
                database=self.config['postgresql']['database'],
                user=self.config['postgresql']['user'],
                password=self.config['postgresql']['password'],
                port=int(self.config['postgresql']['port'])
            )
            print("[+] Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False
    
    def get_recent_connections(self, minutes=5):
        """Get recent connection logs"""
        query = """
        SELECT ts, id_orig_h, id_resp_h, resp_bytes, orig_bytes
        FROM conn_log 
        WHERE ts >= NOW() - INTERVAL '%s minutes'
        ORDER BY ts
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(minutes,))
            return df
        except Exception as e:
            print(f"[-] Query failed: {e}")
            return pd.DataFrame()
    
    def analyze_host(self, host_ip, df):
        """Analyze a specific host for beaconing patterns"""
        host_df = df[df['id_orig_h'] == host_ip].copy()
        
        if len(host_df) < 10:
            return None
        
        # Create time series
        host_df['ts'] = pd.to_datetime(host_df['ts'])
        host_df.set_index('ts', inplace=True)
        
        # Resample to 1-second intervals
        time_series = host_df['resp_bytes'].resample('1S').sum().fillna(0)
        
        # Calculate metrics (simplified P-Score)
        if len(time_series) > 20:
            # FFT analysis
            N = len(time_series)
            yf = np.fft.rfft(time_series.values)
            xf = np.fft.rfftfreq(N, d=1)
            magnitude = 2.0/N * np.abs(yf)
            
            # Skip DC component
            if len(magnitude) > 1:
                fft_peak = np.max(magnitude[1:]) / np.max(magnitude)
            else:
                fft_peak = 0
            
            # Simple autocorrelation
            autocorr = np.correlate(time_series.values, time_series.values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr_max = np.max(autocorr[1:20]) / autocorr[0] if autocorr[0] > 0 else 0
            
            # Entropy
            hist, _ = np.histogram(time_series.values, bins=10, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist))
                entropy_norm = entropy / np.log2(len(hist))
            else:
                entropy_norm = 0
            
            # P-Score
            alpha, beta, gamma = 0.4, 0.4, 0.2
            p_score = (alpha * fft_peak + beta * autocorr_max + gamma * (1 - entropy_norm))
            
            return {
                'host': host_ip,
                'p_score': float(p_score),
                'fft_peak': float(fft_peak),
                'autocorr_max': float(autocorr_max),
                'entropy_norm': float(entropy_norm),
                'samples': len(time_series),
                'detected': p_score > 0.7
            }
        
        return None

def main():
    detector = RealTimeC2Detector()
    
    if not detector.connect():
        print("[-] Exiting")
        return
    
    print("[+] Starting real-time analysis...")
    print("[+] Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Get recent data
            df = detector.get_recent_connections(minutes=2)
            
            if not df.empty:
                # Get unique hosts
                hosts = df['id_orig_h'].unique()[:5]  # Analyze top 5 hosts
                
                print(f"\n{'='*50}")
                print(f"Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total connections: {len(df)}")
                print('='*50)
                
                for host in hosts:
                    result = detector.analyze_host(host, df)
                    if result:
                        status = "⚠️ BEACON DETECTED" if result['detected'] else "✓ Normal"
                        print(f"{host}: P-Score={result['p_score']:.3f} - {status}")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n[+] Analysis stopped")

if __name__ == "__main__":
    main()
