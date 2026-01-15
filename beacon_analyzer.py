#!/usr/bin/env python3
"""
C2 Beacon Detection Analyzer - FIXED VERSION
Parses Suricata logs, creates time series, applies FFT, and visualizes results
Fixed: DC component issue and array broadcasting error
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BeaconAnalyzer:
    def __init__(self, log_file='c2_beacon.log'):
        """
        Initialize Beacon Analyzer
        
        Args:
            log_file: Path to Suricata log file
        """
        self.log_file = log_file
        self.df = None
        self.beacon_data = None
        self.time_series = None
        
    def parse_logs(self):
        """Parse Suricata JSON logs"""
        print("Parsing log file...")
        
        records = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        self.df = pd.DataFrame(records)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"Parsed {len(self.df)} records")
        print(f"Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        # Extract beacon-related alerts
        self.beacon_data = self.df[self.df['alert'].apply(
            lambda x: 'Beacon' in str(x.get('signature', '')) if isinstance(x, dict) else False
        )].copy()
        
        print(f"Found {len(self.beacon_data)} beacon-related alerts")
        
        return self.df
    
    def create_time_series(self, time_resolution='1S'):
        """
        Create time series from beacon data
        
        Args:
            time_resolution: Time bin size (e.g., '1S' for 1 second)
        """
        if self.beacon_data is None:
            self.parse_logs()
        
        print("\nCreating time series...")
        
        # Group by time bins
        self.beacon_data.set_index('timestamp', inplace=True)
        
        # Create multiple time series
        time_series = {}
        
        # 1. Count of beacon alerts per time bin
        time_series['alert_count'] = self.beacon_data.resample(time_resolution).size()
        
        # 2. Total bytes sent to server per time bin
        time_series['bytes_toserver'] = self.beacon_data['bytes_toserver'].resample(time_resolution).sum()
        
        # 3. Total bytes received from server per time bin
        time_series['bytes_toclient'] = self.beacon_data['bytes_toclient'].resample(time_resolution).sum()
        
        # 4. Packet count (sum of packets to/from server)
        time_series['packets_total'] = self.beacon_data['flow'].apply(
            lambda x: x.get('pkts_toserver', 0) + x.get('pkts_toclient', 0) if isinstance(x, dict) else 0
        ).resample(time_resolution).sum()
        
        # Combine into DataFrame
        self.time_series = pd.DataFrame(time_series).fillna(0)
        
        # Add time in seconds from start
        start_time = self.time_series.index[0]
        self.time_series['seconds'] = (self.time_series.index - start_time).total_seconds()
        
        print(f"Time series created with {len(self.time_series)} time bins")
        print(f"Time resolution: {time_resolution}")
        
        return self.time_series
    
    def plot_time_series(self, save_path='time_series_plot.png'):
        """Plot the time series data"""
        if self.time_series is None:
            self.create_time_series()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('C2 Beacon Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Alert Count
        ax = axes[0, 0]
        ax.plot(self.time_series.index, self.time_series['alert_count'], 
                'b-', linewidth=1.5, marker='o', markersize=3)
        ax.set_title('Beacon Alert Count Over Time', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Alert Count')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Bytes to Server
        ax = axes[0, 1]
        ax.plot(self.time_series.index, self.time_series['bytes_toserver'], 
                'r-', linewidth=1.5, marker='s', markersize=3)
        ax.set_title('Bytes to C2 Server Over Time', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Bytes')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Total Packets
        ax = axes[1, 0]
        ax.plot(self.time_series.index, self.time_series['packets_total'], 
                'g-', linewidth=1.5, marker='^', markersize=3)
        ax.set_title('Total Packets Over Time', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Packet Count')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: All metrics normalized
        ax = axes[1, 1]
        metrics = ['alert_count', 'bytes_toserver', 'packets_total']
        colors = ['b', 'r', 'g']
        
        for metric, color in zip(metrics, colors):
            normalized = (self.time_series[metric] - self.time_series[metric].min()) / \
                        (self.time_series[metric].max() - self.time_series[metric].min())
            ax.plot(self.time_series.index, normalized, color + '-', 
                   linewidth=1.5, label=metric)
        
        ax.set_title('Normalized Metrics Comparison', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")
        plt.show()
        
        return fig
    
    def apply_fft(self, signal_data, sampling_rate=1):
        """
        Apply FFT to time series data
        
        Args:
            signal_data: Time series data
            sampling_rate: Samples per second
        
        Returns:
            frequencies, fft_magnitude
        """
        n = len(signal_data)
        
        # Apply window function to reduce spectral leakage
        window = signal.windows.hann(n)
        windowed_signal = signal_data * window
        
        # Remove mean (DC component) before FFT
        signal_no_dc = windowed_signal - np.mean(windowed_signal)
        
        # Apply FFT
        fft_result = fft(signal_no_dc)
        fft_magnitude = np.abs(fft_result)
        
        # Calculate frequencies
        frequencies = fftfreq(n, d=1/sampling_rate)
        
        # Only return positive frequencies
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        fft_magnitude = fft_magnitude[positive_freq_mask]
        
        return frequencies, fft_magnitude
    
    def detect_beacon_periodicity(self, frequencies, fft_magnitude, threshold=0.3):
        """
        Detect periodic beacon patterns from FFT
        
        Args:
            frequencies: Frequency array from FFT
            fft_magnitude: FFT magnitude array
            threshold: Relative threshold for peak detection
        
        Returns:
            dict with detection results
        """
        # Normalize magnitude
        normalized_magnitude = fft_magnitude / np.max(fft_magnitude)
        
        # EXCLUDE DC COMPONENT (frequency = 0) and very low frequencies
        # Ignore frequencies below 0.01 Hz (periods > 100 seconds)
        min_freq = 0.01
        valid_freq_mask = frequencies > min_freq
        
        # Only consider valid frequencies for peak detection
        valid_frequencies = frequencies[valid_freq_mask]
        valid_magnitudes = normalized_magnitude[valid_freq_mask]
        
        # Find peaks above threshold using scipy's find_peaks
        if len(valid_magnitudes) > 0:
            peaks, properties = find_peaks(valid_magnitudes, height=threshold, prominence=0.1)
        else:
            peaks = []
            properties = {'peak_heights': []}
        
        if len(peaks) > 0:
            peak_frequencies = valid_frequencies[peaks]
            peak_magnitudes = valid_magnitudes[peaks]
        else:
            peak_frequencies = np.array([])
            peak_magnitudes = np.array([])
        
        # Calculate periodicity score (research paper formula)
        # Weighted sum of significant peaks, emphasizing lower frequencies
        if len(peak_magnitudes) > 0:
            # Lower frequencies (longer periods) get higher weight for beacons
            weights = 1 / (peak_frequencies + 0.001)  # Add small constant to avoid division by zero
            normalized_weights = weights / np.sum(weights)
            score = np.sum(peak_magnitudes ** 2 * normalized_weights) * 1000
        else:
            score = 0
        
        # Calculate dominant period (excluding DC)
        if len(peak_frequencies) > 0:
            # Find the peak with highest magnitude
            dominant_idx = np.argmax(peak_magnitudes)
            dominant_freq = peak_frequencies[dominant_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else 0
        else:
            dominant_freq = 0
            dominant_period = 0
        
        results = {
            'periodicity_score': score,
            'dominant_frequency': dominant_freq,
            'dominant_period': dominant_period,
            'peak_frequencies': peak_frequencies,
            'peak_magnitudes': peak_magnitudes,
            'is_periodic': score > 50 and dominant_period > 0  # Higher threshold for beacon detection
        }
        
        return results
    
    def plot_fft_analysis(self, save_path='fft_analysis.png'):
        """Plot FFT analysis results"""
        if self.time_series is None:
            self.create_time_series()
        
        # Prepare data for FFT
        signal_data = self.time_series['alert_count'].values
        sampling_rate = 1  # 1 sample per second
        
        # Apply FFT
        frequencies, fft_magnitude = self.apply_fft(signal_data, sampling_rate)
        
        # Detect periodicity
        detection_results = self.detect_beacon_periodicity(frequencies, fft_magnitude)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('C2 Beacon FFT Frequency Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Original Time Series with Beacon Points
        ax = axes[0, 0]
        ax.plot(self.time_series['seconds'], signal_data, 'b-', linewidth=2, label='Alert Count')
        
        # Mark beacon events
        beacon_indices = np.where(signal_data > 0)[0]
        if len(beacon_indices) > 0:
            beacon_times = self.time_series['seconds'].iloc[beacon_indices]
            beacon_values = signal_data[beacon_indices]
            ax.plot(beacon_times, beacon_values, 'ro', markersize=6, label='Beacon Events')
        
        ax.set_title('Time Series with Beacon Events', fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Alert Count')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: FFT Magnitude Spectrum
        ax = axes[0, 1]
        ax.plot(frequencies, fft_magnitude, 'r-', linewidth=1.5)
        ax.set_title('FFT Magnitude Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.3)
        
        # Highlight peaks above threshold
        normalized_magnitude = fft_magnitude / np.max(fft_magnitude)
        threshold = 0.3
        
        # Exclude DC component for highlighting
        non_dc_mask = frequencies > 0.01
        peaks_mask = normalized_magnitude > threshold
        
        # Fix the broadcasting issue
        condition = np.zeros_like(frequencies, dtype=bool)
        condition[non_dc_mask] = peaks_mask[non_dc_mask]
        
        peaks = frequencies[condition]
        peak_mags = fft_magnitude[condition]
        
        if len(peaks) > 0:
            ax.plot(peaks, peak_mags, 'go', markersize=8, label='Significant Peaks')
            ax.axhline(y=threshold * np.max(fft_magnitude), color='gray', 
                      linestyle='--', alpha=0.5, label='Threshold')
            ax.legend()
        
        # Plot 3: Normalized FFT with Period Indication
        ax = axes[1, 0]
        ax.plot(frequencies, normalized_magnitude, 'g-', linewidth=1.5)
        ax.set_title('Normalized FFT Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add period labels for significant peaks
        for freq, mag in zip(detection_results['peak_frequencies'], 
                            detection_results['peak_magnitudes']):
            if freq > 0:
                period = 1 / freq
                ax.axvline(x=freq, color='orange', linestyle=':', alpha=0.7, linewidth=2)
                # Annotate the period
                ax.annotate(f'{period:.1f}s', xy=(freq, mag), xytext=(freq, mag + 0.05),
                           fontsize=9, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))
        
        # Add period axis on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Period (seconds)', color='orange')
        ax2.tick_params(axis='x', labelcolor='orange')
        
        # Plot 4: Power Spectral Density (Welch's method)
        ax = axes[1, 1]
        f_welch, Pxx_welch = signal.welch(signal_data, fs=sampling_rate, 
                                         nperseg=min(256, len(signal_data)))
        ax.semilogy(f_welch, Pxx_welch, 'm-', linewidth=2)
        ax.set_title('Power Spectral Density (Welch)', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.grid(True, alpha=0.3)
        
        # Highlight any peaks in PSD
        if len(Pxx_welch) > 0:
            psd_peaks, _ = find_peaks(Pxx_welch, prominence=np.std(Pxx_welch)/2)
            if len(psd_peaks) > 0:
                ax.plot(f_welch[psd_peaks], Pxx_welch[psd_peaks], 'co', markersize=6, label='PSD Peaks')
                ax.legend()
        
        # Add detection results text
        detection_text = f"""
        Detection Results:
        -----------------
        Periodicity Score: {detection_results['periodicity_score']:.2f}
        Dominant Frequency: {detection_results['dominant_frequency']:.4f} Hz
        Dominant Period: {detection_results['dominant_period']:.2f} seconds
        Beacon Detected: {'YES' if detection_results['is_periodic'] else 'NO'}
        Significant Peaks: {len(detection_results['peak_frequencies'])}
        """
        
        plt.figtext(0.02, 0.02, detection_text, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FFT analysis plot saved to: {save_path}")
        
        # Print detection results
        print("\n" + "="*60)
        print("BEACON DETECTION RESULTS")
        print("="*60)
        print(f"Periodicity Score: {detection_results['periodicity_score']:.2f}")
        print(f"Dominant Frequency: {detection_results['dominant_frequency']:.4f} Hz")
        print(f"Dominant Period: {detection_results['dominant_period']:.2f} seconds")
        print(f"Significant Peaks Found: {len(detection_results['peak_frequencies'])}")
        
        if len(detection_results['peak_frequencies']) > 0:
            print("\nSignificant Frequencies:")
            for i, (freq, mag) in enumerate(zip(detection_results['peak_frequencies'], 
                                              detection_results['peak_magnitudes'])):
                print(f"  Peak {i+1}: {freq:.4f} Hz (Period: {1/freq:.2f}s, Magnitude: {mag:.2%})")
        
        print(f"\nBeacon Detected: {'YES' if detection_results['is_periodic'] else 'NO'}")
        
        if detection_results['is_periodic']:
            print(f"\n⚠️  C2 BEACON DETECTED!")
            print(f"   Estimated beacon interval: {detection_results['dominant_period']:.1f} seconds")
            print(f"   Confidence: {'HIGH' if detection_results['periodicity_score'] > 100 else 'MEDIUM'}")
        else:
            print("\nNo clear periodic beacon pattern detected.")
        
        plt.show()
        
        return fig, detection_results
    
    def analyze_beacon_interval(self):
        """Direct analysis of beacon intervals from timestamps"""
        if self.beacon_data is None:
            self.parse_logs()
        
        print("\n" + "="*60)
        print("DIRECT BEACON INTERVAL ANALYSIS")
        print("="*60)
        
        # Sort beacon data by timestamp
        beacon_times = self.beacon_data.sort_index()
        
        # Calculate intervals between consecutive beacons
        intervals = []
        if len(beacon_times) > 1:
            for i in range(1, len(beacon_times)):
                time_diff = (beacon_times.index[i] - beacon_times.index[i-1]).total_seconds()
                intervals.append(time_diff)
        
        if intervals:
            intervals = np.array(intervals)
            print(f"Total beacon events: {len(beacon_times)}")
            print(f"Number of intervals: {len(intervals)}")
            print(f"Average interval: {np.mean(intervals):.2f} seconds")
            print(f"Std dev of intervals: {np.std(intervals):.2f} seconds")
            print(f"Min interval: {np.min(intervals):.2f} seconds")
            print(f"Max interval: {np.max(intervals):.2f} seconds")
            
            # Create histogram of intervals
            plt.figure(figsize=(10, 6))
            plt.hist(intervals, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
            plt.axvline(x=np.mean(intervals), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(intervals):.2f}s')
            plt.title('Beacon Interval Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Interval (seconds)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('beacon_interval_distribution.png', dpi=150)
            plt.show()
            
            return intervals
        else:
            print("Not enough beacon events for interval analysis")
            return None
    
    def generate_report(self, output_file='beacon_analysis_report.txt'):
        """Generate comprehensive analysis report"""
        if self.time_series is None:
            self.create_time_series()
        
        # Perform FFT analysis
        signal_data = self.time_series['alert_count'].values
        frequencies, fft_magnitude = self.apply_fft(signal_data, sampling_rate=1)
        detection_results = self.detect_beacon_periodicity(frequencies, fft_magnitude)
        
        # Analyze intervals directly
        intervals = self.analyze_beacon_interval()
        
        # Generate report
        report = f"""
        C2 BEACON DETECTION ANALYSIS REPORT
        ===================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. DATA SUMMARY
        ---------------
        Log File: {self.log_file}
        Total Records: {len(self.df)}
        Beacon Alerts: {len(self.beacon_data)}
        Time Range: {self.time_series.index[0]} to {self.time_series.index[-1]}
        Duration: {(self.time_series.index[-1] - self.time_series.index[0]).total_seconds():.1f} seconds
        
        2. TIME SERIES STATISTICS
        -------------------------
        Alert Count:
          - Mean: {self.time_series['alert_count'].mean():.2f}
          - Std Dev: {self.time_series['alert_count'].std():.2f}
          - Max: {self.time_series['alert_count'].max():.0f}
          - Min: {self.time_series['alert_count'].min():.0f}
        
        Bytes to Server:
          - Total: {self.time_series['bytes_toserver'].sum():.0f} bytes
          - Mean per interval: {self.time_series['bytes_toserver'].mean():.2f}
        
        3. FREQUENCY ANALYSIS (FFT)
        ---------------------------
        Dominant Frequency: {detection_results['dominant_frequency']:.4f} Hz
        Dominant Period: {detection_results['dominant_period']:.2f} seconds
        Periodicity Score: {detection_results['periodicity_score']:.2f}/1000
        
        4. DIRECT INTERVAL ANALYSIS
        ---------------------------
        """
        
        if intervals is not None:
            report += f"""
          - Average Interval: {np.mean(intervals):.2f} seconds
          - Standard Deviation: {np.std(intervals):.2f} seconds
          - Minimum Interval: {np.min(intervals):.2f} seconds
          - Maximum Interval: {np.max(intervals):.2f} seconds
          - Consistency: {'HIGH' if np.std(intervals) < 3 else 'MEDIUM' if np.std(intervals) < 5 else 'LOW'}
        """
        
        report += f"""
        5. DETECTION RESULTS
        --------------------
        Beacon Pattern Detected: {'YES' if detection_results['is_periodic'] else 'NO'}
        Confidence Level: {'HIGH' if detection_results['periodicity_score'] > 100 else 
                          'MEDIUM' if detection_results['periodicity_score'] > 50 else 'LOW'}
        
        6. SIGNIFICANT FREQUENCIES
        ---------------------------
        """
        
        if len(detection_results['peak_frequencies']) > 0:
            for i, (freq, mag) in enumerate(zip(detection_results['peak_frequencies'], 
                                               detection_results['peak_magnitudes'])):
                report += f"        Peak {i+1}: {freq:.4f} Hz (Period: {1/freq:.2f}s, Magnitude: {mag:.2%})\n"
        else:
            report += "        No significant frequencies found above threshold.\n"
        
        report += f"""
        7. RECOMMENDATIONS
        ------------------
        """
        
        if detection_results['is_periodic']:
            report += f"""        🔴 HIGH PRIORITY: Periodic C2 beacon detected!
           - Estimated beacon interval: {detection_results['dominant_period']:.2f} seconds
           - Investigate source IP: {self.beacon_data['src_ip'].iloc[0] if len(self.beacon_data) > 0 else 'Unknown'}
           - Target C2 server: {self.beacon_data['dest_ip'].iloc[0] if len(self.beacon_data) > 0 else 'Unknown'}
           - Immediate actions:
              1. Isolate the affected machine
              2. Block C2 server IP in firewall
              3. Scan for malware persistence
              4. Review authentication logs"""
        else:
            report += """        ✅ No clear periodic beacon pattern detected.
           - This could indicate:
              1. No active C2 beacon
              2. Beacon is using random intervals
              3. Beacon traffic is well-hidden in noise
           - Recommended actions:
              1. Continue monitoring
              2. Review other IOC patterns
              3. Check for encrypted C2 channels"""
        
        report += f"""
        
        8. FILES GENERATED
        ------------------
        - time_series_analysis.png: Time vs. traffic volume
        - fft_frequency_analysis.png: Frequency domain analysis
        - beacon_interval_distribution.png: Interval histogram
        - This report: {output_file}
        
        9. ANALYSIS METHODOLOGY
        -----------------------
        1. Time series created with 1-second resolution
        2. FFT applied with Hanning window to reduce spectral leakage
        3. DC component (0 Hz) excluded from analysis
        4. Peaks detected using prominence-based algorithm
        5. Periodicity scoring based on weighted peak magnitudes
        
        END OF REPORT
        """
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"\nAnalysis report saved to: {output_file}")
        
        return report

def main():
    """Main execution function"""
    print("C2 Beacon Detection Analyzer - FIXED VERSION")
    print("="*60)
    print("Fixed: DC component exclusion and array broadcasting error")
    print("="*60)
    
    # Initialize analyzer
    analyzer = BeaconAnalyzer('c2_beacon.log')
    
    # Parse logs
    analyzer.parse_logs()
    
    # Create time series
    analyzer.create_time_series(time_resolution='1S')
    
    # Plot time series
    analyzer.plot_time_series('time_series_analysis.png')
    
    # Plot FFT analysis
    analyzer.plot_fft_analysis('fft_frequency_analysis.png')
    
    # Analyze intervals directly
    analyzer.analyze_beacon_interval()
    
    # Generate report
    analyzer.generate_report('beacon_detection_report.txt')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Output files generated:")
    print("  ✅ time_series_analysis.png - Time vs. volume plot")
    print("  ✅ fft_frequency_analysis.png - FFT with DC component fixed")
    print("  ✅ beacon_interval_distribution.png - Interval histogram")
    print("  ✅ beacon_detection_report.txt - Comprehensive analysis")
    print("="*60)
    print("\nKey improvements in this version:")
    print("  1. Fixed array broadcasting error in peak detection")
    print("  2. DC component (0 Hz) properly excluded from FFT analysis")
    print("  3. Better peak detection using scipy's find_peaks")
    print("  4. Improved periodicity scoring formula")
    print("  5. Direct interval analysis from timestamps")
    print("  6. Enhanced visualization with period annotations")

if __name__ == "__main__":
    main()