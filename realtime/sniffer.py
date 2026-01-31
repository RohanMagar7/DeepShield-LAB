"""
Real-Time Network Traffic Sniffer
Captures live network packets and performs real-time intrusion detection.
"""

import sys
import os
import signal
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    print("⚠ Scapy not available. Install with: pip install scapy")
    SCAPY_AVAILABLE = False

from realtime.flow_extractor import FlowExtractor
from realtime.detector import IntrusionDetector


class NetworkSniffer:
    """
    Real-time network sniffer with intrusion detection.
    """
    
    def __init__(self, interface=None, detector=None):
        """
        Initialize network sniffer.
        
        Args:
            interface: Network interface to capture (None = default)
            detector: IntrusionDetector instance
        """
        self.interface = interface
        self.detector = detector
        self.flow_extractor = FlowExtractor(flow_timeout=5.0)
        
        self.packet_count = 0
        self.attack_count = 0
        self.flow_check_interval = 10  # Check flows every N packets
        self.running = True
        
        # Statistics
        self.stats = defaultdict(int)
        self.start_time = None
        
    def packet_callback(self, packet):
        """
        Callback for each captured packet.
        
        Args:
            packet: Scapy packet object
        """
        try:
            self.packet_count += 1
            
            # Extract packet info
            packet_info = self._extract_packet_info(packet)
            
            if packet_info:
                # Add to flow
                self.flow_extractor.add_packet(packet_info)
                
                # Periodically check flows
                if self.packet_count % self.flow_check_interval == 0:
                    self._check_flows()
                    
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def _extract_packet_info(self, packet):
        """
        Extract relevant information from packet.
        
        Args:
            packet: Scapy packet
            
        Returns:
            Dictionary with packet info or None
        """
        if not packet.haslayer(IP):
            return None
        
        ip_layer = packet[IP]
        
        packet_info = {
            'src_ip': ip_layer.src,
            'dst_ip': ip_layer.dst,
            'protocol': ip_layer.proto,
            'length': len(packet),
            'timestamp': datetime.now(),
            'src_port': 0,
            'dst_port': 0,
            'flags': 0
        }
        
        # Extract transport layer info
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            packet_info['src_port'] = tcp_layer.sport
            packet_info['dst_port'] = tcp_layer.dport
            packet_info['flags'] = int(tcp_layer.flags)
            self.stats['tcp'] += 1
            
        elif packet.haslayer(UDP):
            udp_layer = packet[UDP]
            packet_info['src_port'] = udp_layer.sport
            packet_info['dst_port'] = udp_layer.dport
            self.stats['udp'] += 1
            
        elif packet.haslayer(ICMP):
            self.stats['icmp'] += 1
        else:
            self.stats['other'] += 1
        
        return packet_info
    
    def _check_flows(self):
        """Check active flows for intrusions."""
        active_flows = self.flow_extractor.get_active_flows()
        
        for flow_id in active_flows:
            # Extract features
            features = self.flow_extractor.extract_features(flow_id)
            
            if features is None:
                continue
            
            # Detect intrusion
            try:
                result = self.detector.detect(features, use_dqn=True)
                
                if result['is_attack']:
                    self.attack_count += 1
                    self._handle_attack(flow_id, result)
                    
            except Exception as e:
                print(f"Detection error: {e}")
    
    def _handle_attack(self, flow_id, result):
        """
        Handle detected attack.
        
        Args:
            flow_id: Flow identifier
            result: Detection result
        """
        flow_info = self.flow_extractor.get_flow_info(flow_id)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*70)
        print(f"⚠️  ATTACK DETECTED [{timestamp}]")
        print("="*70)
        print(f"Flow: {flow_id[0]}:{flow_id[1]} -> {flow_id[2]}:{flow_id[3]}")
        print(f"Protocol: {flow_id[4]}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Threshold: {result['threshold']:.3f}")
        print(f"Threat Level: {result['threat_level']}")
        print(f"Packets: {flow_info['num_packets']}")
        print(f"Duration: {flow_info['duration']:.2f}s")
        print("="*70 + "\n")
    
    def print_statistics(self):
        """Print capture statistics."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        else:
            duration = 0
        
        print("\n" + "="*70)
        print("CAPTURE STATISTICS")
        print("="*70)
        print(f"Duration: {duration:.1f}s")
        print(f"Total Packets: {self.packet_count}")
        print(f"Packet Rate: {self.packet_count / duration:.1f} pkt/s" if duration > 0 else "N/A")
        print(f"Active Flows: {len(self.flow_extractor.get_active_flows())}")
        print(f"Attacks Detected: {self.attack_count}")
        print(f"\nProtocol Distribution:")
        print(f"  TCP:   {self.stats['tcp']}")
        print(f"  UDP:   {self.stats['udp']}")
        print(f"  ICMP:  {self.stats['icmp']}")
        print(f"  Other: {self.stats['other']}")
        print("="*70 + "\n")
    
    def start(self, packet_count=0):
        """
        Start packet capture.
        
        Args:
            packet_count: Number of packets to capture (0 = infinite)
        """
        if not SCAPY_AVAILABLE:
            print("✗ Scapy is required for packet capture")
            print("  Install with: pip install scapy")
            return
        
        print("\n" + "="*70)
        print(" "*20 + "DEEPSHIELD-LAB SNIFFER")
        print(" "*15 + "Real-Time Intrusion Detection")
        print("="*70)
        print(f"Interface: {self.interface if self.interface else 'default'}")
        print(f"Capture limit: {'unlimited' if packet_count == 0 else f'{packet_count} packets'}")
        print("\nPress Ctrl+C to stop capture\n")
        print("="*70 + "\n")
        
        self.start_time = datetime.now()
        
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\n⚠️  Stopping capture...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Start sniffing
            sniff(
                iface=self.interface,
                prn=self.packet_callback,
                count=packet_count if packet_count > 0 else 0,
                store=False
            )
            
        except PermissionError:
            print("\n✗ Permission denied. Run with sudo/root privileges:")
            print("  sudo python realtime/sniffer.py")
            
        except Exception as e:
            print(f"\n✗ Capture error: {e}")
            
        finally:
            self.print_statistics()


def main():
    """Main function."""
    print("Initializing DeepShield-LAB Sniffer...")
    
    # Check for required models
    model_path = 'models/deepshield_cnn_lstm.h5'
    if not os.path.exists(model_path):
        print("\n✗ Trained models not found!")
        print("  Train models first with: python train/train_hybrid.py")
        sys.exit(1)
    
    try:
        # Initialize detector
        detector = IntrusionDetector()
        detector.load_models()
        
        # Initialize sniffer
        sniffer = NetworkSniffer(interface=None, detector=detector)
        
        # Start capture
        sniffer.start(packet_count=0)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Train models first with: python train/train_hybrid.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Check if running as root (required for packet capture)
    if os.geteuid() != 0 and sys.platform != 'win32':
        print("⚠️  Warning: Packet capture typically requires root privileges")
        print("   Run with: sudo python realtime/sniffer.py\n")
    
    main()
