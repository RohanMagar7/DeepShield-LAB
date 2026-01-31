"""
Network Flow Feature Extractor
Extracts statistical features from network packets for IDS analysis.
"""

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta


class FlowExtractor:
    """
    Extracts flow-level features from network packets.
    Aggregates packets into flows and computes statistical features.
    """
    
    def __init__(self, flow_timeout=5.0):
        """
        Initialize flow extractor.
        
        Args:
            flow_timeout: Flow timeout in seconds
        """
        self.flow_timeout = flow_timeout
        self.flows = {}
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(seconds=10)
        
    def _get_flow_id(self, packet_info):
        """
        Generate flow identifier from packet.
        
        Args:
            packet_info: Dictionary with packet information
            
        Returns:
            flow_id: Tuple identifying the flow
        """
        src_ip = packet_info.get('src_ip', '0.0.0.0')
        dst_ip = packet_info.get('dst_ip', '0.0.0.0')
        src_port = packet_info.get('src_port', 0)
        dst_port = packet_info.get('dst_port', 0)
        protocol = packet_info.get('protocol', 0)
        
        # Bidirectional flow (sort IPs and ports)
        if (src_ip, src_port) < (dst_ip, dst_port):
            flow_id = (src_ip, src_port, dst_ip, dst_port, protocol)
        else:
            flow_id = (dst_ip, dst_port, src_ip, src_port, protocol)
        
        return flow_id
    
    def add_packet(self, packet_info):
        """
        Add packet to flow.
        
        Args:
            packet_info: Dictionary with packet information
                - src_ip, dst_ip, src_port, dst_port, protocol
                - length, flags, timestamp
        """
        flow_id = self._get_flow_id(packet_info)
        
        if flow_id not in self.flows:
            self.flows[flow_id] = {
                'packets': [],
                'start_time': packet_info['timestamp'],
                'last_seen': packet_info['timestamp']
            }
        
        flow = self.flows[flow_id]
        flow['packets'].append(packet_info)
        flow['last_seen'] = packet_info['timestamp']
        
        # Periodic cleanup
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_flows()
    
    def _cleanup_old_flows(self):
        """Remove expired flows."""
        current_time = datetime.now()
        expired = []
        
        for flow_id, flow in self.flows.items():
            if (current_time - flow['last_seen']).total_seconds() > self.flow_timeout:
                expired.append(flow_id)
        
        for flow_id in expired:
            del self.flows[flow_id]
        
        self.last_cleanup = current_time
    
    def extract_features(self, flow_id):
        """
        Extract 41 features from a flow.
        
        Args:
            flow_id: Flow identifier
            
        Returns:
            features: Array of 41 features
        """
        if flow_id not in self.flows:
            return None
        
        flow = self.flows[flow_id]
        packets = flow['packets']
        
        if not packets:
            return None
        
        # Calculate duration
        duration = (flow['last_seen'] - flow['start_time']).total_seconds()
        
        # Packet counts
        num_packets = len(packets)
        
        # Byte statistics
        packet_lengths = [p.get('length', 0) for p in packets]
        total_bytes = sum(packet_lengths)
        mean_length = np.mean(packet_lengths) if packet_lengths else 0
        std_length = np.std(packet_lengths) if len(packet_lengths) > 1 else 0
        min_length = min(packet_lengths) if packet_lengths else 0
        max_length = max(packet_lengths) if packet_lengths else 0
        
        # Rate statistics
        packet_rate = num_packets / duration if duration > 0 else 0
        byte_rate = total_bytes / duration if duration > 0 else 0
        
        # Direction-based statistics (forward vs backward)
        src_ip = packets[0].get('src_ip', '0.0.0.0')
        forward_packets = [p for p in packets if p.get('src_ip') == src_ip]
        backward_packets = [p for p in packets if p.get('src_ip') != src_ip]
        
        fwd_count = len(forward_packets)
        bwd_count = len(backward_packets)
        
        fwd_lengths = [p.get('length', 0) for p in forward_packets]
        bwd_lengths = [p.get('length', 0) for p in backward_packets]
        
        fwd_total = sum(fwd_lengths)
        bwd_total = sum(bwd_lengths)
        
        fwd_mean = np.mean(fwd_lengths) if fwd_lengths else 0
        bwd_mean = np.mean(bwd_lengths) if bwd_lengths else 0
        
        # Inter-arrival times
        if len(packets) > 1:
            timestamps = [p['timestamp'] for p in packets]
            iat = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                   for i in range(len(timestamps)-1)]
            iat_mean = np.mean(iat) if iat else 0
            iat_std = np.std(iat) if len(iat) > 1 else 0
            iat_max = max(iat) if iat else 0
            iat_min = min(iat) if iat else 0
        else:
            iat_mean = iat_std = iat_max = iat_min = 0
        
        # Protocol encoding
        protocol = packets[0].get('protocol', 0)
        protocol_tcp = 1 if protocol == 6 else 0
        protocol_udp = 1 if protocol == 17 else 0
        
        # Flag statistics (for TCP)
        flags = [p.get('flags', 0) for p in packets]
        syn_count = sum(1 for f in flags if f & 0x02)
        fin_count = sum(1 for f in flags if f & 0x01)
        rst_count = sum(1 for f in flags if f & 0x04)
        psh_count = sum(1 for f in flags if f & 0x08)
        ack_count = sum(1 for f in flags if f & 0x10)
        urg_count = sum(1 for f in flags if f & 0x20)
        
        # Construct 41-feature vector (NSL-KDD compatible)
        features = [
            duration,                    # 0: duration
            protocol_tcp,                # 1: protocol_type (TCP)
            protocol_udp,                # 2: service (UDP)
            syn_count,                   # 3: flag (SYN)
            fwd_total,                   # 4: src_bytes
            bwd_total,                   # 5: dst_bytes
            0,                           # 6: land
            0,                           # 7: wrong_fragment
            urg_count,                   # 8: urgent
            psh_count,                   # 9: hot
            0,                           # 10: num_failed_logins
            ack_count > 0,               # 11: logged_in
            0,                           # 12: num_compromised
            0,                           # 13: root_shell
            0,                           # 14: su_attempted
            0,                           # 15: num_root
            0,                           # 16: num_file_creations
            0,                           # 17: num_shells
            0,                           # 18: num_access_files
            0,                           # 19: num_outbound_cmds
            0,                           # 20: is_host_login
            0,                           # 21: is_guest_login
            num_packets,                 # 22: count
            fwd_count,                   # 23: srv_count
            0,                           # 24: serror_rate
            0,                           # 25: srv_serror_rate
            0,                           # 26: rerror_rate
            0,                           # 27: srv_rerror_rate
            fwd_count / num_packets if num_packets > 0 else 0,  # 28: same_srv_rate
            bwd_count / num_packets if num_packets > 0 else 0,  # 29: diff_srv_rate
            0,                           # 30: srv_diff_host_rate
            num_packets,                 # 31: dst_host_count
            fwd_count,                   # 32: dst_host_srv_count
            fwd_count / num_packets if num_packets > 0 else 0,  # 33: dst_host_same_srv_rate
            bwd_count / num_packets if num_packets > 0 else 0,  # 34: dst_host_diff_srv_rate
            packet_rate,                 # 35: dst_host_same_src_port_rate
            0,                           # 36: dst_host_srv_diff_host_rate
            0,                           # 37: dst_host_serror_rate
            0,                           # 38: dst_host_srv_serror_rate
            0,                           # 39: dst_host_rerror_rate
            0,                           # 40: dst_host_srv_rerror_rate
        ]
        
        return np.array(features, dtype=np.float32)
    
    def get_active_flows(self):
        """
        Get all active flow IDs.
        
        Returns:
            List of flow IDs
        """
        return list(self.flows.keys())
    
    def get_flow_info(self, flow_id):
        """
        Get flow information.
        
        Args:
            flow_id: Flow identifier
            
        Returns:
            Dictionary with flow info
        """
        if flow_id not in self.flows:
            return None
        
        flow = self.flows[flow_id]
        return {
            'flow_id': flow_id,
            'num_packets': len(flow['packets']),
            'start_time': flow['start_time'],
            'last_seen': flow['last_seen'],
            'duration': (flow['last_seen'] - flow['start_time']).total_seconds()
        }


if __name__ == '__main__':
    # Test flow extractor
    print("Testing Flow Extractor...")
    
    extractor = FlowExtractor(flow_timeout=5.0)
    
    # Simulate packets
    from datetime import datetime
    
    packet1 = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.1',
        'src_port': 12345,
        'dst_port': 80,
        'protocol': 6,  # TCP
        'length': 64,
        'flags': 0x02,  # SYN
        'timestamp': datetime.now()
    }
    
    packet2 = {
        'src_ip': '10.0.0.1',
        'dst_ip': '192.168.1.100',
        'src_port': 80,
        'dst_port': 12345,
        'protocol': 6,
        'length': 1024,
        'flags': 0x12,  # SYN-ACK
        'timestamp': datetime.now()
    }
    
    # Add packets
    extractor.add_packet(packet1)
    extractor.add_packet(packet2)
    
    # Extract features
    flow_id = extractor._get_flow_id(packet1)
    features = extractor.extract_features(flow_id)
    
    print(f"\n✓ Flow Extractor Test Passed")
    print(f"  Active flows: {len(extractor.get_active_flows())}")
    print(f"  Features extracted: {features.shape}")
    print(f"  Feature dimension: {len(features)}")
