#!/usr/bin/env python3

import rospy
import subprocess
import time

def debug_system():
    """Debug the Phase 2 system to identify issues"""
    
    print("PHASE 2 SYSTEM DEBUGGER")
    print("=" * 50)
    
    # Check ROS topics
    print("\nAVAILABLE TOPICS:")
    try:
        result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=5)
        topics = result.stdout.strip().split('\n')
        for topic in sorted(topics):
            if 'object_follower' in topic or 'camera' in topic or 'cmd_vel' in topic:
                print(f"  ✓ {topic}")
    except Exception as e:
        print(f"Error getting topics: {e}")
    
    # Check active nodes
    print("\nACTIVE NODES:")
    try:
        result = subprocess.run(['rosnode', 'list'], capture_output=True, text=True, timeout=5)
        nodes = result.stdout.strip().split('\n')
        for node in sorted(nodes):
            print(f"{node}")
    except Exception as e:
        print(f"Error getting nodes: {e}")
    
    # Test topic publishing rates
    print("\nTOPIC RATES:")
    test_topics = [
        '/camera/image_raw',
        '/object_follower/target_found', 
        '/object_follower/target_position',
        '/cmd_vel'
    ]
    
    for topic in test_topics:
        try:
            result = subprocess.run(['rostopic', 'hz', topic], 
                                  capture_output=True, text=True, timeout=3)
            if result.stdout:
                rate_line = [line for line in result.stdout.split('\n') if 'average rate' in line]
                if rate_line:
                    print(f"{topic}: {rate_line[0].strip()}")
                else:
                    print(f"{topic}: Publishing but no rate data")
            else:
                print(f"{topic}: No data")
        except subprocess.TimeoutExpired:
            print(f"{topic}: Timeout - likely no data")
        except Exception as e:
            print(f"{topic}: Error - {e}")
    
    # Check recent topic messages
    print("\n📝 RECENT MESSAGES:")
    
    # Check target_found
    try:
        result = subprocess.run(['rostopic', 'echo', '/object_follower/target_found', '-n', '1'], 
                              capture_output=True, text=True, timeout=2)
        if result.stdout:
            print(f"target_found: {result.stdout.strip()}")
        else:
            print("target_found: No recent messages")
    except:
        print("target_found: Topic not available")
    
    # Check if camera images are being published
    try:
        result = subprocess.run(['rostopic', 'echo', '/camera/image_raw', '-n', '1'], 
                              capture_output=True, text=True, timeout=2)
        if result.stdout:
            print(" camera/image_raw: Images being published")
        else:
            print(" camera/image_raw: No images")
    except:
        print("  camera/image_raw: No camera data")

if __name__ == '__main__':
    debug_system()