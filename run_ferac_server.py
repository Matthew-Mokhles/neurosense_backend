"""
run_server.py
==============
Convenience launcher for server.py — prints the LAN IP your phone needs
to connect to, then starts uvicorn.

Usage:
    python run_server.py
    python run_server.py --port 8080
"""

import argparse
import socket

import uvicorn


def get_lan_ip() -> str:
    """
    Best-effort LAN IP detection: opens a UDP socket toward a public IP
    (no actual packet is sent for UDP connect — this just makes the OS
    pick the right local interface/IP) and reads back the local address.
    Falls back to localhost if anything goes wrong (e.g. no network).
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="0.0.0.0 binds all interfaces so your phone on the same LAN can reach it")
    args = parser.parse_args()

    lan_ip = get_lan_ip()
    print("=" * 60)
    print("  FERAC Inference Server")
    print("=" * 60)
    print(f"  Local access:  http://127.0.0.1:{args.port}")
    print(f"  LAN access:    http://{lan_ip}:{args.port}   <- use THIS in the Flutter app")
    print(f"  Health check:  http://{lan_ip}:{args.port}/health")
    print("=" * 60)
    print("  Make sure your phone is on the SAME Wi-Fi network as this machine.")
    print("  If the phone can't connect, check your firewall allows inbound")
    print(f"  connections on port {args.port} (Windows Defender Firewall may")
    print("  prompt you the first time uvicorn binds the port - allow it).")
    print("=" * 60)

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
