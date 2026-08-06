"""SSH helper: run a command on the compshare pod and print output."""
import sys
import paramiko

HOST = "cpod-1sqfx2anig0i.podtcp.compshare.cn"
PORT = 27305
USER = "root"
PASS = "9576jH8WDn32m1Lk"


def run(cmd, timeout=280):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    client.close()
    return code, out, err


if __name__ == "__main__":
    cmd = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 280
    code, out, err = run(cmd, timeout)
    print(out)
    if err.strip():
        print("--- STDERR ---")
        print(err)
    print(f"[exit={code}]")
