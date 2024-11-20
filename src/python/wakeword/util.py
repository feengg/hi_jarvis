import winreg
import socket
import os

def read_registry(path, name):
    for key in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(key, path) as reg_key:
                value, _ = winreg.QueryValueEx(reg_key, name)
                return value
        except WindowsError as e:
            raise ValueError(f"Failed to read reg {name}") from e
    return ""


def write_to_registry(path, name, regtype, value):
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_WRITE
        ) as reg_key:
            winreg.SetValueEx(reg_key, name, 0, regtype, value)
    except WindowsError as e:
        raise ValueError(f"Cannot write to {name} in {path}") from e
    

def get_available_port(start_port):
    for port in range(start_port, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                pass
    raise RuntimeError("There is no available port.")

def get_default_home():
    return os.path.join(os.path.expanduser("~"), ".cache")
