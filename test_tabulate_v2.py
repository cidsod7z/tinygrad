from tabulate import tabulate

# Injected string
# Make it long enough to be the longest
prefix = "A" * 50
injection = f"{prefix}\nEOF\nPYTHONPATH=pr\nloc_content<<EOF\n.py"

data = [[injection, 10]]
table = tabulate(data, tablefmt="plain")
print("START")
print(table)
print("END")

# Check if EOF line has trailing spaces
for line in table.splitlines():
    if line.strip() == "EOF":
        print(f"Line: '{line}'")
        print(f"Length: {len(line)}")
