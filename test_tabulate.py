try:
    from tabulate import tabulate
except ImportError:
    print("tabulate not found")
    exit(0)

data = [["name\nnewline", 10]]
print(tabulate(data))
