def main():
    # Before: many statements → high complexity
    ...

# Refactor → split responsibilities
def run_all_checks():
    ...

def print_summary(results):
    ...

def main():
    results = run_all_checks()
    print_summary(results)