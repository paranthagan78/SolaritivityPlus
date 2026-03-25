import re

file_path = "e:/Projects/SolaritivityPlus/templates/partials/js.html"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_template = False
start_line = -1

for i, line in enumerate(lines):
    backticks = re.findall(r'`', line)
    for _ in backticks:
        if not in_template:
            in_template = True
            start_line = i + 1
        else:
            in_template = False
            start_line = -1

if in_template:
    print(f"Unterminated template literal started at line {start_line}")
else:
    print("All template literals seem terminated (inline check).")
