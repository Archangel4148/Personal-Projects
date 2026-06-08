
with open("raw_data.txt", "r") as f:
    lines = f.readlines()

clean_lines = []
current_line = []
for line in lines:
    if "\t" in line and not "," in line:
        if current_line:
            clean_lines.append(current_line)
        current_line = [line.split("\t")[0]]
    else:
        current_line.append(line.replace("\t", "").replace("\n", ""))

headers = ["index", "team", ""]
whole_clean_lines = [", ".join(line) for line in clean_lines]

# print("\n".join(clean_lines[:10]))

with open("new_data.txt", "w", encoding="utf-8") as g:
    for line in clean_lines:
        fline = line
        if len(line) < 11:
            fline.insert(3, "View")
        g.write(", ".join(line))

print(clean_lines[-5:-3])

print(set([len(l) for l in clean_lines]))

chunks = [lines[i:i + 10] for i in range(0, len(lines), 10)]
print(chunks[:5])