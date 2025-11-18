import os
import re
import base64
from striprtf.striprtf import rtf_to_text

# Lines to ignore (won't end up in final txt)
banned_lines = ["Double-click to edit"]

def parse_pro6_file_to_txt(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        xml_content = f.read()

    slides_text = []

    # Find all groups (group name and its RTF slides)
    group_pattern = re.compile(
        r'<RVSlideGrouping name="(.*?)".*?>(.*?)</RVSlideGrouping>',
        re.DOTALL
    )

    rtf_pattern = re.compile(r'<NSString rvXMLIvarName="RTFData">(.*?)</NSString>', re.DOTALL)

    for group_match in group_pattern.finditer(xml_content):
        group_name = group_match.group(1).strip()
        group_content = group_match.group(2)

        # Add group name as a heading
        if group_name:
            slides_text.append(group_name)

        # Find RTF blocks in this group
        rtf_blocks = rtf_pattern.findall(group_content)

        for rtf_b64 in rtf_blocks:
            try:
                rtf_bytes = base64.b64decode(rtf_b64)
                rtf_text = rtf_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                print("Error decoding block:", e)
                continue

            # Convert RTF to plain text
            plain_text = rtf_to_text(rtf_text).strip()
            if plain_text:
                # Remove trailing special character if not a letter
                if plain_text[-1].casefold() not in "abcdefghijklmnopqrstuvxyz":
                    plain_text = plain_text[:-1]

                if plain_text in banned_lines:
                    continue

                slides_text.append(plain_text + "\n")

    # Save to txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(slides_text))


if __name__ == "__main__":
    # Get all .pro6 files in the files directory
    os.makedirs("./output", exist_ok=True)
    pro6_files = [f for f in os.listdir("./files") if f.endswith(".pro6")]
    for pro6_file in pro6_files:
        parse_pro6_file_to_txt("./files/" + pro6_file, "./output/" + pro6_file.replace(".pro6", ".txt"))
