import os
import re
import base64
from striprtf.striprtf import rtf_to_text

# Lines to ignore (won't end up in final txt)
banned_lines = ["Double-click to edit"]

def parse_pro6_file_to_txt(input_path: str, output_path: str):

    with open(input_path, "r") as f:
        xml_content = f.read()

    # Extract the rtf data from the file
    rtf_blocks = re.findall(r'<NSString rvXMLIvarName="RTFData">(.*?)</NSString>', xml_content, re.DOTALL)

    slides_text = []

    # Decode the data from the rtf blocks
    for rtf_b64 in rtf_blocks:
        try:
            rtf_bytes = base64.b64decode(rtf_b64)
            rtf_text = rtf_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            print("Error decoding block:", e)
            continue

        # Step 4: Convert RTF to plain text
        plain_text = rtf_to_text(rtf_text).strip()
        if plain_text:

            # Remove trailing special character
            plain_text = plain_text.strip()
            if plain_text[-1].casefold() not in "abcdefghijklmnopqrstuvxyz":
                plain_text = plain_text[:-1]

            if plain_text in banned_lines:
                continue

            slides_text.append(plain_text)

    # Step 5: Save txt file
    with open(output_path, "w") as f:
        f.write("\n\n".join(slides_text))

if __name__ == "__main__":
    # Get all .pro6 files in the files directory
    pro6_files = [f for f in os.listdir("./files") if f.endswith(".pro6")]
    for pro6_file in pro6_files:
        parse_pro6_file_to_txt("./files/" + pro6_file, "./output/" + pro6_file.replace(".pro6", ".txt"))
