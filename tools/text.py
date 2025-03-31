import re
def extract_box_content(text):
    results = []
    start_keyword = r"\boxed{"
    start_len = len(start_keyword)
    stack = []
    level = 0
    start_pos = -1

    i = 0
    while i < len(text):
        if text[i:i+start_len] == start_keyword and (i == 0 or text[i-1] != "\\"):
            start_pos = i + start_len
            level = 1
            i += start_len
            continue
        if start_pos != -1:
            if text[i] == "{":
                level += 1
            elif text[i] == "}":
                level -= 1
                if level == 0:
                    results.append(text[start_pos:i])
                    start_pos = -1
        i += 1
    return results

def extract_box_number(text):
    results = extract_box_content(text)
    if(len(results)==0):
        return None
    match = re.findall(r'\d+', results[-1])
    if match:
        return int(match[-1])
    else:
        return None