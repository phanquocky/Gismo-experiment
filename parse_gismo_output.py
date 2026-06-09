from typing import List, Dict

def parse_gismo_ind_from_text(text: str) -> List[int]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("c ind "):
            toks = line.split()[2:]
            out = []
            for t in toks:
                try:
                    v = int(t)
                    if v != 0:
                        out.append(v)
                except:
                    pass
            return out
    raise RuntimeError("No 'c ind' line found in GiSMo output.")

def parse_groups_from_gcnf(gcnf_path: str) -> Dict[int, int]:
    var2grp = {}
    group_id = 0
    with open(gcnf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("c grp "):
                group_id += 1
                toks = line.split()[2:]
                for t in toks:
                    try:
                        v = int(t)
                        if v != 0:
                            var2grp[v] = group_id
                    except:
                        pass
    if not var2grp:
        raise RuntimeError("No 'c grp' lines found. Encode with two_step=True.")
    return var2grp

def parse_sensor_set_from_gismo_output(gismo_text: str, gcnf_path: str) -> List[int]:
    ind_vars = parse_gismo_ind_from_text(gismo_text)
    var2grp = parse_groups_from_gcnf(gcnf_path)
    sensor_S = sorted({ var2grp[v] for v in ind_vars })
    return sensor_S


