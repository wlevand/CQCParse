import re

# regex to match mode tokens like "1(2)"
MODE_RE = re.compile(r"^(\d+)\((\d+)\)$")

def expand_mode(mode: int, n: int):
    """Expand a mode token into repeated integers, e.g. 1(3) -> (1,1,1)"""
    return (mode,) * n

def parse_freq_line(line: str):
    tokens = line.split()
    mode_tokens = []
    numbers = []

    for t in tokens:
        if MODE_RE.match(t):
            mode_tokens.append(t)
        else:
            try:
                numbers.append(float(t))
            except ValueError:
                pass

    if len(numbers) < 2 or not mode_tokens:
        return None  # not a data line

    # Expand modes
    modes = []
    for mt in mode_tokens:
        m, n = map(int, MODE_RE.match(mt).groups())
        modes.extend(expand_mode(m, n))

    E_harm, E_anharm = numbers[0], numbers[1]
    modes = [i-1 for i in modes]
    return tuple(modes), E_harm, E_anharm

def parse_section(lines):
    result = []
    for line in lines:
        parsed = parse_freq_line(line)
        if parsed is not None and parsed not in result:
            result.append(parsed)
    return result

def extract_sections(lines):
    sections = {
        "Fundamental Bands": [],
        "Overtones": [],
        "Combination Bands": [],
    }

    current = None
    for line in lines:
        line_strip = line.strip()
        if line_strip in sections:
            current = line_strip
            continue
        if 'Dipole strengths (DS) in 10^-40 esu^2.cm^2' in line_strip:
            print('HELLO')
            return sections

        if current:
            # if line
            sections[current].append(line)

    return sections

def extract_relevant_block(lines):
    """
    Extract the lines between the start and end markers.
    Start markers: "Anharmonic Infrared Spectroscopy" or "Vibrational Energies at Anharmonic Level"
    End marker: "Dipole strengths (DS) in 10^-40 esu^2.cm^2"
    """
    start_markers = [
        "Anharmonic Infrared Spectroscopy",
        "Vibrational Energies at Anharmonic Level"
    ]
    end_marker = "Dipole strengths (DS) in 10^-40 esu^2.cm^2"

    start_idx = None
    end_idx = None

    # find start
    for i, line in enumerate(lines):
        if any(marker in line for marker in start_markers):
            start_idx = i + 1  # start after this line
            break

    # find end
    for i, line in enumerate(lines):
        if end_marker in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise RuntimeError("Could not find start or end markers in file.")

    return lines[start_idx:end_idx]


def parse_frequencies(lines) -> dict[str, list]:
    relevant = extract_relevant_block(lines)
    blocks = extract_sections(relevant)
    return {
        name: parse_section(block)
        for name, block in blocks.items()
        if block
    }


def get_allStates_from_parsed_freqs(parsed_dict: dict[str, list]) -> tuple[dict, dict]:
    """
    {'Fundamental Bands': [((1,), 2878.687, 2726.813), ((2,), 1820.416, 1794.54), ((3,), 1534.549, 1501.586), ((4,), 1203.179, 1185.288), ((5,), 2933.526, 2682.765), ((6,), 1268.91, 1247.878)], 
    'Overtones': [((1, 1), 5757.374, 5390.225), ((1, 1, 1), 8636.061, 7990.237), ((2, 2), 3640.831, 3570.468), ((2, 2, 2), 5461.247, 5327.784), ((3, 3), 3069.097, 3004.865), ((3, 3, 3), 4603.646, 4509.837), ((4, 4), 2406.357, 2364.564), ((4, 4, 4), 3609.536, 3537.828), ((5, 5), 5867.051, 5428.69), ((5, 5, 5), 8800.577, 8032.189), ((6, 6), 2537.819, 2491.481), ((6, 6, 6), 3806.729, 3730.809)], 
    'Combination Bands': [((2, 1), 4699.103, 4522.216), ((2, 2, 1), 6519.518, 6299.008), ((2, 1, 1), 7577.79, 7186.492), ((3, 1), 4413.236, 4197.002), ((3, 3, 1), 5947.784, 5668.884), ((3, 1, 1), 7291.923, 6829.018), ((3, 2), 3354.964, 3290.38), ((3, 3, 2), 4889.513, 4787.912), ((3, 2, 2), 5175.38, 5060.561), ((3, 2, 1), 6233.651, 5986.659), ((4, 1), 4081.866, 3905.476), ((4, 4, 1), 5285.044, 5078.128), ((4, 1, 1), 6960.553, 6562.264), ((4, 2), 3023.594, 2974.691), ((4, 4, 2), 4226.773, 4148.83), ((4, 2, 2), 4844.01, 4745.482), ((4, 2, 1), 5902.281, 5695.743), ((4, 3), 2737.727, 2687.219), ((4, 4, 3), 3940.906, 3866.839), ((4, 3, 3), 4272.276, 4190.842), ((4, 3, 1), 5616.414, 5376.01), ((4, 3, 2), 4558.143, 4470.875), ((5, 1), 5812.213, 5345.29), ((5, 5, 1), 8745.739, 7889.869), ((5, 1, 1), 8690.9, 7875.885), ((5, 2), 4753.941, 4547.872), ((5, 5, 2), 7687.467, 7227.306), ((5, 2, 2), 6574.357, 6325.838), ((5, 2, 1), 7632.628, 7142.731), ((5, 3), 4468.074, 4217.292), ((5, 5, 3), 7401.6, 6859.101), ((5, 3, 3), 6002.623, 5684.984), ((5, 3, 1), 7346.761, 6779.891), ((5, 3, 2), 6288.49, 6008.124), ((5, 4), 4136.704, 3916.796), ((5, 5, 4), 7070.23, 6574.407), ((5, 4, 4), 5339.883, 5076.287), ((5, 4, 1), 7015.391, 6504.168), ((5, 4, 2), 5957.12, 5708.237), ((5, 4, 3), 5671.253, 5383.14), ((6, 1), 4147.597, 3967.717), ((6, 6, 1), 5416.506, 5204.346), ((6, 1, 1), 7026.284, 6624.156), ((6, 2), 3089.325, 3043.988), ((6, 6, 2), 4358.235, 4272.935), ((6, 2, 2), 4909.741, 4805.26), ((6, 2, 1), 5968.012, 5756.577), ((6, 3), 2803.458, 2812.955), ((6, 6, 3), 4072.368, 3999.219), ((6, 3, 3), 4338.007, 4258.895), ((6, 3, 1), 5682.145, 5440.983), ((6, 3, 2), 4623.874, 4534.79), ((6, 4), 2472.088, 2440.61), ((6, 6, 4), 3740.998, 3691.657), ((6, 4, 4), 3675.267, 3627.33), ((6, 4, 1), 5350.775, 5153.825), ((6, 4, 2), 4292.504, 4223.47), ((6, 4, 3), 4006.637, 3945.617), ((6, 5), 4202.435, 3968.654), ((6, 6, 5), 5471.345, 5181.74), ((6, 5, 5), 7135.961, 6615.533), ((6, 5, 1), 7081.122, 6555.676), ((6, 5, 2), 6022.851, 5758.689), ((6, 5, 3), 5736.984, 5437.729), ((6, 5, 4), 5405.614, 5141.601)]}
    """
    d_harm = {}
    d_anharm = {}
    
    for key in parsed_dict:
        for tuple_modes, harm, anharm in parsed_dict[key]:
            d_harm[tuple_modes] = harm
            d_anharm[tuple_modes] = anharm
    
    return d_harm, d_anharm