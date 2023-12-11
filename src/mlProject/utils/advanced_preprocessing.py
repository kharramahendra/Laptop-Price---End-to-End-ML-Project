import re

def parse_processor_name(processor_name):
    # Define regular expressions for extracting information
    regexes = [
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel|AMD) (Core|i\d+|Celeron|Pentium|Atom|Ryzen|Athlon) ?(\w*)'),
        re.compile(r'(Apple) (M1|M2(?: Pro)?(?: Max)?)'),
        re.compile(r'(Intel) (Celeron|Pentium|Atom) (\w+)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Celeron) (\w+)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Pentium) (\w+)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Core) (i\d+) (\w*)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Core) (i\d+)'),
    ]

    # Match the regular expressions against the processor name
    for regex in regexes:
        match = regex.match(processor_name)
        if match:
            groups = match.groups()
            if groups[0] == 'Apple':
                return {'generation':'1','company': groups[0],'model_type': 'M1', 'version': groups[1]}
            elif groups[0] == 'Intel':
                if groups[2] in ['Celeron', 'Pentium', 'Atom']:
                    return {'generation': groups[1], 'company': groups[0], 'model_type': groups[2], 'version': groups[3]}
                elif groups[2] == 'Core':
                    return {'generation': groups[1], 'company': groups[0], 'model_type': f'{groups[2]} {groups[4]}', 'version': groups[5]}
                else:
                    return None
            else:
                return {'generation': groups[0], 'company': groups[1], 'model_type': groups[2], 'version': groups[3]}

    return None



def get_gpu_type(gpu_name):
    # Define regular expressions for extracting GPU type information
    regexes = [
        re.compile(r'(NVIDIA|AMD)\s*(Radeon)?'),
        re.compile(r'(Apple)\s*(Integrated Graphics)'),
        re.compile(r'(Intel)\s*(Iris Xe Graphics|UHD Graphics|HD Graphics|Graphics)?'),
        re.compile(r'(ARM)\s*(Mali G\d+)'),
    ]

    # Match the regular expressions against the GPU name
    for regex in regexes:
        match = regex.search(gpu_name)
        if match:
            groups = match.groups()
            gpu_type = groups[1] if len(groups) > 1 and groups[1] else groups[0] if groups[0] else None
            return gpu_type

    return None


def extract_cores_threads(cpu_name):
    # Check for the presence of Cores and Threads in the name
    cores_match = re.search(r'(\d+|Dual|Quad|Hexa|Octa)\s*Cores?', cpu_name)
    threads_match = re.search(r'(\d+)\s*Threads?', cpu_name)

    # Extract the number of cores and threads from the matches
    cores = 0 if cores_match is None else cores_match.group(1)
    threads = 0 if threads_match is None else threads_match.group(1)

    # Convert 'Dual', 'Quad', 'Hexa', 'Octa' to corresponding numbers
    cores_dict = {'Dual': 2, 'Quad': 4, 'Hexa': 6, 'Octa': 8}
    cores = cores_dict.get(cores, cores)

    return int(cores), int(threads)