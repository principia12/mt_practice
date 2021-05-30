# append path
import sys
import os

src_root_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])

def iter_subdir(root_path):
    yield root_path
    for sib in os.listdir(root_path):
        if not os.path.isfile(os.path.join(root_path, sib)) and not sib.startswith('.'):
            yield from [os.path.join(root_path, p) \
                for p in iter_subdir(os.path.join(root_path, sib))]

for sib in iter_subdir(src_root_path):
    sys.path.append(sib)