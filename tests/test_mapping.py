import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mapping import Mapping


def test_update_map():
    mapping = Mapping()
    pose = [0, 0, 0]
    observations = []
    result = mapping.update_map(pose, observations)
    assert result is None
