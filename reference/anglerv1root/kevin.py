import anglerdroid
import json

with open('kevin_config.json') as f:
    config=json.load(f)

with anglerdroid.Brain(config) as kevin:
    kevin.wake()