import multiprocessing

class Configuration:
    def __init__(self, configJson):

        manager = multiprocessing.Manager()
        config = manager.dict()

        self._manager = manager
        self._config = config

        for section_key, section_data in configJson.items():
            for key, value in section_data.items():
                print(f"{section_key}.{key}")
                self._config[f"{section_key}.{key}"] = manager.Value(str(type(value)), value)

    def __getitem__(self, key):
        return self._config[key].value

    def __setitem__(self, key, value):
        self._config[key].value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._manager.shutdown()
        
