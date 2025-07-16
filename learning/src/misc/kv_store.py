import os
import json

class KVStore:
    def __init__(self):
        self.kv_store = {}

        if os.path.exists("kv_store.txt"):
            with open("kv_store.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    json_obj = json.loads(line)
                    
                    for key, value in json_obj.items():
                        self.kv_store[key] = value


    def get(self, key):
        return self.kv_store.get(key, None)

    def put(self, key, value):
        self.kv_store[key] = value

        with open("kv_store.txt", "a") as f:
            new_dict = {key: value}
            f.write(json.dumps(new_dict) + "\n")


def main():
    kv_store = KVStore()
    kv_store.put("key1", "value1")
    kv_store.put("key2", "value2")
    kv_store.put("key2", "value3")
    print(kv_store.get("key1"))
    print(kv_store.get("key2"))

    del kv_store
    kv_store = KVStore()
    print(kv_store.get("key1"))
    print(kv_store.get("key2"))

    kv_store.put("key2", "value4")
    print(kv_store.get("key2"))


if __name__ == "__main__":
    main()