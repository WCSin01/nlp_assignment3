import pickle

def pickle_dump(obj, filepath: str):
  f = open(filepath, "wb")
  pickle.dump(obj, f)
  f.close()

def pickle_load(filepath: str):
  f = open(filepath, "rb")
  return pickle.load(f)

def flatten(xss):
    return [x for xs in xss for x in xs]