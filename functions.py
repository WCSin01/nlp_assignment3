import pickle
import matplotlib.pyplot as plt

def pickle_dump(obj, filepath: str):
    f = open(filepath, "wb")
    pickle.dump(obj, f)
    f.close()

def pickle_load(filepath: str):
    f = open(filepath, "rb")
    return pickle.load(f)

def flatten(xss):
    return [x for xs in xss for x in xs]

def plot_hist(values, bins, density: bool, title, x_label, y_label):
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, density=density)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()

def plot_2dhist(x_values, y_values, bins, title, x_label, y_label):
    fig, ax = plt.subplots()
    ax.hist2d(x_values, y_values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()