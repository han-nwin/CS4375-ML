import matplotlib.pyplot as plt


def plot_data_from_file(filename):
    # Read data from file
    x = []
    y = []
    with open(filename, "r") as file:
        for line in file:
            # Assuming each line contains two numbers separated by whitespace
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    x.append(x_val)
                    y.append(y_val)
                except ValueError:
                    # Skip lines that don't contain valid floats
                    continue

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot of data from file: " + filename)
    plt.grid(True)
    plt.show()


import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    plot_data_from_file(filename)
