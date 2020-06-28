import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from strategies.run_zipline import run_strategy


def main():
    print("*** TFM Miguel Sánchez Mendoza ***")
    print("*** Máster en Ingelitencia Artificial Avanzada ***")
    perf = run_strategy("buy_and_hold")
    perf.to_csv("buy_and_holdX.csv")


if __name__ == '__main__':
    main()
