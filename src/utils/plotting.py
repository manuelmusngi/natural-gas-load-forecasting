import matplotlib.pyplot as plt

def plot_forecast(actual, predicted, title: str = "Forecast vs Actual"):
    plt.figure(figsize=(12, 4))
    plt.plot(actual.index, actual.values, label="Actual")
    plt.plot(predicted.index, predicted.values, label="Predicted")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    return plt
