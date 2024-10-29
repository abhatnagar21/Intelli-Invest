import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to load mutual fund data from CSV
def load_csv_data(filename="MF - Sheet1 (1).csv"):
    try:
        df = pd.read_csv(filename)
        # Check if required columns exist
        required_columns = [
            "Fund Name", "P/E Ratio", "P/B Ratio", "Alpha",
            "Beta", "Sharpe Ratio", "Sortino Ratio",
            "Top 5 Rating", "Top 20 Rating", "Expense Ratio",
            "Exit Load", "Exit Load Duration", "1 Year Return",
            "3 Year Return", "5 Year Return"
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except ValueError as ve:
        print(ve)
        return None

# Function to calculate suitability score with user-defined weights
def calculate_suitability_score(df, weights):
    df["Suitability Score"] = (
        weights["lower_is_better"] * (1 / (df["P/E Ratio"] + df["P/B Ratio"] +
                                              df["Expense Ratio"] + df["Exit Load"] +
                                              df["Exit Load Duration"] / 365))
        + weights["alpha"] * df["Alpha"]
        + weights["beta"] * (1 - df["Beta"])
        + weights["sharpe"] * df["Sharpe Ratio"]
        + weights["sortino"] * df["Sortino Ratio"]
        + weights["top_5"] * df["Top 5 Rating"]
        + weights["top_20"] * df["Top 20 Rating"]
        + weights["one_year"] * df["1 Year Return"]
        + weights["three_year"] * df["3 Year Return"]
        + weights["five_year"] * df["5 Year Return"]
    )

    # Sort funds by suitability score in descending order
    df.sort_values(by="Suitability Score", ascending=False, inplace=True)

    return df

# Function to plot the mutual funds' suitability
def plot_suitability(df):
    plt.figure(figsize=(10, 6))
    plt.barh(df["Fund Name"], df["Suitability Score"], color="skyblue")
    plt.xlabel("Suitability Score")
    plt.title("Mutual Funds Ranked by Suitability")
    plt.gca().invert_yaxis()  # Highest score on top
    plt.show()

# Function to plot return comparisons
def plot_return_comparison(df):
    plt.figure(figsize=(12, 8))
    x = range(len(df))
    plt.bar(x, df["1 Year Return"], width=0.2, label='1 Year Return', color='orange', align='center')
    plt.bar([i + 0.2 for i in x], df["3 Year Return"], width=0.2, label='3 Year Return', color='green', align='center')
    plt.bar([i + 0.4 for i in x], df["5 Year Return"], width=0.2, label='5 Year Return', color='blue', align='center')

    plt.xticks([i + 0.2 for i in x], df["Fund Name"], rotation=45, ha='right')
    plt.xlabel("Mutual Funds")
    plt.ylabel("Returns (%)")
    plt.title("Comparison of Returns for Mutual Funds")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to plot feature importance
def plot_feature_importance(df, weights):
    # Prepare features and target for regression
    features = df[[
        "P/E Ratio", "P/B Ratio", "Alpha", "Beta",
        "Sharpe Ratio", "Sortino Ratio",
        "Top 5 Rating", "Top 20 Rating",
        "Expense Ratio", "Exit Load",
        "Exit Load Duration", "1 Year Return",
        "3 Year Return", "5 Year Return"
    ]]

    # Define the target variable
    target = df["Suitability Score"]

    # Fit linear regression model
    model = LinearRegression()
    model.fit(features, target)

    # Get feature importance
    importance = model.coef_

    # Plotting feature importance
    plt.figure(figsize=(12, 6))
    plt.barh(features.columns, importance, color='teal')
    plt.xlabel("Coefficient Value")
    plt.title("Feature Importance in Suitability Score Prediction")
    plt.axvline(0, color='grey', lw=0.8)  # Line for zero importance
    plt.show()

# Function to plot scatter plot of two selected metrics
def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], color='purple', alpha=0.6)
    plt.title(f"Scatter Plot: {y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.show()

# Function to save the ranked results to a CSV file
def save_results(df, filename="ranked_mutual_funds.csv"):
    df.to_csv(filename, index=False)
    print(f"Ranked results saved to {filename}")

# Main program
def main():
    # Load fund data from CSV file
    fund_data = load_csv_data()

    if fund_data is None:
        return

    # User-defined weights for each metric
    weights = {
        "lower_is_better": 1,  # Weight for metrics where lower is better
        "alpha": 1,
        "beta": 1,
        "sharpe": 1,
        "sortino": 1,
        "top_5": 0.5,
        "top_20": 0.3,
        "one_year": 1,  # Weight for 1 year return
        "three_year": 1,  # Weight for 3 year return
        "five_year": 1,  # Weight for 5 year return
    }

    # Calculate suitability scores and rank
    ranked_funds = calculate_suitability_score(fund_data, weights)

    # Display ranked data on console
    print("\nRanked Mutual Funds by Suitability:")
    print(ranked_funds[["Fund Name", "Suitability Score"]])

    # Plot the suitability scores
    plot_suitability(ranked_funds)

    # Plot return comparisons
    plot_return_comparison(ranked_funds)

    # Plot feature importance
    plot_feature_importance(ranked_funds, weights)

    # Scatter plot of Sharpe Ratio vs Alpha
    plot_scatter(ranked_funds, "Sharpe Ratio", "Alpha")

    # Save results to a new CSV file
    save_results(ranked_funds)

# Execute the main program
if __name__ == "__main__":
    main()
