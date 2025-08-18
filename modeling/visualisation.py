import polars as pl
import numpy as np
from typing import Literal, List, Dict, Set, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from scipy.stats import gaussian_kde


def plot_bar_graph(
    data: pl.DataFrame, x: str, y: str, title: str, color: Optional[str] = None
) -> None:
    """
    Plots a bar graph using Plotly and a Polars DataFrame.
    Parameters:
    - data (pl.DataFrame): The input Polars DataFrame.
    - x (str): The column name to be used for the x-axis.
    - y (str): The column name to be used for the y-axis.
    - title (str): The title of the graph.
    - color (Optional[str]): The column name to color the bars by (default is None).
    Returns:
    - None
    """
    fig = px.bar(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        barmode="group",
        text=y,
    )
    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        }
    )
    fig.update_traces(textposition="inside", textfont=dict(size=14, color="black"))
    fig.show()


def plot_10_box_plot(data: pl.DataFrame, target: str = "ER") -> None:
    """
    Plots box plots for continuous features in the dataset, grouped by a binary target variable.
    Args:
        data (pl.DataFrame): The input dataset containing numeric features and a binary target.
        target (str, optional): The binary target variable. Defaults to "ER".
    Returns:
        None: Displays the generated box plots.
    """

    feature_columns: List[str] = data.columns[:-1]
    binary_variable: str = target

    fig = make_subplots(rows=2, cols=5, subplot_titles=feature_columns)

    for i, feature in enumerate(feature_columns):
        row, col = (i // 5) + 1, (i % 5) + 1
        fig.add_trace(
            go.Box(
                y=data[feature].to_list(),
                x=[str(x) for x in data[binary_variable].to_list()],
                name=feature,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Boxplots of Continuous Features by {target}",
        height=800,
        width=1200,
        showlegend=False,
    )

    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        }
    )

    fig.show()


def plot_10_kde_plots(data: pl.DataFrame, target: str = "ER") -> None:
    """
    Plots KDE plots for continuous features in the dataset, grouped by a binary target variable.
    Args:
        data (pl.DataFrame): The input dataset containing numeric features and a binary target.
        target (str, optional): The binary target variable. Defaults to "ER".
    Returns:
        None: Displays the generated KDE plots.
    """

    feature_columns: List[str] = data.columns[:-1]
    binary_variable: str = target

    fig = make_subplots(rows=2, cols=5, subplot_titles=feature_columns)

    unique_classes = data[binary_variable].unique().to_list()

    colors = ["blue", "red"]

    for i, feature in enumerate(feature_columns):
        row, col = (i // 5) + 1, (i % 5) + 1

        for j, cls in enumerate(unique_classes):
            feature_values = data.filter(pl.col(binary_variable) == cls)[feature].to_list()

            if len(feature_values) > 1:
                kde = gaussian_kde(feature_values)
                x_range = np.linspace(min(feature_values), max(feature_values), 100)
                y_density = kde(x_range)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_density,
                        mode="lines",
                        name=f"{feature} (Class {cls})",
                        line=dict(color=colors[j]),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=f"KDE Plots of Continuous Features by {target}",
        height=800,
        width=1200,
        showlegend=True,
    )

    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        }
    )

    fig.show()


def plot_scatter_graph(
    data: pl.DataFrame, x: str, y: str, color: Optional[str] = "ER", title: str = ""
) -> None:
    """
    Plots a scatter graph using Plotly and a Polars DataFrame.
    Parameters:
    - data (pl.DataFrame): The input Polars DataFrame.
    - x (str): The column name for the x-axis.
    - y (str): The column name for the y-axis.
    - color (Optional[str]): The column name for coloring points (default: "ER").
    - title (str): The title of the graph.
    Returns:
    - None
    """

    # Create a Plotly scatter plot
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        labels={"color": color},
        title=title,
    )

    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        }
    )

    # Show the plot
    fig.show()


def plot_auc(
    fpr: List[float], tpr: List[float], auc: float, model_name: str, approach: str
) -> None:
    """
    Plots an ROC curve using Plotly.
    Parameters:
    - fpr (List[float]): A list of false positive rate values.
    - tpr (List[float]): A list of true positive rate values.
    - auc (float): The area under the curve (AUC) score.
    - model_name (str): The name of the model being evaluated.
    - approach (str): The name of approach used.
    Returns:
    - None
    """

    fig = go.Figure()

    # Add the ROC curve for the given model
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{model_name} (AUC = {auc:.4f})",  # Format AUC value to 4 decimal places
            line=dict(color="blue"),  # Customize line color for better visibility
        )
    )

    # Add a diagonal reference line for a random classifier (baseline)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="red")
        )
    )

    # Update layout with meaningful titles and labels
    fig.update_layout(
        title=f"ROC Curve for {model_name} of {approach}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        width=1100,
        height=700,
    )

    # Center the title and adjust font size
    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        },
        xaxis=dict(title_font=dict(size=20)),
        yaxis=dict(title_font=dict(size=20)),
        legend=dict(font=dict(size=16)),
    )

    fig.show()
