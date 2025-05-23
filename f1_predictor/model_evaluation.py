import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

from . import config

def calculate_position_metrics(actual, predicted):
    """
    Calculate evaluation metrics for position predictions.
    
    Args:
        actual: Array-like of actual positions
        predicted: Array-like of predicted positions
        
    Returns:
        Dictionary of metrics
    """
    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Basic regression metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    # Position-specific metrics
    exact_matches = np.mean(actual == predicted.round())
    within_one = np.mean(np.abs(actual - predicted.round()) <= 1)
    within_two = np.mean(np.abs(actual - predicted.round()) <= 2)
    
    # Rank correlation
    # Convert to ranks if not already
    if np.max(actual) <= 20 and np.max(predicted) <= 20:  # Already positions
        actual_ranks = actual
        pred_ranks = np.round(predicted)
    else:  # Convert to ranks
        actual_ranks = pd.Series(actual).rank()
        pred_ranks = pd.Series(predicted).rank()
    
    # Calculate Spearman correlation
    corr = pd.Series(actual_ranks).corr(pd.Series(pred_ranks), method='spearman')
    
    # Top positions accuracy (for podium predictions)
    top3_accuracy = calculate_top_n_accuracy(actual_ranks, pred_ranks, n=3)
    top5_accuracy = calculate_top_n_accuracy(actual_ranks, pred_ranks, n=5)
    top10_accuracy = calculate_top_n_accuracy(actual_ranks, pred_ranks, n=10)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Exact_Match_Rate': exact_matches,
        'Within_One_Position': within_one,
        'Within_Two_Positions': within_two,
        'Rank_Correlation': corr,
        'Top3_Accuracy': top3_accuracy,
        'Top5_Accuracy': top5_accuracy,
        'Top10_Accuracy': top10_accuracy
    }

def calculate_top_n_accuracy(actual_ranks, pred_ranks, n=3):
    """
    Calculate the accuracy of predicting the top N positions.
    
    Args:
        actual_ranks: Array-like of actual positions/ranks
        pred_ranks: Array-like of predicted positions/ranks
        n: Number of top positions to consider
        
    Returns:
        Accuracy score for top N predictions
    """
    # Find the drivers that are actually in the top N
    actual_top_n = set(np.where(actual_ranks <= n)[0])
    
    # Find the drivers predicted to be in the top N
    pred_top_n = set(np.where(pred_ranks <= n)[0])
    
    # Calculate the overlap and accuracy
    overlap = len(actual_top_n.intersection(pred_top_n))
    
    # Return the accuracy (0 to 1)
    return overlap / n if n > 0 else 0

def plot_prediction_vs_actual(actual, predicted, driver_labels=None, title="Predicted vs Actual Positions", save_path=None):
    """
    Create a scatter plot comparing predicted vs actual positions.
    
    Args:
        actual: Array-like of actual positions
        predicted: Array-like of predicted positions
        driver_labels: List of driver names (optional)
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the scatter plot
    scatter = ax.scatter(actual, predicted, alpha=0.7, s=80)
    
    # Add perfect prediction line
    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add driver labels if provided
    if driver_labels is not None:
        for i, label in enumerate(driver_labels):
            ax.annotate(label, (actual[i], predicted[i]), fontsize=9,
                        xytext=(5, 5), textcoords='offset points')
    
    # Add grid, title, and labels
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Actual Position')
    ax.set_ylabel('Predicted Position')
    
    # Set axis limits to be equal and include all data
    padding = 0.5
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    
    # Add metrics text
    metrics = calculate_position_metrics(actual, predicted)
    metrics_text = f"MAE: {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nR²: {metrics['R2']:.2f}\n"
    metrics_text += f"Within 1 Pos: {metrics['Within_One_Position']*100:.1f}%\n"
    metrics_text += f"Top 3 Accuracy: {metrics['Top3_Accuracy']*100:.1f}%"
    
    # Position the text in the bottom right
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return fig

def compare_predictions(with_quali_pred, without_quali_pred, actual_results=None):
    """
    Compare predictions with and without qualifying data.
    
    Args:
        with_quali_pred: DataFrame of predictions with qualifying data
        without_quali_pred: DataFrame of predictions without qualifying data
        actual_results: DataFrame of actual race results (optional)
        
    Returns:
        DataFrame comparing the predictions
    """
    # Merge predictions
    with_quali = with_quali_pred[['Driver', 'Team', 'Predicted_Race_Rank']].copy()
    with_quali = with_quali.rename(columns={'Predicted_Race_Rank': 'With_Quali_Rank'})
    
    without_quali = without_quali_pred[['Driver', 'Predicted_Race_Rank']].copy()
    without_quali = without_quali.rename(columns={'Predicted_Race_Rank': 'Without_Quali_Rank'})
    
    comparison = pd.merge(with_quali, without_quali, on='Driver')
    
    # Calculate rank changes
    comparison['Rank_Change'] = comparison['Without_Quali_Rank'] - comparison['With_Quali_Rank']
    
    # Add actual results if provided
    if actual_results is not None and 'Driver' in actual_results.columns and 'Position' in actual_results.columns:
        actual = actual_results[['Driver', 'Position']].copy()
        actual = actual.rename(columns={'Position': 'Actual_Rank'})
        
        comparison = pd.merge(comparison, actual, on='Driver', how='left')
        
        # Calculate errors
        comparison['With_Quali_Error'] = comparison['With_Quali_Rank'] - comparison['Actual_Rank']
        comparison['Without_Quali_Error'] = comparison['Without_Quali_Rank'] - comparison['Actual_Rank']
        comparison['Error_Improvement'] = abs(comparison['Without_Quali_Error']) - abs(comparison['With_Quali_Error'])
        
        # Calculate overall metrics
        with_quali_metrics = calculate_position_metrics(
            comparison['Actual_Rank'].dropna(),
            comparison['With_Quali_Rank'][comparison['Actual_Rank'].notna()]
        )
        
        without_quali_metrics = calculate_position_metrics(
            comparison['Actual_Rank'].dropna(),
            comparison['Without_Quali_Rank'][comparison['Actual_Rank'].notna()]
        )
        
        print("\n--- Prediction Performance Metrics ---")
        print("\nWith Qualifying Data:")
        for metric, value in with_quali_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nWithout Qualifying Data:")
        for metric, value in without_quali_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Calculate improvement
        mae_improvement = without_quali_metrics['MAE'] - with_quali_metrics['MAE']
        print(f"\nMAE Improvement from Using Qualifying Data: {mae_improvement:.4f}")
    
    # Sort by the with-quali rank
    comparison = comparison.sort_values('With_Quali_Rank')
    
    return comparison

def evaluate_model_on_historical_data(model, features_used, historical_data, encoders, imputation_values):
    """
    Evaluate model performance on historical data where we know the actual results.
    
    Args:
        model: Trained model
        features_used: List of feature columns used by the model
        historical_data: DataFrame of historical race data
        encoders: Dictionary of fitted LabelEncoders
        imputation_values: Dictionary of imputation values
        
    Returns:
        Dictionary of evaluation metrics
    """
    from . import prediction
    
    # Get unique races
    races = historical_data.drop_duplicates(['Year', 'Race_Num', 'Circuit']).copy()
    recent_races = races.sort_values(['Year', 'Race_Num']).tail(10).copy()
    
    all_metrics = []
    
    for idx, race in recent_races.iterrows():
        year = race['Year']
        race_num = race['Race_Num']
        circuit = race['Circuit']
        
        print(f"\nEvaluating model on {year} {circuit} GP (Race {race_num})...")
        
        # Get race data
        race_data = historical_data[
            (historical_data['Year'] == year) & 
            (historical_data['Race_Num'] == race_num)
        ].copy()
        
        # Get qualifying data for this race
        quali_data = race_data[['Driver', 'Quali_Pos']].copy()
        
        # Get actual race results
        actual_results = race_data[['Driver', 'Finish_Pos_Clean']].rename(
            columns={'Finish_Pos_Clean': 'Actual_Pos'}
        )
        
        # Exclude this race from training data
        train_data = historical_data[
            ~((historical_data['Year'] == year) & 
              (historical_data['Race_Num'] == race_num))
        ].copy()
        
        # Predict with and without quali data
        with_quali_pred = prediction.predict_results(
            model=model,
            features_used=features_used,
            upcoming_info=race_data.copy(),
            historical_data=train_data,
            encoders=encoders,
            imputation_values=imputation_values,
            actual_quali=quali_data,
            model_type="race"
        )
        
        without_quali_pred = prediction.predict_results(
            model=model,
            features_used=features_used,
            upcoming_info=race_data.copy(),
            historical_data=train_data,
            encoders=encoders,
            imputation_values=imputation_values,
            actual_quali=None,
            model_type="race"
        )
        
        # Merge with actual results
        with_quali_results = pd.merge(
            with_quali_pred[['Driver', 'Predicted_Race_Rank']], 
            actual_results,
            on='Driver'
        )
        
        without_quali_results = pd.merge(
            without_quali_pred[['Driver', 'Predicted_Race_Rank']], 
            actual_results,
            on='Driver'
        )
        
        # Calculate metrics
        with_quali_metrics = calculate_position_metrics(
            with_quali_results['Actual_Pos'],
            with_quali_results['Predicted_Race_Rank']
        )
        
        without_quali_metrics = calculate_position_metrics(
            without_quali_results['Actual_Pos'],
            without_quali_results['Predicted_Race_Rank']
        )
        
        # Store metrics
        race_metrics = {
            'Year': year,
            'Race_Num': race_num,
            'Circuit': circuit,
            'With_Quali_MAE': with_quali_metrics['MAE'],
            'Without_Quali_MAE': without_quali_metrics['MAE'],
            'MAE_Improvement': without_quali_metrics['MAE'] - with_quali_metrics['MAE'],
            'With_Quali_Top3': with_quali_metrics['Top3_Accuracy'],
            'Without_Quali_Top3': without_quali_metrics['Top3_Accuracy'],
            'Top3_Improvement': with_quali_metrics['Top3_Accuracy'] - without_quali_metrics['Top3_Accuracy']
        }
        
        all_metrics.append(race_metrics)
        
        # Print metrics
        print(f"With Qualifying Data - MAE: {with_quali_metrics['MAE']:.4f}, Top3: {with_quali_metrics['Top3_Accuracy']:.4f}")
        print(f"Without Qualifying Data - MAE: {without_quali_metrics['MAE']:.4f}, Top3: {without_quali_metrics['Top3_Accuracy']:.4f}")
        print(f"MAE Improvement: {race_metrics['MAE_Improvement']:.4f}")
        
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate average metrics
    avg_metrics = {
        'Avg_With_Quali_MAE': metrics_df['With_Quali_MAE'].mean(),
        'Avg_Without_Quali_MAE': metrics_df['Without_Quali_MAE'].mean(),
        'Avg_MAE_Improvement': metrics_df['MAE_Improvement'].mean(),
        'Avg_With_Quali_Top3': metrics_df['With_Quali_Top3'].mean(),
        'Avg_Without_Quali_Top3': metrics_df['Without_Quali_Top3'].mean(),
        'Avg_Top3_Improvement': metrics_df['Top3_Improvement'].mean()
    }
    
    print("\n--- Average Metrics Across Historical Races ---")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return {
        'race_metrics': metrics_df,
        'average_metrics': avg_metrics
    } 