import numpy as np
import pandas as pd

class ModelScorer:
    def __init__(self, alpha=0.5, W_accuracy=0.8, W_time=0.2):
        """
        Initialize the ModelScorer with parameters for accuracy, time sensitivity, and weights.
        
        Parameters:
        alpha (float): Exponent for time sensitivity. Default is 0.5.
        W_accuracy (float): Weight for accuracy in final score calculation. Default is 0.8.
        W_time (float): Weight for response time in final score calculation. Default is 0.2.
        """
        self.alpha = alpha
        self.W_accuracy = W_accuracy
        self.W_time = W_time

    def compute_scores(self, results_df):
        """
        Compute final scores for the model's performance based on accuracy and response time.
        
        Parameters:
        results_df (pd.DataFrame): A dataframe containing the model's results with columns:
                                    'Human_EvaluatorScore', 'Max_Score', and 'response_time'.
                                    
        Returns:
        pd.DataFrame: The original dataframe with the additional columns:
                      - 'Accuracy_Ratio', 'Time_Penalty', 'Final_Score', and 'Normalized_Final_Score'.
        """
        # Ensure necessary columns are present
        required_columns = ['Human_EvaluatorScore', 'Max_Score', 'response_time']
        if not all(col in results_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        # Calculate Accuracy Ratio (Human_EvaluatorScore / Max_Score)
        results_df['Accuracy_Ratio'] = results_df['Human_EvaluatorScore'] / results_df['Max_Score']
        
        # Apply log transformation to response time to compress effect of outliers
        results_df['Time_Penalty'] = 1 / np.log(results_df['response_time'] + 1)
        
        # Calculate Final Score based on weighted sum of accuracy and time penalty
        results_df['Final_Score'] = (self.W_accuracy * results_df['Accuracy_Ratio']) + (self.W_time * results_df['Time_Penalty'])
        
        # Normalize Final Score to range [0, 1]
        max_score = results_df['Final_Score'].max()
        min_score = results_df['Final_Score'].min()
        results_df['Normalized_Final_Score'] = (results_df['Final_Score'] - min_score) / (max_score - min_score)

        return results_df

# Example usage:
# scorer = ModelScorer(alpha=0.5, W_accuracy=0.8, W_time=0.2)
# results_df = scorer.compute_scores(results_df)
# print(results_df[['category', 'Final_Score', 'Normalized_Final_Score']].head())
