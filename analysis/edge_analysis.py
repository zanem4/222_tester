import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gamma, lognorm, norm, expon, weibull_min
from scipy.integrate import trapezoid
from sklearn.mixture import GaussianMixture
import os
from pathlib import Path
from scipy.optimize import fsolve

class DistributionalEdgeAnalyzer:
    """
    Advanced edge analysis using distribution fitting and overlap calculations
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.results = {}
        self.current_metric = None
        self.clean_data()
    
    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Convert win_loss to boolean
        self.df['win_loss'] = self.df['win_loss'].astype(bool)
        
        # Remove any rows with NaN values in key columns
        self.df = self.df.dropna(subset=['win_loss'])
    
    def get_metric_columns(self):
        """Get all numeric metric columns for analysis"""
        return [
            'price_delta', 'sma', 'price_range', 'atr',
            'body1_size', 'body2_size', 'body3_size',
            'upwick1', 'downwick1', 'upwick2', 'downwick2', 'upwick3', 'downwick3',
            'norm_price_delta', 'norm_price_range', 'norm_body1', 'norm_body2', 'norm_body3',
            'norm_upwick1', 'norm_downwick1', 'norm_upwick2', 'norm_downwick2', 'norm_upwick3', 'norm_downwick3',
            'trade_duration_bars', 'max_drawdown_pct', 'max_profit_before_stop_pips'
        ]
    
    def fit_distributions(self, data, distributions=None):
        """Fit multiple distributions and return the best one"""
        if distributions is None:
            distributions = [
                ('normal', norm, 'normal'),
                ('gamma', gamma, 'gamma'),
                ('lognorm', lognorm, 'lognorm'),
                ('exponential', expon, 'exponential'),
                ('weibull', weibull_min, 'weibull')
            ]
        
        best_fit = None
        best_aic = float('inf')
        
        for name, dist, dist_name in distributions:
            try:
                # Fit the distribution
                params = dist.fit(data)
                aic = self._calculate_aic(data, dist, params)
                
                if aic < best_aic:
                    best_aic = aic
                    best_fit = (dist_name, dist, params)
            except Exception as e:
                continue
        
        return best_fit
    
    def _calculate_aic(self, data, dist, params):
        """Calculate AIC for distribution fit"""
        try:
            # Determine number of shape parameters
            n_shapes = 0
            if dist.shapes:
                n_shapes = len(dist.shapes.split(','))
            
            # Calculate log-likelihood
            if len(params) == 0:
                log_likelihood = np.sum(dist.logpdf(data))
            elif len(params) == 1:
                if n_shapes == 0:
                    log_likelihood = np.sum(dist.logpdf(data, loc=params[0]))
                else:
                    log_likelihood = np.sum(dist.logpdf(data, params[0]))
            elif len(params) == 2:
                if n_shapes == 0:
                    log_likelihood = np.sum(dist.logpdf(data, loc=params[0], scale=params[1]))
                elif n_shapes == 1:
                    log_likelihood = np.sum(dist.logpdf(data, params[0], loc=params[1]))
                else:
                    log_likelihood = np.sum(dist.logpdf(data, params[0], params[1]))
            else:
                shapes = params[:n_shapes]
                loc = params[n_shapes] if len(params) > n_shapes else 0
                scale = params[n_shapes + 1] if len(params) > n_shapes + 1 else 1
                log_likelihood = np.sum(dist.logpdf(data, *shapes, loc=loc, scale=scale))
            
            # AIC = 2k - 2ln(L) where k is number of parameters
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            return aic
        except:
            return float('inf')
    
    def calculate_distribution_overlap(self, dist1_params, dist2_params, dist_type, x_range):
        """Calculate area between two fitted distributions"""
        name, dist, _ = dist_type

        # Generate points for integration
        x = np.linspace(x_range[0], x_range[1], 1000)

        def get_pdf(params):
            # Determine number of shape parameters
            n_shapes = 0
            if dist.shapes:
                n_shapes = len(dist.shapes.split(','))
            
            # Handle different parameter structures
            if len(params) == 0:
                return dist.pdf(x)
            elif len(params) == 1:
                if n_shapes == 0:
                    return dist.pdf(x, loc=params[0])
                else:
                    return dist.pdf(x, params[0])
            elif len(params) == 2:
                if n_shapes == 0:
                    return dist.pdf(x, loc=params[0], scale=params[1])
                elif n_shapes == 1:
                    return dist.pdf(x, params[0], loc=params[1])
                else:
                    return dist.pdf(x, params[0], params[1])
            else:
                shapes = params[:n_shapes]
                loc = params[n_shapes] if len(params) > n_shapes else 0
                scale = params[n_shapes + 1] if len(params) > n_shapes + 1 else 1
                return dist.pdf(x, *shapes, loc=loc, scale=scale)

        # Calculate PDFs
        pdf1 = get_pdf(dist1_params)
        pdf2 = get_pdf(dist2_params)
        
        # Handle any NaN or infinite values
        pdf1 = np.nan_to_num(pdf1, nan=0.0, posinf=0.0, neginf=0.0)
        pdf2 = np.nan_to_num(pdf2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate areas
        overlap_area = trapezoid(np.minimum(pdf1, pdf2), x)
        total_area1 = trapezoid(pdf1, x)
        total_area2 = trapezoid(pdf2, x)
        
        # Edge area (difference)
        edge_area = abs(total_area1 - total_area2) - overlap_area
        
        # Handle division by zero
        min_area = min(total_area1, total_area2)
        max_area = max(total_area1, total_area2)
        
        overlap_ratio = overlap_area / min_area if min_area > 0 else 0
        edge_ratio = edge_area / max_area if max_area > 0 else 0

        return {
            'overlap_area': overlap_area,
            'edge_area': edge_area,
            'overlap_ratio': overlap_ratio,
            'edge_ratio': edge_ratio
        }
    
    def find_intersection_points(self, dist1_params, dist2_params, dist_type, x_range, max_intersections=5):
        """Find intersection points between two distribution curves"""
        name, dist, _ = dist_type
        
        def difference_function(x):
            """Function to find where pdf1 - pdf2 = 0"""
            def get_pdf_at_point(params, x_val):
                n_shapes = 0
                if dist.shapes:
                    n_shapes = len(dist.shapes.split(','))
                
                if len(params) == 0:
                    return dist.pdf(x_val)
                elif len(params) == 1:
                    if n_shapes == 0:
                        return dist.pdf(x_val, loc=params[0])
                    else:
                        return dist.pdf(x_val, params[0])
                elif len(params) == 2:
                    if n_shapes == 0:
                        return dist.pdf(x_val, loc=params[0], scale=params[1])
                    elif n_shapes == 1:
                        return dist.pdf(x_val, params[0], loc=params[1])
                    else:
                        return dist.pdf(x_val, params[0], params[1])
                else:
                    shapes = params[:n_shapes]
                    loc = params[n_shapes] if len(params) > n_shapes else 0
                    scale = params[n_shapes + 1] if len(params) > n_shapes + 1 else 1
                    return dist.pdf(x_val, *shapes, loc=loc, scale=scale)
            
            pdf1 = get_pdf_at_point(dist1_params, x)
            pdf2 = get_pdf_at_point(dist2_params, x)
            return pdf1 - pdf2
        
        # Find intersection points using multiple starting points
        x_min, x_max = x_range
        intersections = []
        
        # Try different starting points across the range
        start_points = np.linspace(x_min, x_max, 20)
        
        for start_point in start_points:
            try:
                intersection = fsolve(difference_function, start_point, full_output=True)
                if intersection[2] == 1:  # fsolve succeeded
                    x_intersect = intersection[0][0]
                    # Check if it's within range and not already found
                    if x_min <= x_intersect <= x_max:
                        # Check if it's close to any existing intersection
                        is_duplicate = any(abs(x_intersect - existing) < (x_max - x_min) * 0.01 
                                         for existing in intersections)
                        if not is_duplicate:
                            intersections.append(x_intersect)
            except:
                continue
        
        # Sort and limit to max_intersections
        intersections = sorted(intersections)[:max_intersections]
        
        # Pad with None if fewer than max_intersections
        while len(intersections) < max_intersections:
            intersections.append(None)
        
        return intersections
    
    def analyze_metric(self, metric_name):
        """Analyze a single metric for distributional edge"""
        print(f"Analyzing {metric_name}...")
        
        # Track current metric for distribution selection
        self.current_metric = metric_name
        
        # Split data by win/loss
        winners = self.df.loc[self.df['win_loss'] == True, metric_name].dropna()
        losers = self.df.loc[self.df['win_loss'] == False, metric_name].dropna()
        
        # For wick metrics, filter out zero values to get better distribution fits
        if 'wick' in metric_name.lower():
            winners = winners[winners > 0]
            losers = losers[losers > 0]
            print(f"  Filtered out zero values for {metric_name}: {len(winners)} winners, {len(losers)} losers")
        
        # Handle problematic metrics with extreme values
        if metric_name in ['max_drawdown_pct', 'max_profit_before_stop_pips']:
            # Remove infinite and extreme values
            winners = winners[np.isfinite(winners)]
            losers = losers[np.isfinite(losers)]
            
            # Remove outliers (values beyond 99th percentile)
            winner_99 = winners.quantile(0.99)
            loser_99 = losers.quantile(0.99)
            winners = winners[winners <= winner_99]
            losers = losers[losers <= loser_99]
            
            print(f"  Filtered extreme values for {metric_name}: {len(winners)} winners, {len(losers)} losers")
        
        if len(winners) < 10 or len(losers) < 10:
            print(f"  Insufficient data for {metric_name}")
            return None
        
        # Fit distributions with error handling
        try:
            winner_fit = self.fit_distributions(winners)
            loser_fit = self.fit_distributions(losers)
        except Exception as e:
            print(f"  Distribution fitting failed for {metric_name}: {e}")
            return None
        
        if winner_fit is None or loser_fit is None:
            print(f"  Could not fit distributions for {metric_name}")
            return None
        
        # Calculate overlap with error handling
        try:
            x_range = (min(winners.min(), losers.min()), max(winners.max(), losers.max()))
            overlap_results = self.calculate_distribution_overlap(
                winner_fit[2], loser_fit[2], winner_fit, x_range
            )
            
            # Find intersection points
            intersection_points = self.find_intersection_points(
                winner_fit[2], loser_fit[2], winner_fit, x_range
            )
        except Exception as e:
            print(f"  Overlap calculation failed for {metric_name}: {e}")
            return None
        
        # Store results
        result = {
            'metric': metric_name,
            'winner_count': len(winners),
            'loser_count': len(losers),
            'winner_fit': winner_fit,
            'loser_fit': loser_fit,
            'winner_stats': {
                'mean': winners.mean(),
                'std': winners.std(),
                'median': winners.median()
            },
            'loser_stats': {
                'mean': losers.mean(),
                'std': losers.std(),
                'median': losers.median()
            },
            'overlap': overlap_results,
            'x_range': x_range,
            'intersection_points': intersection_points
        }
        
        self.results[metric_name] = result
        return result
    
    def plot_distribution_comparison(self, metric_name, save_path=None):
        """Plot winner vs loser distributions with fitted curves"""
        if metric_name not in self.results:
            print(f"No results for {metric_name}")
            return
        
        result = self.results[metric_name]
        
        # Get data
        winners = self.df.loc[self.df['win_loss'] == True, metric_name].dropna()
        losers = self.df.loc[self.df['win_loss'] == False, metric_name].dropna()
        
        # Set dark theme
        plt.style.use('dark_background')
        
        # Create plot with larger size for better readability
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Calculate better x-range for scaling (focus on main distribution area)
        winner_q1, winner_q3 = winners.quantile([0.25, 0.75])
        loser_q1, loser_q3 = losers.quantile([0.25, 0.75])
        
        # Use 95th percentile to avoid extreme outliers
        x_min = min(winners.quantile(0.05), losers.quantile(0.05))
        x_max = max(winners.quantile(0.95), losers.quantile(0.95))
        
        # Histogram comparison with better scaling
        ax1.hist(winners, bins=30, alpha=0.6, density=True, label='Winners', color='lime', range=(x_min, x_max))
        ax1.hist(losers, bins=30, alpha=0.6, density=True, label='Losers', color='red', range=(x_min, x_max))
        ax1.set_title(f'{metric_name} - Histogram Comparison', fontsize=14, fontweight='bold', color='white')
        ax1.set_xlabel(metric_name, fontsize=12, color='white')
        ax1.set_ylabel('Density', fontsize=12, color='white')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.tick_params(colors='white')
        
        # Fitted distributions with better x-range
        x = np.linspace(x_min, x_max, 500)
        
        # Helper function to get PDF with proper parameter handling
        def get_pdf_for_plotting(dist, params, x):
            n_shapes = 0
            if dist.shapes:
                n_shapes = len(dist.shapes.split(','))
            shapes = params[:n_shapes]
            loc = params[n_shapes] if len(params) > n_shapes else 0
            scale = params[n_shapes + 1] if len(params) > n_shapes + 1 else 1
            return dist.pdf(x, *shapes, loc=loc, scale=scale)
        
        # Plot winner distribution with parameters
        winner_name, winner_dist, winner_params = result['winner_fit']
        winner_pdf = get_pdf_for_plotting(winner_dist, winner_params, x)
        
        # Format parameters for display
        winner_param_str = self._format_distribution_params(winner_name, winner_params)
        ax2.plot(x, winner_pdf, 'lime', linewidth=2, label=f'Winners ({winner_name})\n{winner_param_str}')
        
        # Plot loser distribution with parameters
        loser_name, loser_dist, loser_params = result['loser_fit']
        loser_pdf = get_pdf_for_plotting(loser_dist, loser_params, x)
        
        # Format parameters for display
        loser_param_str = self._format_distribution_params(loser_name, loser_params)
        ax2.plot(x, loser_pdf, 'red', linewidth=2, label=f'Losers ({loser_name})\n{loser_param_str}')
        
        # Plot intersection points if any
        intersection_points = result.get('intersection_points', [])
        for i, point in enumerate(intersection_points):
            if point is not None and x_min <= point <= x_max:
                # Get PDF values at intersection point
                winner_pdf_val = get_pdf_for_plotting(winner_dist, winner_params, point)
                ax2.plot(point, winner_pdf_val, 'yellow', markersize=8, label=f'Intersection {i+1}' if i == 0 else "")
                ax2.annotate(f'I{i+1}: {point:.6f}', (point, winner_pdf_val), 
                           xytext=(10, 10), textcoords='offset points', fontsize=10, color='white')
        
        ax2.set_title(f'{metric_name} - Fitted Distributions', fontsize=14, fontweight='bold', color='white')
        ax2.set_xlabel(metric_name, fontsize=12, color='white')
        ax2.set_ylabel('Probability Density', fontsize=12, color='white')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
            print(f"Distribution plot saved to: {save_path}")
        
        # Close the figure to prevent popup and free memory
        plt.close(fig)

    def _format_distribution_params(self, dist_name, params):
        """Format distribution parameters for display with high precision"""
        if dist_name == 'normal':
            return f"μ={params[0]:.12f}, σ={params[1]:.12f}"
        elif dist_name == 'gamma':
            return f"α={params[0]:.12f}, β={params[1]:.12f}"
        elif dist_name == 'lognorm':
            return f"s={params[0]:.12f}, scale={params[1]:.12f}"
        elif dist_name == 'exponential':
            return f"λ={params[0]:.12f}"
        elif dist_name == 'weibull':
            return f"c={params[0]:.12f}, scale={params[1]:.12f}"
        else:
            return f"params: {[f'{p:.12f}' for p in params[:3]]}"  # Show first 3 parameters
    
    def _get_distribution_param_count(self, dist_name):
        """Get the number of parameters for a distribution"""
        if dist_name == 'normal':
            return 2
        elif dist_name == 'gamma':
            return 2
        elif dist_name == 'lognorm':
            return 2
        elif dist_name == 'exponential':
            return 1
        elif dist_name == 'weibull':
            return 2
        else:
            return 2  # Default to 2
    
    def generate_edge_report(self, output_dir="edge_analysis"):
        """Generate comprehensive edge analysis report"""
        # Create the edge_distributions folder in the same location as histograms
        base_dir = os.path.dirname(os.path.abspath(self.csv_path))
        edge_distributions_dir = os.path.join(base_dir, "edge_distributions")
        os.makedirs(edge_distributions_dir, exist_ok=True)
        
        # Analyze all metrics
        metrics = self.get_metric_columns()
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        print(f"Analyzing {len(available_metrics)} metrics...")
        
        for metric in available_metrics:
            self.analyze_metric(metric)
        
        # Create summary report
        self.create_summary_report(edge_distributions_dir)
        
        # Generate plots and save to edge_distributions folder
        for metric in available_metrics:
            if metric in self.results:
                plot_path = os.path.join(edge_distributions_dir, f"{metric}_edge_analysis.png")
                self.plot_distribution_comparison(metric, plot_path)
        
        print(f"Edge analysis complete. Results saved to: {edge_distributions_dir}")
        print(f"Distribution charts saved to: {edge_distributions_dir}")
    
    def create_summary_report(self, output_dir):
        """Create a summary report of all edge findings"""
        report_data = []
        
        for metric, result in self.results.items():
            winner_name, _, winner_params = result['winner_fit']
            loser_name, _, loser_params = result['loser_fit']
            
            # Get parameter counts
            winner_param_count = self._get_distribution_param_count(winner_name)
            loser_param_count = self._get_distribution_param_count(loser_name)
            
            # Create row data
            row_data = {
                'name': metric,
                'distribution_1_name': winner_name,
                'distribution_2_name': loser_name,
                'distribution_1_parameter_1': winner_params[0] if len(winner_params) > 0 else None,
                'distribution_1_parameter_2': winner_params[1] if len(winner_params) > 1 else "-",
                'distribution_2_parameter_1': loser_params[0] if len(loser_params) > 0 else None,
                'distribution_2_parameter_2': loser_params[1] if len(loser_params) > 1 else "-"
            }
            
            # Add intersection points
            intersection_points = result.get('intersection_points', [])
            for i, point in enumerate(intersection_points, 1):
                row_data[f'intersection_point_{i}'] = point if point is not None else "-"
            
            report_data.append(row_data)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Save report with high precision
        report_path = os.path.join(output_dir, "edge_analysis_summary.csv")
        report_df.to_csv(report_path, index=False, float_format='%.12f')
        
        # Print top findings
        print("\n" + "="*80)
        print("DISTRIBUTION ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {report_path}")
        print(f"Total metrics analyzed: {len(report_data)}")
        
        return report_df

# Usage example
if __name__ == "__main__":
    import sys
    
    # Check if CSV path provided as command line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path if no argument provided
        csv_path = r"C:\Users\zanem\OneDrive\Desktop\Personal\222_Backtester\backtester\analysis\EUR_USD_20250717_225042\EUR_USD_2014_2023_fixed.csv"
    
    print(f"Analyzing CSV file: {csv_path}")
    
    analyzer = DistributionalEdgeAnalyzer(csv_path)
    analyzer.generate_edge_report("edge_analysis_results")