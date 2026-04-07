#!/usr/bin/env python3
"""
Model Evaluation Module

This module provides comprehensive evaluation tools for trained models,
including accuracy metrics, performance analysis, and comparison tools.
"""

import os
import cv2
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    This class provides:
    - Accuracy and performance metrics
    - Visualization of results
    - Model comparison tools
    - Error analysis
    """
    
    def __init__(self, 
                 results_dir: str = "evaluation_results",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
        """
        Initialize the model evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Create subdirectories
        self.plots_dir = self.results_dir / "plots"
        self.reports_dir = self.results_dir / "reports"
        self.comparisons_dir = self.results_dir / "comparisons"
        
        for dir_path in [self.plots_dir, self.reports_dir, self.comparisons_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"ModelEvaluator initialized with results dir: {results_dir}")
    
    def evaluate_model(self, 
                      model_path: str,
                      test_dataset: str,
                      model_name: str = None) -> Dict:
        """
        Comprehensive evaluation of a trained model.
        
        Args:
            model_path: Path to the trained model
            test_dataset: Path to test dataset configuration
            model_name: Name for the model (for reporting)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if model_name is None:
            model_name = Path(model_path).stem
        
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=test_dataset,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                save_json=True,
                plots=True
            )
            
            # Extract comprehensive metrics
            metrics = self._extract_detailed_metrics(results, model_name)
            
            # Generate detailed report
            self._generate_evaluation_report(metrics, model_name)
            
            # Create visualizations
            self._create_evaluation_plots(results, model_name)
            
            logger.info(f"Evaluation completed for {model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _extract_detailed_metrics(self, results, model_name: str) -> Dict:
        """Extract detailed metrics from validation results."""
        try:
            box_metrics = results.box
            
            metrics = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'overall_metrics': {
                    'mAP50': float(box_metrics.map50),
                    'mAP50-95': float(box_metrics.map),
                    'precision': float(box_metrics.mp),
                    'recall': float(box_metrics.mr),
                    'f1_score': float(2 * (box_metrics.mp * box_metrics.mr) / (box_metrics.mp + box_metrics.mr)) if (box_metrics.mp + box_metrics.mr) > 0 else 0.0
                },
                'per_class_metrics': {},
                'speed_metrics': {
                    'preprocess_time': float(results.speed.get('preprocess', 0)),
                    'inference_time': float(results.speed.get('inference', 0)),
                    'postprocess_time': float(results.speed.get('postprocess', 0))
                }
            }
            
            # Per-class metrics if available
            if hasattr(box_metrics, 'ap_class_index') and hasattr(box_metrics, 'ap'):
                class_names = results.names
                for i, class_idx in enumerate(box_metrics.ap_class_index):
                    class_name = class_names.get(class_idx, f'class_{class_idx}')
                    metrics['per_class_metrics'][class_name] = {
                        'mAP50': float(box_metrics.ap50[i]) if i < len(box_metrics.ap50) else 0.0,
                        'mAP50-95': float(box_metrics.ap[i]) if i < len(box_metrics.ap) else 0.0,
                        'precision': float(box_metrics.p[i]) if i < len(box_metrics.p) else 0.0,
                        'recall': float(box_metrics.r[i]) if i < len(box_metrics.r) else 0.0
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {str(e)}")
            return {'error': str(e)}
    
    def _generate_evaluation_report(self, metrics: Dict, model_name: str):
        """Generate a detailed evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"{model_name}_evaluation_{timestamp}.json"
        
        # Save detailed metrics
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate human-readable report
        txt_report_path = self.reports_dir / f"{model_name}_report_{timestamp}.txt"
        
        with open(txt_report_path, 'w') as f:
            f.write(f"Model Evaluation Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Evaluation Date: {metrics.get('timestamp', 'Unknown')}\n\n")
            
            # Overall metrics
            f.write("Overall Performance:\n")
            f.write("-" * 20 + "\n")
            overall = metrics.get('overall_metrics', {})
            for metric, value in overall.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            # Speed metrics
            f.write("\nSpeed Performance:\n")
            f.write("-" * 20 + "\n")
            speed = metrics.get('speed_metrics', {})
            for metric, value in speed.items():
                f.write(f"{metric}: {value:.2f}ms\n")
            
            # Per-class metrics
            per_class = metrics.get('per_class_metrics', {})
            if per_class:
                f.write("\nPer-Class Performance:\n")
                f.write("-" * 25 + "\n")
                for class_name, class_metrics in per_class.items():
                    f.write(f"\n{class_name}:\n")
                    for metric, value in class_metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Evaluation reports saved to {self.reports_dir}")
    
    def _create_evaluation_plots(self, results, model_name: str):
        """Create visualization plots for evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Set up matplotlib
            plt.style.use('default')
            
            # Plot 1: mAP scores
            self._plot_map_scores(results, model_name, timestamp)
            
            # Plot 2: Precision-Recall curve
            self._plot_precision_recall(results, model_name, timestamp)
            
            # Plot 3: Class-wise performance
            self._plot_class_performance(results, model_name, timestamp)
            
            logger.info(f"Evaluation plots saved to {self.plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
    
    def _plot_map_scores(self, results, model_name: str, timestamp: str):
        """Plot mAP scores."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Overall mAP
            map_values = [results.box.map50, results.box.map]
            map_labels = ['mAP@0.5', 'mAP@0.5:0.95']
            
            ax1.bar(map_labels, map_values, color=['skyblue', 'lightcoral'])
            ax1.set_title(f'{model_name} - Overall mAP Scores')
            ax1.set_ylabel('mAP Score')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(map_values):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Per-class mAP (if available)
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
                class_names = [results.names.get(idx, f'class_{idx}') 
                              for idx in results.box.ap_class_index]
                ap50_values = results.box.ap50
                
                if len(class_names) > 0:
                    ax2.bar(range(len(class_names)), ap50_values, color='lightgreen')
                    ax2.set_title(f'{model_name} - Per-Class mAP@0.5')
                    ax2.set_ylabel('mAP@0.5')
                    ax2.set_xlabel('Classes')
                    ax2.set_xticks(range(len(class_names)))
                    ax2.set_xticklabels(class_names, rotation=45, ha='right')
                    ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = self.plots_dir / f"{model_name}_map_scores_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting mAP scores: {str(e)}")
    
    def _plot_precision_recall(self, results, model_name: str, timestamp: str):
        """Plot precision-recall metrics."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Overall P-R
            precision = results.box.mp
            recall = results.box.mr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = [precision, recall, f1]
            labels = ['Precision', 'Recall', 'F1-Score']
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            
            bars = ax.bar(labels, metrics, color=colors)
            ax.set_title(f'{model_name} - Precision, Recall, and F1-Score')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, metric in zip(bars, metrics):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{metric:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.plots_dir / f"{model_name}_precision_recall_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall: {str(e)}")
    
    def _plot_class_performance(self, results, model_name: str, timestamp: str):
        """Plot per-class performance comparison."""
        try:
            if not (hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'p')):
                return
            
            class_names = [results.names.get(idx, f'class_{idx}') 
                          for idx in results.box.ap_class_index]
            
            if len(class_names) == 0:
                return
            
            # Create DataFrame for easier plotting
            data = {
                'Class': class_names,
                'Precision': results.box.p,
                'Recall': results.box.r,
                'mAP@0.5': results.box.ap50
            }
            
            df = pd.DataFrame(data)
            
            # Create grouped bar chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            x = np.arange(len(class_names))
            width = 0.25
            
            ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
            ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
            ax.bar(x + width, df['mAP@0.5'], width, label='mAP@0.5', alpha=0.8)
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title(f'{model_name} - Per-Class Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = self.plots_dir / f"{model_name}_class_performance_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting class performance: {str(e)}")
    
    def compare_models(self, 
                      model_results: List[Dict],
                      comparison_name: str = "model_comparison") -> Dict:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: List of evaluation result dictionaries
            comparison_name: Name for the comparison
            
        Returns:
            Comparison summary
        """
        if len(model_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comparison DataFrame
        comparison_data = []
        for result in model_results:
            overall = result.get('overall_metrics', {})
            speed = result.get('speed_metrics', {})
            
            row = {
                'Model': result.get('model_name', 'Unknown'),
                'mAP50': overall.get('mAP50', 0),
                'mAP50-95': overall.get('mAP50-95', 0),
                'Precision': overall.get('precision', 0),
                'Recall': overall.get('recall', 0),
                'F1-Score': overall.get('f1_score', 0),
                'Inference Time (ms)': speed.get('inference_time', 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        csv_path = self.comparisons_dir / f"{comparison_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Create comparison visualizations
        self._plot_model_comparison(df, comparison_name, timestamp)
        
        # Generate comparison report
        self._generate_comparison_report(df, comparison_name, timestamp)
        
        # Find best model
        best_model_idx = df['mAP50-95'].idxmax()
        best_model = df.iloc[best_model_idx]['Model']
        
        comparison_summary = {
            'best_model': best_model,
            'best_map': float(df.iloc[best_model_idx]['mAP50-95']),
            'comparison_file': str(csv_path),
            'models_compared': len(model_results)
        }
        
        logger.info(f"Model comparison completed. Best model: {best_model}")
        return comparison_summary
    
    def _plot_model_comparison(self, df: pd.DataFrame, comparison_name: str, timestamp: str):
        """Create comparison plots."""
        try:
            # Metrics comparison
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            models = df['Model']
            
            # mAP comparison
            ax1.bar(models, df['mAP50'], alpha=0.7, label='mAP@0.5')
            ax1.bar(models, df['mAP50-95'], alpha=0.7, label='mAP@0.5:0.95')
            ax1.set_title('mAP Comparison')
            ax1.set_ylabel('mAP Score')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # Precision vs Recall
            ax2.scatter(df['Recall'], df['Precision'], s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax2.annotate(model, (df['Recall'].iloc[i], df['Precision'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision vs Recall')
            ax2.grid(True, alpha=0.3)
            
            # F1-Score comparison
            ax3.bar(models, df['F1-Score'], color='lightgreen', alpha=0.7)
            ax3.set_title('F1-Score Comparison')
            ax3.set_ylabel('F1-Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Speed comparison
            ax4.bar(models, df['Inference Time (ms)'], color='lightcoral', alpha=0.7)
            ax4.set_title('Inference Speed Comparison')
            ax4.set_ylabel('Time (ms)')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = self.comparisons_dir / f"{comparison_name}_comparison_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating comparison plots: {str(e)}")
    
    def _generate_comparison_report(self, df: pd.DataFrame, comparison_name: str, timestamp: str):
        """Generate comparison report."""
        report_path = self.comparisons_dir / f"{comparison_name}_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Model Comparison Report: {comparison_name}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Compared: {len(df)}\n\n")
            
            # Summary table
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Best performers
            f.write("Best Performers:\n")
            f.write("-" * 15 + "\n")
            for metric in ['mAP50-95', 'Precision', 'Recall', 'F1-Score']:
                best_idx = df[metric].idxmax()
                best_model = df.iloc[best_idx]['Model']
                best_value = df.iloc[best_idx][metric]
                f.write(f"Best {metric}: {best_model} ({best_value:.4f})\n")
            
            # Fastest model
            fastest_idx = df['Inference Time (ms)'].idxmin()
            fastest_model = df.iloc[fastest_idx]['Model']
            fastest_time = df.iloc[fastest_idx]['Inference Time (ms)']
            f.write(f"Fastest Inference: {fastest_model} ({fastest_time:.2f}ms)\n")


def main():
    """Example usage of the ModelEvaluator."""
    evaluator = ModelEvaluator()
    
    # Example: Evaluate a model
    # metrics = evaluator.evaluate_model(
    #     model_path="models/custom/surveillance_model.pt",
    #     test_dataset="datasets/surveillance/data.yaml",
    #     model_name="surveillance_v1"
    # )
    
    logger.info("Model evaluator example completed!")


if __name__ == "__main__":
    main()
