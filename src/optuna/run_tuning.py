#!/usr/bin/env python3
"""
Hyperparameter tuning script for RAG System.
This script runs Optuna optimization for RAG pipeline hyperparameters.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.optuna import RAGHyperparameterTuner
from src.utils import load_config


def main() -> None:
    """Run hyperparameter optimization."""
    try:
        # Load configuration
        config = load_config('conf')
        
        # Check if tuning is enabled
        if not config.optuna.get('enabled', False):
            print('❌ Hyperparameter tuning disabled in config.')
            print('   Enable it by setting optuna.enabled: true in conf/optuna/optuna_default.yaml')
            sys.exit(1)
        
        # Initialize tuner
        tuner = RAGHyperparameterTuner(config)
        
        # Run optimization
        results = tuner.optimize()
        
        if 'error' in results:
            print(f'❌ Tuning failed: {results["error"]}')
            sys.exit(1)
        else:
            print('✅ Hyperparameter tuning completed!')
            print(f'📊 Best score: {results["best_score"]:.4f}')
            print(f'🏆 Best parameters: {results["best_params"]}')
            print(f'🔢 Total trials: {results["n_trials"]}')
            
            # Show results location
            print('')
            print('📁 Results saved in hyperparameter_results/')
            print('💡 View Optuna dashboard with: optuna-dashboard sqlite:///hyperparameter_studies.db')

    except Exception as e:
        print(f'❌ Error: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()