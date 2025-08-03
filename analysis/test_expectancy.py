import subprocess
import sys

def test_expectancy_calculator():
    """Test the expectancy calculator output format"""
    csv_path = "analysis/group_backtests/multiasset_1.25_2024/AUD_CAD/AUD_CAD_2024_fixed.csv"
    
    result = subprocess.run([
        sys.executable, "analysis/expectancy_calculator.py", csv_path
    ], capture_output=True, text=True, timeout=60)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")

if __name__ == "__main__":
    test_expectancy_calculator() 